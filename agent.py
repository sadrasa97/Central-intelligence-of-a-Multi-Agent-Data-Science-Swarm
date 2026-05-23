"""
================================================================================
Dr. Data — Multi-Agent Swarm  (MoE Edition)
Intent Router + 8 Specialized Agents + Thinking Suppression
================================================================================
"""

import os
import json
import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# THINKING SUPPRESSION
# Qwen3 / DeepSeek / any <think> model: inject this prefix into assistant turn
# so the model skips internal reasoning and goes straight to the answer.
# ──────────────────────────────────────────────────────────────────────────────
_NO_THINK_ASSISTANT_PREFIX = "<think>\n\n</think>\n\n"

_NO_THINK_SYSTEM_SUFFIX = """

STRICT OUTPUT RULES (NEVER VIOLATE):
- Output ONLY the final answer. No reasoning, no thinking, no chain-of-thought.
- Do NOT use <think>, <reasoning>, or any internal monologue tags.
- Do NOT explain how you arrived at the answer.
- Be direct, concise, and professional.
"""


# ──────────────────────────────────────────────────────────────────────────────
# MODEL DOWNLOAD / FALLBACK
# ──────────────────────────────────────────────────────────────────────────────
def download_gguf_from_huggingface(model_path: str) -> bool:
    from pathlib import Path
    import urllib.request
    import urllib.error

    mp = Path(model_path)
    if mp.exists() and mp.stat().st_size > 0:
        return True

    mp.parent.mkdir(parents=True, exist_ok=True)

    fallbacks = [
        ("Qwen/Qwen3.5-2B-GGUF", "qwen3-5-2b-q4_k_s.gguf"),
        ("Qwen/Qwen3.5-1.5B-GGUF", "qwen3-5-1.5b-q4_k_s.gguf"),
    ]

    filename = mp.name.lower()
    known_patterns = {
        "qwen":    ("Qwen/Qwen3.5-2B-GGUF",                          "qwen3-5-2b-q4_k_s.gguf"),
        "gemma":   ("google/gemma-2-2b-it-GGUF",                     "gemma-2-2b-it-Q4_K_M.gguf"),
        "llama":   ("QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",    "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"),
        "mistral": ("QuantFactory/Mistral-7B-Instruct-v0.3-GGUF",    "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"),
        "phi":     ("microsoft/Phi-3-mini-4k-instruct-gguf",         "Phi-3-mini-4k-instruct-q4.gguf"),
    }

    repo_id, hf_filename = fallbacks[0]
    for key, (rid, hf_fn) in known_patterns.items():
        if key in filename:
            repo_id, hf_filename = rid, hf_fn
            break

    try:
        logger.info(f"Downloading GGUF: {repo_id}/{hf_filename} → {model_path}")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=repo_id, filename=hf_filename,
                local_dir=str(mp.parent), local_dir_use_symlinks=False,
                resume_download=True,
            )
            from pathlib import Path as P
            if P(downloaded) != mp:
                import shutil; shutil.move(downloaded, str(mp))
            return True
        except ImportError:
            import shutil, urllib.request
            hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{hf_filename}"
            req = urllib.request.Request(hf_url, headers={"User-Agent": "DrDataAgent/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                with open(str(mp) + ".tmp", "wb") as out:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk: break
                        out.write(chunk)
            shutil.move(str(mp) + ".tmp", str(mp))
            return True
    except Exception as e:
        logger.error(f"GGUF download failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# LLM PROVIDERS
# ──────────────────────────────────────────────────────────────────────────────
class BaseLLMProvider:
    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 stop: List[str] = None) -> str:
        raise NotImplementedError


class LlamaCppGGUFProvider(BaseLLMProvider):
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = None):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count() or 4
        self._model = None

    def _load_model(self):
        from pathlib import Path
        mp = Path(self.model_path)
        if not mp.exists():
            if not download_gguf_from_huggingface(self.model_path):
                raise RuntimeError(f"Model not found: {self.model_path}")
        try:
            from llama_cpp import Llama
            self._model = Llama(
                model_path=str(mp), n_ctx=self.n_ctx,
                n_threads=self.n_threads, verbose=False,
                n_batch=256, use_mmap=True, use_mlock=False,
            )
            logger.info(f"Model loaded: {mp.name}")
        except ImportError:
            raise ImportError("Run: pip install llama-cpp-python")

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 stop: List[str] = None) -> str:
        try:
            if self._model is None:
                self._load_model()
            stop_tokens = stop or ["<|im_end|>", "<|endoftext|>", "Human:", "User:"]
            resp = self._model(prompt, max_tokens=max_tokens, temperature=temperature,
                               top_p=top_p, stop=stop_tokens, echo=False)
            return str(resp["choices"][0]["text"]).strip()
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"[Model error: {str(e)}]"


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 stop: List[str] = None) -> str:
        import urllib.request, urllib.error
        payload = {"model": self.model_name, "prompt": prompt, "stream": False,
                   "options": {"temperature": temperature, "top_p": top_p,
                               "num_predict": int(max_tokens)}}
        if stop:
            payload["options"]["stop"] = stop
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/api/generate", data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return str(json.loads(resp.read().decode()).get("response", "")).strip()
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")


class OpenAIChatGPTProvider(BaseLLMProvider):
    def __init__(self, api_base: str, model_name: str, api_key: str):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 stop: List[str] = None) -> str:
        import urllib.request, urllib.error
        payload = {"model": self.model_name,
                   "messages": [{"role": "system", "content": "You are Dr. Data, a data science assistant."},
                                 {"role": "user", "content": prompt}],
                   "temperature": temperature, "top_p": top_p, "max_tokens": int(max_tokens)}
        if stop:
            payload["stop"] = stop
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"{self.api_base}/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return str(json.loads(resp.read().decode())["choices"][0]["message"]["content"]).strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI error: {e}")


def build_provider(provider_cfg: Dict[str, Any]) -> BaseLLMProvider:
    ptype = (provider_cfg.get("type") or "llamacpp").lower()
    if ptype in ["llamacpp", "llama-cpp", "gguf", "llama_cpp"]:
        model_path = provider_cfg.get("model_path") or provider_cfg.get("gguf_path") or ""
        if not model_path:
            raise ValueError("Missing model_path for llamacpp provider.")
        return LlamaCppGGUFProvider(model_path=model_path,
                                    n_ctx=int(provider_cfg.get("n_ctx", 2048)),
                                    n_threads=provider_cfg.get("n_threads"))
    if ptype == "ollama":
        return OllamaProvider(
            base_url=provider_cfg.get("ollama_base_url") or "http://localhost:11434",
            model_name=provider_cfg.get("ollama_model") or "llama3")
    if ptype in ["openai", "chatgpt"]:
        api_key = provider_cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY") or ""
        if not api_key:
            raise ValueError("Missing openai_api_key.")
        return OpenAIChatGPTProvider(
            api_base=provider_cfg.get("openai_api_base") or "https://api.openai.com",
            model_name=provider_cfg.get("openai_model") or "gpt-4o-mini",
            api_key=api_key)
    raise ValueError(f"Unsupported provider: {ptype}")


# ──────────────────────────────────────────────────────────────────────────────
# INTENT TYPES  (MoE routing targets)
# ──────────────────────────────────────────────────────────────────────────────
class Intent(Enum):
    CASUAL_CHAT   = "casual_chat"
    DATA_QA       = "data_qa"
    EDA           = "eda"
    PREPROCESSING = "preprocessing"
    ML_TRAINING   = "ml_training"
    DEEP_LEARNING = "deep_learning"
    VISUALIZATION = "visualization"
    PREDICTION    = "prediction"
    REPORT        = "report"
    WEB_SEARCH    = "web_search"


# keyword → intent (fast path, no LLM call)
_INTENT_KEYWORDS: Dict[Intent, List[str]] = {
    Intent.WEB_SEARCH:    ["search", "find online", "look up", "latest news", "جستجو", "سرچ",
                           "اینترنت", "find info", "بیا پیدا کن", "تحقیق کن روی"],
    Intent.EDA:           ["eda", "exploratory", "analyze", "statistics", "distribution",
                           "correlation", "آنالیز", "تحلیل", "آمار", "بررسی داده"],
    Intent.PREPROCESSING: ["clean", "preprocess", "missing", "outlier", "encode", "scale",
                           "پیش‌پردازش", "تمیز", "مقادیر گمشده", "نرمال"],
    Intent.ML_TRAINING:   ["train", "model", "random forest", "xgboost", "classification",
                           "regression", "آموزش مدل", "یادگیری ماشین", "دقت مدل"],
    Intent.DEEP_LEARNING: ["neural", "ann", "deep learning", "keras", "epoch",
                           "شبکه عصبی", "یادگیری عمیق"],
    Intent.VISUALIZATION: ["chart", "plot", "graph", "visualize", "heatmap", "scatter",
                           "نمودار", "رسم", "چارت", "ویژوالیز"],
    Intent.PREDICTION:    ["predict", "inference", "forecast", "پیش‌بینی", "پیشگویی"],
    Intent.REPORT:        ["report", "summary", "export", "گزارش", "خلاصه"],
    Intent.DATA_QA:       ["column", "row", "dataset", "data", "value", "ستون", "ردیف",
                           "داده", "فایل", "دیتاست"],
}


class IntentRouter:
    """
    Lightweight MoE intent classifier.
    Step 1: keyword match (zero LLM calls, ~1ms).
    Step 2: if ambiguous, ask LLM with a tiny prompt.
    """

    def classify(self, message: str, has_session: bool = False,
                 llm: BaseLLMProvider = None) -> Intent:
        msg_lower = message.lower()

        # Fast keyword scan (priority order)
        for intent in [Intent.WEB_SEARCH, Intent.EDA, Intent.PREPROCESSING,
                       Intent.ML_TRAINING, Intent.DEEP_LEARNING, Intent.VISUALIZATION,
                       Intent.PREDICTION, Intent.REPORT, Intent.DATA_QA]:
            for kw in _INTENT_KEYWORDS[intent]:
                if kw in msg_lower:
                    logger.info(f"[IntentRouter] keyword match → {intent.value} ('{kw}')")
                    return intent

        # If user has a dataset loaded and message is question-like, assume DATA_QA
        if has_session and any(q in msg_lower for q in ["?", "what", "how many", "show", "چند", "چیست", "چه"]):
            return Intent.DATA_QA

        # LLM fallback (only if provider available)
        if llm:
            return self._llm_classify(message, llm)

        # Default
        return Intent.CASUAL_CHAT

    def _llm_classify(self, message: str, llm: BaseLLMProvider) -> Intent:
        options = ", ".join(i.value for i in Intent)
        prompt = (
            f"<|im_start|>system\nClassify the user message into exactly one intent. "
            f"Output ONLY the intent name, nothing else.\n"
            f"Options: {options}\n<|im_end|>\n"
            f"<|im_start|>user\n{message[:200]}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{_NO_THINK_ASSISTANT_PREFIX}"
        )
        try:
            raw = llm.generate(prompt, max_tokens=12, temperature=0.1)
            raw = raw.strip().lower().replace("-", "_")
            for intent in Intent:
                if intent.value in raw:
                    logger.info(f"[IntentRouter] LLM classified → {intent.value}")
                    return intent
        except Exception as e:
            logger.warning(f"IntentRouter LLM fallback failed: {e}")
        return Intent.CASUAL_CHAT


# ──────────────────────────────────────────────────────────────────────────────
# AGENT ROLES
# ──────────────────────────────────────────────────────────────────────────────
class AgentRole(Enum):
    ORCHESTRATOR  = "orchestrator"
    DATA_ANALYST  = "data_analyst"
    PREPROCESSOR  = "preprocessor"
    ML_ENGINEER   = "ml_engineer"
    DEEP_LEARNING = "deep_learning"
    VISUALIZER    = "visualizer"
    PREDICTOR     = "predictor"
    REPORTER      = "reporter"
    CHAT          = "chat"


@dataclass
class AgentMessage:
    role: str
    content: str
    agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# BASE AGENT
# ──────────────────────────────────────────────────────────────────────────────
class BaseAgent:
    ROLE: AgentRole = AgentRole.ORCHESTRATOR
    SYSTEM_PROMPT: str = "You are a helpful AI assistant."
    MAX_TOKENS: int = 300

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.memory: List[AgentMessage] = []

    def _build_prompt(self, user_input: str, context: str = "") -> str:
        history = "".join(
            f"<|im_start|>{m.role}\n{m.content}\n<|im_end|>\n"
            for m in self.memory[-4:]
        )
        ctx_block = f"\n[DATASET CONTEXT]\n{context}\n" if context else ""
        system = self.SYSTEM_PROMPT + _NO_THINK_SYSTEM_SUFFIX + ctx_block
        return (
            f"<|im_start|>system\n{system}\n<|im_end|>\n"
            f"{history}"
            f"<|im_start|>user\n{user_input}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{_NO_THINK_ASSISTANT_PREFIX}"
        )

    def run(self, user_input: str, context: str = "", temperature: float = 0.6) -> str:
        try:
            prompt = self._build_prompt(user_input, context)
            response = self.llm.generate(prompt, temperature=temperature,
                                         max_tokens=self.MAX_TOKENS)
            # Strip any leaked <think> blocks that slipped through
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            self.memory.append(AgentMessage("user", user_input, self.ROLE.value))
            self.memory.append(AgentMessage("assistant", response, self.ROLE.value))
            return response
        except Exception as e:
            logger.error(f"Agent {self.ROLE.value} error: {e}")
            return f"[{self.ROLE.value} agent error: {str(e)}]"


# ──────────────────────────────────────────────────────────────────────────────
# SPECIALIZED AGENTS
# ──────────────────────────────────────────────────────────────────────────────
class ChatAgent(BaseAgent):
    ROLE = AgentRole.CHAT
    MAX_TOKENS = 200
    SYSTEM_PROMPT = (
        "You are Dr. Data, a friendly AI assistant specialized in data science. "
        "Answer general questions concisely. If the user has a dataset loaded, "
        "you can refer to it. Support both Persian and English."
    )


class OrchestratorAgent(BaseAgent):
    ROLE = AgentRole.ORCHESTRATOR
    MAX_TOKENS = 256
    SYSTEM_PROMPT = (
        "Orchestrator of Dr. Data swarm. Decompose user requests into sub-tasks.\n"
        "Available agents: data_analyst, preprocessor, ml_engineer, deep_learning, "
        "visualizer, predictor, reporter.\n"
        "Output ONLY valid JSON: "
        '{\"tasks\":[{\"agent\":\"name\",\"task\":\"description\",\"priority\":1}],'
        '\"summary\":\"brief\"}'
    )

    def decompose(self, user_input: str, context: str = "") -> List[Dict[str, Any]]:
        prompt = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}{_NO_THINK_SYSTEM_SUFFIX}\n<|im_end|>\n"
            f"<|im_start|>user\nDecompose: \"{user_input[:300]}\"\n<|im_end|>\n"
            f"<|im_start|>assistant\n{_NO_THINK_ASSISTANT_PREFIX}"
        )
        try:
            raw = self.llm.generate(prompt, temperature=0.2, max_tokens=256)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                plan = json.loads(match.group())
                return plan.get("tasks", [])
        except Exception as e:
            logger.warning(f"Orchestrator decompose failed: {e}")
        return [{"agent": "data_analyst", "task": user_input, "priority": 1}]


class DataAnalystAgent(BaseAgent):
    ROLE = AgentRole.DATA_ANALYST
    MAX_TOKENS = 350
    SYSTEM_PROMPT = (
        "You are the Data Analyst of Dr. Data platform. "
        "Analyze datasets: statistics, distributions, correlations, data quality, missing values, PII, imbalance. "
        "Be precise and professional. Use Persian for explanations, English for technical terms."
    )


class PreprocessorAgent(BaseAgent):
    ROLE = AgentRole.PREPROCESSOR
    MAX_TOKENS = 300
    SYSTEM_PROMPT = (
        "You are the Preprocessor of Dr. Data. "
        "Handle: missing values (mean/median/KNN), outliers (IQR/Z-score), "
        "encoding (label/one-hot/target), scaling (standard/minmax/robust), feature engineering. "
        "Always fit on train set only. Be concise and precise."
    )


class MLEngineerAgent(BaseAgent):
    ROLE = AgentRole.ML_ENGINEER
    MAX_TOKENS = 350
    SYSTEM_PROMPT = (
        "You are the ML Engineer of Dr. Data. "
        "Train and compare: RF, XGBoost, LightGBM, Linear, SVM, KNN, Naive Bayes, clustering. "
        "Auto-detect problem type, cross-validate, report CV mean±std, detect overfitting. "
        "Be concise with exact numbers."
    )


class DeepLearningAgent(BaseAgent):
    ROLE = AgentRole.DEEP_LEARNING
    MAX_TOKENS = 300
    SYSTEM_PROMPT = (
        "You are the Deep Learning Engineer of Dr. Data. "
        "Design and train ANNs/MLPs: BatchNorm, Dropout, AdamW, early stopping, learning rate scheduling. "
        "Auto-scale architecture by input size. Warn if dataset < 500 samples."
    )


class VisualizerAgent(BaseAgent):
    ROLE = AgentRole.VISUALIZER
    MAX_TOKENS = 300
    SYSTEM_PROMPT = (
        "You are the Visualizer of Dr. Data. "
        "Recommend and describe charts: histogram, KDE, boxplot, scatter, heatmap, bar, ROC, confusion matrix. "
        "Be colorblind-friendly. Use Persian labels for Persian data."
    )


class PredictorAgent(BaseAgent):
    ROLE = AgentRole.PREDICTOR
    MAX_TOKENS = 250
    SYSTEM_PROMPT = (
        "You are the Predictor of Dr. Data. "
        "Handle single/batch inference, confidence intervals, input validation, drift detection. "
        "Be precise with predicted values and confidence scores."
    )


class ReporterAgent(BaseAgent):
    ROLE = AgentRole.REPORTER
    MAX_TOKENS = 400
    SYSTEM_PROMPT = (
        "You are the Reporter of Dr. Data. "
        "Write concise professional reports: executive summary, methodology, results, recommendations. "
        "Persian for business insights, English for metrics and technical terms."
    )


# ──────────────────────────────────────────────────────────────────────────────
# MoE AGENT SWARM
# ──────────────────────────────────────────────────────────────────────────────

# Intent → which agents to activate (Mixture-of-Experts routing table)
_INTENT_TO_AGENTS: Dict[Intent, List[AgentRole]] = {
    Intent.CASUAL_CHAT:   [AgentRole.CHAT],
    Intent.DATA_QA:       [AgentRole.DATA_ANALYST],
    Intent.EDA:           [AgentRole.DATA_ANALYST, AgentRole.VISUALIZER],
    Intent.PREPROCESSING: [AgentRole.PREPROCESSOR],
    Intent.ML_TRAINING:   [AgentRole.ML_ENGINEER],
    Intent.DEEP_LEARNING: [AgentRole.DEEP_LEARNING],
    Intent.VISUALIZATION: [AgentRole.VISUALIZER],
    Intent.PREDICTION:    [AgentRole.PREDICTOR],
    Intent.REPORT:        [AgentRole.REPORTER, AgentRole.DATA_ANALYST],
    Intent.WEB_SEARCH:    [AgentRole.DATA_ANALYST],   # used after web results injected
}


class AgentSwarm:
    """MoE swarm: classify intent → route to relevant agents only → synthesize."""

    def __init__(self, provider_cfg: Dict[str, Any]):
        self.llm = build_provider(provider_cfg)
        self.router = IntentRouter()
        self.agents: Dict[AgentRole, BaseAgent] = {
            AgentRole.ORCHESTRATOR:  OrchestratorAgent(self.llm),
            AgentRole.CHAT:          ChatAgent(self.llm),
            AgentRole.DATA_ANALYST:  DataAnalystAgent(self.llm),
            AgentRole.PREPROCESSOR:  PreprocessorAgent(self.llm),
            AgentRole.ML_ENGINEER:   MLEngineerAgent(self.llm),
            AgentRole.DEEP_LEARNING: DeepLearningAgent(self.llm),
            AgentRole.VISUALIZER:    VisualizerAgent(self.llm),
            AgentRole.PREDICTOR:     PredictorAgent(self.llm),
            AgentRole.REPORTER:      ReporterAgent(self.llm),
        }
        self.execution_log: List[Dict] = []

    # ── public entry point ────────────────────────────────────────────────────
    def execute(self, user_input: str, session_data: Dict = None,
                web_search_results: str = None) -> Dict[str, Any]:

        has_session = bool(session_data and session_data.get("columns"))
        context = self._build_context(session_data, web_search_results)

        # ── Step 1: Intent classification (MoE router) ───────────────────────
        intent = self.router.classify(user_input, has_session=has_session, llm=self.llm)

        # Force WEB_SEARCH intent if results were injected
        if web_search_results:
            intent = Intent.WEB_SEARCH

        logger.info(f"[AgentSwarm] intent={intent.value} | session={has_session}")

        # ── Step 2: Activate only relevant agents ────────────────────────────
        active_roles = _INTENT_TO_AGENTS.get(intent, [AgentRole.CHAT])
        agent_outputs: Dict[str, Any] = {}

        for role in active_roles:
            agent = self.agents[role]
            try:
                output = agent.run(user_input, context, temperature=0.6)
            except Exception as e:
                logger.error(f"Agent {role.value} error: {e}")
                output = f"[{role.value} error: {str(e)}]"

            agent_outputs[role.value] = {"task": user_input, "output": output}
            self.execution_log.append({
                "agent": role.value, "intent": intent.value,
                "timestamp": datetime.now().isoformat()
            })

        # ── Step 3: Synthesize if multiple agents, else pass through ─────────
        if len(agent_outputs) == 1:
            synthesis = next(iter(agent_outputs.values()))["output"]
        else:
            synthesis = self._synthesize(user_input, agent_outputs, context)

        return {
            "synthesis": synthesis,
            "intent": intent.value,
            "agent_outputs": agent_outputs,
            "active_agents": [r.value for r in active_roles],
            "execution_plan": [{"agent": r.value, "intent": intent.value} for r in active_roles],
            "context": context,
        }

    # ── synthesis ─────────────────────────────────────────────────────────────
    def _synthesize(self, user_input: str, agent_outputs: Dict, context: str) -> str:
        outputs_summary = {
            k: v.get("output", "")[:400]
            for k, v in agent_outputs.items()
        }
        prompt = (
            f"<|im_start|>system\n"
            f"Synthesize the following agent outputs into ONE concise professional answer. "
            f"No repetition. No internal reasoning. Direct answer only.{_NO_THINK_SYSTEM_SUFFIX}\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"User request: {user_input}\n\n"
            f"Agent outputs:\n{json.dumps(outputs_summary, ensure_ascii=False)}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n{_NO_THINK_ASSISTANT_PREFIX}"
        )
        try:
            result = self.llm.generate(prompt, temperature=0.5, max_tokens=350)
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            return result
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            parts = [f"**{name.title()}:** {data.get('output', '')[:300]}"
                     for name, data in agent_outputs.items()]
            return "\n\n".join(parts)

    # ── context builder ───────────────────────────────────────────────────────
    def _build_context(self, session_data: Dict, web_search: str = None) -> str:
        ctx = ""
        if session_data:
            ctx += (
                f"=== DATASET ===\n"
                f"File: {session_data.get('filename', 'N/A')}\n"
                f"Shape: {session_data.get('shape', 'N/A')}\n"
                f"Columns: {session_data.get('columns', [])}\n"
                f"Numeric: {session_data.get('numeric_columns', [])}\n"
                f"Categorical: {session_data.get('categorical_columns', [])}\n"
                f"Health: {session_data.get('health_score', 'N/A')}/100\n"
                f"Issues: {session_data.get('health_issues', [])}\n"
                f"PII: {session_data.get('pii_detected', [])}\n"
            )
        if web_search:
            ctx += f"\n=== WEB SEARCH RESULTS ===\n{web_search}\n"
        return ctx


# ──────────────────────────────────────────────────────────────────────────────
# LEGACY COMPATIBILITY WRAPPERS
# ──────────────────────────────────────────────────────────────────────────────
class DataScientistAgent:
    def __init__(self, model_path: str):
        self.swarm = AgentSwarm({"type": "llamacpp", "model_path": model_path})

    def chat(self, message: str, data_context: str = "", temperature: float = 0.7) -> str:
        return self.swarm.execute(message, {"context": data_context})["synthesis"]

    def analyze_dataset(self, metadata: Dict[str, Any]) -> str:
        return self.swarm.execute("Analyze this dataset comprehensively", metadata)["synthesis"]

    def web_search_enhanced_response(self, query: str, data_context: str, search_results: str) -> str:
        return self.swarm.execute(query, {"context": data_context}, search_results)["synthesis"]

    def generate_report(self, session_data: Dict, analysis_history: List[Dict]) -> str:
        return self.swarm.execute("Generate comprehensive analysis report", session_data)["synthesis"]


class AgentOrchestrator:
    def __init__(self, model_path: str):
        self.swarm = AgentSwarm({"type": "llamacpp", "model_path": model_path})

    def process_request(self, user_input: str, session_data: Dict = None) -> Dict[str, Any]:
        result = self.swarm.execute(user_input, session_data)
        return {"response": result["synthesis"], "actions": [],
                "requires_tool": False, "agent_outputs": result["agent_outputs"],
                "plan": result["execution_plan"]}


# ──────────────────────────────────────────────────────────────────────────────
# TOOL SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────
TOOLS_SCHEMA = [
    {"name": "load_data", "description": "Load CSV/XLSX into analysis environment",
     "parameters": {"type": "object", "properties": {
         "file_path": {"type": "string"}, "file_type": {"type": "string", "enum": ["csv", "xlsx"]}},
         "required": ["file_path"]}},
    {"name": "run_eda", "description": "Run comprehensive EDA with visualizations",
     "parameters": {"type": "object", "properties": {"columns": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "train_ml_model", "description": "Train ML models (RF, XGBoost, Linear)",
     "parameters": {"type": "object", "properties": {
         "target_column": {"type": "string"},
         "model_type": {"type": "string", "enum": ["linear", "random_forest", "xgboost", "all"]}},
         "required": ["target_column"]}},
    {"name": "search_web", "description": "Search web for domain-specific information",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]


if __name__ == "__main__":
    import sys
    model_path = os.environ.get("MODEL_PATH", "./models/qwen3.5-2b-q4_k_s.gguf")
    provider_cfg = {
        "type": "llamacpp",
        "model_path": model_path,
        "n_ctx": int(os.environ.get("N_CTX", 1024)),
        "n_threads": int(os.environ.get("N_THREADS", os.cpu_count() or 8)),
    }
    swarm = AgentSwarm(provider_cfg)

    test_queries = [
        "سلام، چطوری؟",
        "ستون‌های دیتاست رو نشونم بده",
        "search for best practices in customer churn prediction",
        "آموزش مدل random forest روی داده‌ها",
    ]
    for q in test_queries:
        print(f"\n{'='*60}\nQuery: {q}")
        result = swarm.execute(q)
        print(f"Intent: {result['intent']}")
        print(f"Agents: {result['active_agents']}")
        print(f"Response: {result['synthesis'][:200]}")