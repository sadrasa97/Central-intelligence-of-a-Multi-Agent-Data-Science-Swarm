"""
================================================================================

Dr. Data — Multi-Agent Data Science Platform v2.1
FastAPI + 7-Agent Swarm + ANN + ML + Clustering + Prediction
================================================================================
"""

import os
import sys
import json
import base64
import io
import warnings
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

# Setup logging
from config import setup_logging, Config, ensure_model_available
setup_logging()
logger = logging.getLogger(__name__)

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
logger.info(f"Base directory: {BASE_DIR}")


def _resolve_gguf_path(model_path: str) -> str:
    """Resolve relative gguf paths to absolute paths and auto-detect/download if missing."""
    p = Path(model_path)

    # If caller passed relative path (like .\models\xxx.gguf), anchor to BASE_DIR.
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()

    # Check if model exists
    success, message = ensure_model_available(str(p), auto_download=True)
    
    if success:
        logger.info(message)
        return str(p)
    
    # If still not found, try auto-detect in BASE_DIR/models/
    models_dir = (BASE_DIR / "models").resolve()
    if models_dir.exists():
        candidates = sorted(models_dir.glob("*.gguf"))
        if len(candidates) >= 1:
            logger.warning(f"Using fallback model: {candidates[0]}")
            return str(candidates[0])
    
    # Last resort: log warning but return the path anyway
    logger.warning(message)
    logger.warning("You can set a custom model path via MODEL_PATH environment variable or /chat endpoint")
    return str(p)


MODEL_PATH = _resolve_gguf_path(os.environ.get("MODEL_PATH", Config.MODEL_PATH))

# Use config values
UPLOAD_DIR = str(Config.UPLOAD_DIR)
OUTPUT_DIR = str(Config.OUTPUT_DIR)
CHARTS_DIR = str(Config.CHARTS_DIR)
MODELS_DIR = str(Config.MODELS_DIR)
REPORTS_DIR = str(Config.REPORTS_DIR)

logger.info(f"Model path: {MODEL_PATH}")
logger.info(f"Upload directory: {UPLOAD_DIR}")
logger.info(f"Output directory: {OUTPUT_DIR}")



# ==============================================================================
# DATA SESSION MANAGER
# ==============================================================================
class DataSession:
    """Manages uploaded datasets with full metadata and model registry"""

    def __init__(self, session_id: str, filepath: str, filename: str):
        self.session_id = session_id
        self.filepath = filepath
        self.filename = filename
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        self.analysis_history: List[Dict] = []
        self.model_registry: Dict[str, Any] = {}
        self.ann_model_path: Optional[str] = None
        self.ann_metadata: Optional[Dict] = None
        self.scaler: Optional[Any] = None
        self.label_encoders: Dict[str, Any] = {}
        self.feature_columns: List[str] = []
        self.target_column: Optional[str] = None
        self.model_results: Dict[str, Any] = {}
        self.created_at = datetime.now().isoformat()
        self._load_data()

    def _load_data(self):
        try:
            logger.info(f"Loading data from {self.filepath}")
            if self.filename.endswith('.csv'):
                self.df = pd.read_csv(self.filepath, low_memory=False)
            elif self.filename.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(self.filepath)
            elif self.filename.endswith('.parquet'):
                self.df = pd.read_parquet(self.filepath)
            elif self.filename.endswith('.json'):
                self.df = pd.read_json(self.filepath)
            else:
                raise ValueError("Unsupported file format. Use CSV, XLSX, Parquet, or JSON.")

            self.original_df = self.df.copy()
            self._compute_metadata()
            logger.info(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise ValueError(f"Failed to load data: {str(e)}")

    def _compute_metadata(self):
        df = self.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        for col in cat_cols[:]:
            try:
                pd.to_datetime(df[col], errors='raise')
                datetime_cols.append(col)
                cat_cols.remove(col)
            except:
                pass

        self.metadata = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_columns": numeric_cols,
            "categorical_columns": cat_cols,
            "datetime_columns": datetime_cols,
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "numeric_summary": df[numeric_cols].describe().to_dict() if numeric_cols else {},
            "categorical_summary": {col: {
                "unique": int(df[col].nunique()),
                "top": str(df[col].mode()[0]) if not df[col].mode().empty else None,
                "freq": int(df[col].value_counts().iloc[0]) if len(df[col].value_counts()) > 0 else 0
            } for col in cat_cols},
        }

        pii_patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b(?:\d[ -]*?){13,16}\b'
        }
        pii_detected = []
        for col in df.columns:
            for pii_type, pattern in pii_patterns.items():
                if df[col].astype(str).str.match(pattern, na=False).any():
                    pii_detected.append(f"🔒 {col}: Potential {pii_type.upper()} detected")

        self.metadata["pii_detected"] = pii_detected

        health_issues = []
        for col, pct in self.metadata["missing_percentage"].items():
            if pct > 50:
                health_issues.append(f"🚨 {col}: {pct}% missing — CRITICAL")
            elif pct > 20:
                health_issues.append(f"⚠️ {col}: {pct}% missing — HIGH")
            elif pct > 5:
                health_issues.append(f"ℹ️ {col}: {pct}% missing — MODERATE")

        if self.metadata["duplicate_rows"] > 0:
            health_issues.append(f"⚠️ {self.metadata['duplicate_rows']} duplicate rows detected")

        for col in cat_cols:
            if df[col].nunique() <= 10:
                vc = df[col].value_counts(normalize=True)
                if vc.max() > 0.9:
                    health_issues.append(f"⚠️ {col}: Severe class imbalance (top class {vc.max():.1%})")

        self.metadata["health_issues"] = health_issues
        self.metadata["health_score"] = max(0, 100 - len(health_issues) * 8)

sessions: Dict[str, DataSession] = {}

# ==============================================================================
# FASTAPI APP
# ==============================================================================
app = FastAPI(
    title="Dr. Data — Multi-Agent Data Science Swarm",
    description="Seven specialized AI agents collaborate to ingest, analyze, model, visualize, and predict.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
from fastapi.staticfiles import StaticFiles
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")
    logger.info(f"Static files mounted from {static_dir}")

# ==============================================================================
# HOME ROUTE
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main dashboard HTML"""
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dr. Data Platform</h1><p>index.html not found. Please ensure index.html is in the root directory.</p>")

# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================
class ANNConfig(BaseModel):
    """Advanced Neural Network Configuration"""
    auto_architecture: bool = True  # Auto-detect layers based on data
    hidden_layers: Optional[List[int]] = None  # e.g., [128, 64, 32]
    dropout_rates: Optional[List[float]] = None  # e.g., [0.3, 0.2, 0.1]
    activation: str = "relu"  # relu, tanh, sigmoid, elu, selu
    output_activation: Optional[str] = None  # auto-detect if None
    
    # Optimizer settings
    optimizer: str = "adamw"  # adamw, adam, sgd, rmsprop
    learning_rate: float = 0.001
    weight_decay: float = 0.004  # For AdamW
    
    # Loss function
    loss: Optional[str] = None  # auto-detect if None (mse/mae for regression, categorical_crossentropy for classification)
    
    # Training settings
    epochs: int = 150
    batch_size: Optional[int] = None  # auto-detect if None
    validation_split: float = 0.2
    
    # Regularization
    l2_regularization: float = 0.001
    use_batch_normalization: bool = True
    
    # Callbacks
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 7
    min_learning_rate: float = 1e-7

class MLHyperparameters(BaseModel):
    """Machine Learning Model Hyperparameters"""
    # Random Forest / Gradient Boosting
    n_estimators: int = 200
    max_depth: Optional[int] = 20
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    
    # SVM
    svm_kernel: str = "rbf"  # rbf, linear, poly, sigmoid
    svm_c: float = 1.0
    svm_gamma: str = "scale"  # scale, auto, or float
    
    # KNN
    knn_neighbors: int = 5
    knn_weights: str = "uniform"  # uniform, distance
    knn_metric: str = "minkowski"  # minkowski, euclidean, manhattan
    
    # Ridge/Lasso
    ridge_alpha: float = 1.0
    lasso_alpha: float = 0.1
    
    # Logistic Regression
    logistic_penalty: str = "l2"  # l1, l2, elasticnet, none
    logistic_c: float = 1.0
    logistic_solver: str = "lbfgs"  # lbfgs, liblinear, saga, sag
    
    # XGBoost (if available)
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # LightGBM (if available)
    lgb_num_leaves: int = 31
    lgb_learning_rate: float = 0.1
    lgb_feature_fraction: float = 0.9

class AnalysisRequest(BaseModel):
    session_id: str
    analysis_type: str = "eda"
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    model_type: Optional[str] = "all"  # all, linear, random_forest, xgboost, lightgbm, svm, knn, etc.
    test_size: float = 0.2
    problem_type: Optional[str] = "auto"
    cv_folds: int = 5
    
    # Advanced ML/ANN Controls
    ann_config: Optional[ANNConfig] = None
    ml_hyperparameters: Optional[MLHyperparameters] = None

class ChatRequest(BaseModel):
    session_id: str
    message: str
    search_web: bool = False
    agent_mode: bool = True

    # LLM Provider selection (optional; defaults to local GGUF via llama-cpp-python)
    llm_provider: str = "llamacpp"  # llamacpp | ollama | openai
    gguf_model_path: Optional[str] = None

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # OpenAI (Chat Completions)
    openai_api_base: str = "https://api.openai.com"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None

class PreprocessRequest(BaseModel):
    session_id: str
    handle_missing: str = "auto"
    handle_outliers: str = "none"
    encode_categorical: str = "auto"
    scale_numeric: str = "standard"
    handle_imbalance: str = "none"

class PredictRequest(BaseModel):
    session_id: str
    model_type: str = "best"
    input_data: Optional[Dict[str, Any]] = None
    input_file: Optional[str] = None
    return_confidence: bool = True

class BatchPredictRequest(BaseModel):
    session_id: str
    model_type: str = "best"
    data: List[Dict[str, Any]]

# ==============================================================================
# UTILITIES
# ==============================================================================
def generate_session_id() -> str:
    return f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def save_chart(fig, name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.png"
    filepath = os.path.join(CHARTS_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    return filepath

def get_health_label(score: int) -> str:
    if score >= 80: return "Excellent"
    elif score >= 60: return "Good"
    elif score >= 40: return "Fair"
    else: return "Critical"

def suggest_target_candidates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Heuristic target candidates:
    - Prefer columns with <= 20 unique values (classification) and not obvious ID columns
    - Prefer numeric columns with > 20 unique values (regression)
    - Prefer columns whose name hints at target/label/price/churn/y
    """
    if df is None or df.empty:
        return []

    id_like = {"id", "identifier", "uuid", "uid", "index", "rowid", "customer_id", "user_id"}
    name_hints = {"target", "label", "class", "y", "price", "amount", "churn", "outcome", "response", "value", "score"}

    candidates: List[Dict[str, Any]] = []
    n_rows = len(df)

    for col in df.columns:
        s = df[col]
        col_l = str(col).lower()
        uniq = s.nunique(dropna=True)
        is_numeric = pd.api.types.is_numeric_dtype(s)

        if col_l in id_like or any(h in col_l for h in ["customer_id", "user_id", "account_id", "order_id"]):
            continue

        hint_score = 0
        if any(h in col_l for h in name_hints):
            hint_score += 3
        if uniq <= 20:
            hint_score += 2
        if is_numeric and uniq > 20:
            hint_score += 2

        # Missingness penalty
        miss_pct = float(s.isnull().sum() / max(1, n_rows) * 100)
        if miss_pct > 50:
            hint_score -= 2
        elif miss_pct > 20:
            hint_score -= 1

        if hint_score > 0:
            candidates.append({
                "column": col,
                "dtype": str(s.dtype),
                "unique": int(uniq),
                "missing_pct": round(miss_pct, 2),
                "hint_score": hint_score
            })

    candidates.sort(key=lambda x: x["hint_score"], reverse=True)
    return candidates[:8]

# ==============================================================================
# ROOT UI — Serve index.html
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def root_dashboard():
    """Serve the complete English dashboard UI"""
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding='utf-8'))
    # Fallback minimal dashboard if index.html missing
    return HTMLResponse(content="<h1>Dr. Data</h1><p>Please ensure index.html is in the same directory.</p>")

# ==============================================================================
# UPLOAD & SESSION ENDPOINTS
# ==============================================================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed = ('.csv', '.xlsx', '.xls', '.parquet', '.json')
    if not file.filename.endswith(allowed):
        raise HTTPException(400, f"Only {allowed} files are supported")

    # Sanitize filename to avoid path separators / traversal issues.
    original_name = file.filename or "upload"
    safe_name = original_name.replace("\\", "_").replace("/", "_")
    session_id = generate_session_id()
    filepath = os.path.join(UPLOAD_DIR, f"{session_id}_{safe_name}")

    # Ensure uploads dir exists.
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
    except PermissionError:
        raise HTTPException(
            500,
            {
                "error": "upload_permission_denied",
                "message": "Permission denied while writing to uploads directory.",
                "uploads_dir": UPLOAD_DIR,
                "filepath": filepath,
                "hint": "Run the server as Administrator or change UPLOAD_DIR to a writable folder (e.g., a temp directory)."
            },
        )



    try:
        session = DataSession(session_id, filepath, file.filename)
        sessions[session_id] = session

        target_candidates = suggest_target_candidates(session.df)

        metadata = dict(session.metadata)
        metadata["target_candidates"] = target_candidates
        if not target_candidates:
            metadata["target_warning"] = "No obvious target column detected. Please specify target_column in /analysis/model or /analysis/ann."

        # Make metadata JSON-safe for FastAPI/Starlette.
        def _to_json_safe(obj: Any):
            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return obj
            if isinstance(obj, dict):
                return {k: _to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_json_safe(v) for v in obj]
            return obj

        metadata = _to_json_safe(metadata)


        return {
            "success": True,
            "session_id": session_id,
            "filename": file.filename,
            "metadata": metadata,
            "message": f"Dataset loaded: {session.metadata['shape'][0]} rows × {session.metadata['shape'][1]} columns",
            "next_steps": ["POST /analysis/eda", "POST /analysis/preprocess", "POST /chat"]
        }

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(
            500,
            f"Failed to process file: {str(e)}\nResolved filepath: {filepath}\nUploads dir: {UPLOAD_DIR}"
        )


@app.get("/sessions/{session_id}/preview")
async def get_preview(session_id: str, rows: int = 10):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    df = sessions[session_id].df
    return {
        "columns": list(df.columns),
        "preview": df.head(rows).replace({np.nan: None}).to_dict(orient='records'),
        "total_rows": len(df),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()}
    }

@app.get("/sessions/{session_id}/columns")
async def get_columns(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[session_id]
    df = session.df
    columns_info = []
    for col in df.columns:
        info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isnull().sum()),
            "missing_pct": round(df[col].isnull().sum() / len(df) * 100, 2),
            "unique": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(5).astype(str).tolist()
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            info.update({
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                "skewness": float(df[col].skew()) if not pd.isna(df[col].skew()) else None,
            })
        columns_info.append(info)
    return {"columns": columns_info}

@app.get("/sessions/{session_id}/profile")
async def get_profile(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[session_id]
    df = session.df
    profile = {
        "basic": session.metadata,
        "advanced": {
            "memory_deep_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "column_density": {c: round((1 - df[c].isnull().sum()/len(df))*100, 2) for c in df.columns},
            "constant_columns": [c for c in df.columns if df[c].nunique() == 1],
            "near_constant_columns": [c for c in df.columns if df[c].nunique() / len(df) < 0.01 and df[c].nunique() > 1],
            "high_cardinality": [c for c in session.metadata["categorical_columns"] if df[c].nunique() > 100],
        }
    }
    return profile

# ==============================================================================
# EDA ENDPOINT
# ==============================================================================
@app.post("/analysis/eda")
async def run_eda(request: AnalysisRequest):
    logger.info(f"EDA request received for session: {request.session_id}")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    if request.session_id not in sessions:
        logger.warning(f"Session not found: {request.session_id}")
        raise HTTPException(404, "Session not found")

    session = sessions[request.session_id]
    df = session.df
    charts = []
    insights = []
    advanced_stats = {}

    try:
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False

        numeric_cols = session.metadata["numeric_columns"]
        categorical_cols = session.metadata["categorical_columns"]
        datetime_cols = session.metadata["datetime_columns"]
        
        logger.info(f"EDA started - Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}, Datetime: {len(datetime_cols)}")

        # 1. Missing Values Heatmap
        if df.isnull().sum().sum() > 0:
            fig, ax = plt.subplots(figsize=(14, max(4, len(df.columns) * 0.35)))
            missing_data = df.isnull().astype(int)
            sns.heatmap(missing_data, cbar=True, yticklabels=False, cmap='viridis', ax=ax)
            ax.set_title('Missing Values Pattern Heatmap', fontsize=14, fontweight='bold')
            charts.append({
                "title": "Missing Values Heatmap",
                "image": fig_to_base64(fig),
                "filepath": save_chart(fig, "missing_heatmap"),
                "type": "heatmap"
            })
            plt.close(fig)
            insights.append(f"Missing Data: {df.isnull().sum().sum()} total missing values across {df.isnull().any().sum()} columns")

        # 2. Distribution plots
        if numeric_cols:
            n_cols = min(len(numeric_cols), 6)
            n_rows = (n_cols + 2) // 3
            fig, axes = plt.subplots(n_rows, 3, figsize=(16, n_rows * 4))
            if n_rows == 1:
                axes = np.array([axes]).flatten() if n_cols > 1 else [axes]
            else:
                axes = axes.flatten()

            for i, col in enumerate(numeric_cols[:n_cols]):
                data_clean = df[col].dropna()
                sns.histplot(data_clean, kde=True, ax=axes[i], color='steelblue', alpha=0.7, stat='density')
                axes[i].set_title(f'Distribution: {col}', fontweight='bold')
                mean_val = data_clean.mean()
                skew_val = data_clean.skew()
                axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'μ={mean_val:.2f}')
                axes[i].axvline(data_clean.median(), color='green', linestyle='--', linewidth=2, label=f'median={data_clean.median():.2f}')
                axes[i].legend(fontsize=8)

                if len(data_clean) >= 8:
                    try:
                        shapiro_stat, shapiro_p = stats.shapiro(data_clean[:5000])
                        normal = "Normal" if shapiro_p > 0.05 else "Non-normal"
                        advanced_stats[col] = {"shapiro_wilk": {"statistic": round(shapiro_stat, 4), "p_value": round(shapiro_p, 4), "interpretation": normal}}
                    except:
                        pass

                skew_desc = "highly skewed" if abs(skew_val) > 1 else "moderately skewed" if abs(skew_val) > 0.5 else "symmetric"
                insights.append(f"{col}: μ={mean_val:.2f}, σ={data_clean.std():.2f}, skew={skew_val:.2f} ({skew_desc})")

            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            charts.append({
                "title": "Numeric Distributions",
                "image": fig_to_base64(fig),
                "filepath": save_chart(fig, "distributions"),
                "type": "distribution"
            })
            plt.close(fig)

        # 3. Box plots
        if numeric_cols and len(numeric_cols) <= 12:
            fig, ax = plt.subplots(figsize=(16, 7))
            plot_df = df[numeric_cols[:12]].copy()
            plot_df.boxplot(ax=ax, rot=45)
            ax.set_title('Outlier Detection — Box Plots', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            charts.append({
                "title": "Outlier Detection",
                "image": fig_to_base64(fig),
                "filepath": save_chart(fig, "boxplots"),
                "type": "boxplot"
            })
            plt.close(fig)

        # 4. Violin plots
        if numeric_cols and len(numeric_cols) <= 6:
            fig, axes = plt.subplots(1, min(len(numeric_cols), 6), figsize=(18, 5))
            if len(numeric_cols) == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            for i, col in enumerate(numeric_cols[:6]):
                sns.violinplot(y=df[col].dropna(), ax=axes[i], color='mediumpurple', inner='quartile')
                axes[i].set_title(col, fontweight='bold', fontsize=10)
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            plt.suptitle('Distribution Shapes — Violin Plots', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            charts.append({
                "title": "Violin Plots",
                "image": fig_to_base64(fig),
                "filepath": save_chart(fig, "violin_plots"),
                "type": "violin"
            })
            plt.close(fig)

        # 5. Categorical bar charts
        if categorical_cols:
            n_cat = min(len(categorical_cols), 4)
            fig, axes = plt.subplots(1, n_cat, figsize=(n_cat * 5.5, 5))
            if n_cat == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            for i, col in enumerate(categorical_cols[:n_cat]):
                value_counts = df[col].value_counts().head(12)
                colors = sns.color_palette("viridis", len(value_counts))
                sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[i], palette=colors)
                axes[i].set_title(f'Top Categories: {col}', fontweight='bold', fontsize=10)
                if len(value_counts) < df[col].nunique():
                    insights.append(f"{col}: Top category '{value_counts.index[0]}' ({value_counts.iloc[0]} rows), {df[col].nunique()} total unique values")
            plt.suptitle('Categorical Distributions', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            charts.append({
                "title": "Categorical Distributions",
                "image": fig_to_base64(fig),
                "filepath": save_chart(fig, "categorical"),
                "type": "bar"
            })
            plt.close(fig)

        # 6. Correlation Heatmap
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols)), max(8, len(numeric_cols) * 0.7)))
            corr_matrix = df[numeric_cols].corr(method='pearson')
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
            ax.set_title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            charts.append({
                "title": "Correlation Heatmap",
                "image": fig_to_base64(fig),
                "filepath": save_chart(fig, "correlation_heatmap"),
                "type": "heatmap"
            })
            plt.close(fig)

            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            if corr_pairs:
                insights.append("Strong correlations detected (|r| > 0.7):")
                for c1, c2, val in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                    insights.append(f"   {c1} ↔ {c2}: r = {val:.3f}")

        # 7. Pair plot
        if len(numeric_cols) >= 2 and len(numeric_cols) <= 5:
            pair_cols = numeric_cols[:4]
            hue_col = categorical_cols[0] if categorical_cols and df[categorical_cols[0]].nunique() <= 5 else None
            plot_df = df[pair_cols + ([hue_col] if hue_col else [])].dropna()
            if len(plot_df) > 10:
                g = sns.pairplot(plot_df, hue=hue_col, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30},
                                palette='viridis' if hue_col else None, corner=True)
                g.fig.suptitle('Pairwise Relationships', y=1.02, fontsize=14, fontweight='bold')
                charts.append({
                    "title": "Pair Plot",
                    "image": fig_to_base64(g.fig),
                    "filepath": save_chart(g.fig, "pairplot"),
                    "type": "pairplot"
                })
                plt.close(g.fig)

        # 8. Time series
        if datetime_cols and numeric_cols:
            try:
                ts_col = datetime_cols[0]
                ts_df = df[[ts_col] + numeric_cols[:3]].copy()
                ts_df[ts_col] = pd.to_datetime(ts_df[ts_col], errors='coerce')
                ts_df = ts_df.dropna().sort_values(ts_col)
                if len(ts_df) > 10:
                    fig, axes = plt.subplots(len(numeric_cols[:3]), 1, figsize=(14, 3 * len(numeric_cols[:3])))
                    if len(numeric_cols[:3]) == 1:
                        axes = [axes]
                    for i, col in enumerate(numeric_cols[:3]):
                        axes[i].plot(ts_df[ts_col], ts_df[col], color='steelblue', linewidth=1.5, alpha=0.8)
                        axes[i].fill_between(ts_df[ts_col], ts_df[col], alpha=0.2, color='steelblue')
                        axes[i].set_title(f'{col} over Time', fontweight='bold')
                    plt.tight_layout()
                    charts.append({
                        "title": "Time Series Overview",
                        "image": fig_to_base64(fig),
                        "filepath": save_chart(fig, "timeseries"),
                        "type": "line"
                    })
                    plt.close(fig)
                    insights.append(f"Time-series pattern detected in '{ts_col}' with {len(ts_df)} records")
            except:
                pass
        
        # 9. Q-Q Plots for Normality Check
        if numeric_cols:
            n_plots = min(len(numeric_cols), 6)
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            axes = axes.flatten()
            for i, col in enumerate(numeric_cols[:n_plots]):
                data_clean = df[col].dropna()
                if len(data_clean) >= 3:
                    stats.probplot(data_clean, dist="norm", plot=axes[i])
                    axes[i].set_title(f'Q-Q Plot: {col}', fontweight='bold', fontsize=10)
                    axes[i].grid(True, alpha=0.3)
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            plt.suptitle('Normality Assessment (Q-Q Plots)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            charts.append({
                "title": "Q-Q Plots",
                "image": fig_to_base64(fig),
                "filepath": save_chart(fig, "qq_plots"),
                "type": "scatter"
            })
            plt.close(fig)
        
        # 10. KDE (Kernel Density Estimation) overlays for multi-variable comparison
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(12, 7))
            for col in numeric_cols[:8]:
                data_clean = df[col].dropna()
                if len(data_clean) > 5:
                    data_clean.plot.kde(ax=ax, label=col, linewidth=2.5, alpha=0.7)
            ax.set_title('Overlapping Density Distributions', fontsize=14, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            charts.append({
                "title": "KDE Overlay",
                "image": fig_to_base64(fig),
                "filepath": save_chart(fig, "kde_overlay"),
                "type": "line"
            })
            plt.close(fig)
        
        # 11. Categorical Pie Charts for proportion visualization
        if categorical_cols:
            n_cat = min(len(categorical_cols), 4)
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()
            for i, col in enumerate(categorical_cols[:n_cat]):
                value_counts = df[col].value_counts().head(10)
                colors = sns.color_palette("Set3", len(value_counts))
                axes[i].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
                           colors=colors, startangle=90, textprops={'fontsize': 9})
                axes[i].set_title(f'{col} Distribution', fontweight='bold', fontsize=11)
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            plt.suptitle('Categorical Proportions', fontsize=14, fontweight='bold')
            plt.tight_layout()
            charts.append({
                "title": "Categorical Pie Charts",
                "image": fig_to_base64(fig),
                "filepath": save_chart(fig, "categorical_pie"),
                "type": "pie"
            })
            plt.close(fig)
        
        # 12. Advanced Correlation: Spearman (rank-based) for non-linear relationships
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols)), max(8, len(numeric_cols) * 0.7)))
            corr_matrix_spearman = df[numeric_cols].corr(method='spearman')
            mask = np.triu(np.ones_like(corr_matrix_spearman, dtype=bool), k=1)
            sns.heatmap(corr_matrix_spearman, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
            ax.set_title('Spearman Correlation Matrix (Rank-Based)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            charts.append({
                "title": "Spearman Correlation",
                "image": fig_to_base64(fig),
                "filepath": save_chart(fig, "spearman_correlation"),
                "type": "heatmap"
            })
            plt.close(fig)
        
        # 13. Skewness and Kurtosis visualization
        if numeric_cols:
            skewness_data = []
            kurtosis_data = []
            for col in numeric_cols:
                data_clean = df[col].dropna()
                if len(data_clean) >= 3:
                    skewness_data.append((col, data_clean.skew()))
                    kurtosis_data.append((col, data_clean.kurtosis()))
            
            if skewness_data:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Skewness
                cols_sk, vals_sk = zip(*skewness_data)
                colors_sk = ['#10b981' if abs(v) < 0.5 else '#f59e0b' if abs(v) < 1 else '#ef4444' for v in vals_sk]
                axes[0].barh(cols_sk, vals_sk, color=colors_sk, edgecolor='black', linewidth=1)
                axes[0].axvline(0, color='black', linestyle='-', linewidth=1)
                axes[0].set_title('Skewness by Feature', fontweight='bold', fontsize=13)
                axes[0].set_xlabel('Skewness')
                axes[0].grid(axis='x', alpha=0.3)
                
                # Kurtosis
                cols_kt, vals_kt = zip(*kurtosis_data)
                colors_kt = ['#10b981' if abs(v) < 3 else '#f59e0b' if abs(v) < 5 else '#ef4444' for v in vals_kt]
                axes[1].barh(cols_kt, vals_kt, color=colors_kt, edgecolor='black', linewidth=1)
                axes[1].axvline(3, color='red', linestyle='--', linewidth=1, label='Normal (3)')
                axes[1].set_title('Kurtosis by Feature', fontweight='bold', fontsize=13)
                axes[1].set_xlabel('Kurtosis')
                axes[1].legend()
                axes[1].grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                charts.append({
                    "title": "Skewness & Kurtosis",
                    "image": fig_to_base64(fig),
                    "filepath": save_chart(fig, "skewness_kurtosis"),
                    "type": "bar"
                })
                plt.close(fig)
        
        # 14. PCA / Dimensionality Reduction Visualization
        if len(numeric_cols) >= 3:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            try:
                pca_df = df[numeric_cols].dropna()
                if len(pca_df) >= 10:
                    pca_scaler = StandardScaler()
                    pca_data = pca_scaler.fit_transform(pca_df)
                    pca = PCA(n_components=min(3, len(numeric_cols)))
                    pca_result = pca.fit_transform(pca_data)
                    
                    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
                    
                    # 2D PCA
                    hue_col_pca = None
                    if categorical_cols and df[categorical_cols[0]].nunique() <= 6:
                        hue_col_pca = categorical_cols[0]
                        hue_data = pca_df[hue_col_pca].values
                    else:
                        hue_data = None
                    
                    scatter = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                              c=pd.factorize(hue_data)[0] if hue_data is not None else 'steelblue',
                                              cmap='viridis', alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
                    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
                    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
                    axes[0].set_title('PCA - 2D Projection', fontweight='bold', fontsize=13)
                    axes[0].grid(True, alpha=0.3)
                    if hue_data is not None:
                        axes[0].legend(*scatter.legend_elements(), title=hue_col_pca, fontsize=8)
                    
                    # Explained variance
                    cumsum = np.cumsum(pca.explained_variance_ratio_)
                    axes[1].bar(range(1, len(cumsum)+1), pca.explained_variance_ratio_, alpha=0.7, color='steelblue', edgecolor='black')
                    axes[1].plot(range(1, len(cumsum)+1), cumsum, 'r-o', linewidth=2, markersize=8, label='Cumulative')
                    axes[1].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% threshold')
                    axes[1].set_xlabel('Principal Component', fontsize=11)
                    axes[1].set_ylabel('Explained Variance Ratio', fontsize=11)
                    axes[1].set_title('PCA - Cumulative Explained Variance', fontweight='bold', fontsize=13)
                    axes[1].legend(fontsize=9)
                    axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    charts.append({
                        "title": "PCA Dimensionality Reduction",
                        "image": fig_to_base64(fig),
                        "filepath": save_chart(fig, "pca_analysis"),
                        "type": "scatter"
                    })
                    plt.close(fig)
                    
                    insights.append(f"PCA: First 2 components explain {pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1]:.1%} of variance")
            except Exception:
                pass

        # 15. Feature Pairwise Scatter Matrix (lower triangle, colored by categorical)
        if len(numeric_cols) >= 3 and len(numeric_cols) <= 5:
            try:
                pair_cols = numeric_cols[:4]
                pair_hue = None
                for c in categorical_cols:
                    if df[c].nunique() <= 5:
                        pair_hue = c
                        break
                plot_df = df[pair_cols + ([pair_hue] if pair_hue else [])].dropna().sample(min(500, len(df)), random_state=42)
                if len(plot_df) > 10:
                    g = sns.PairGrid(plot_df, hue=pair_hue, diag_sharey=False, corner=True)
                    g.map_lower(sns.scatterplot, alpha=0.6, s=25, edgecolor='black', linewidth=0.3)
                    g.map_diag(sns.histplot, kde=True, alpha=0.6)
                    if pair_hue:
                        g.add_legend(title=pair_hue, fontsize=8)
                    g.fig.suptitle('Feature Pairwise Relationships', y=1.02, fontsize=14, fontweight='bold')
                    charts.append({
                        "title": "Pairwise Scatter Matrix",
                        "image": fig_to_base64(g.fig),
                        "filepath": save_chart(g.fig, "pairwise_scatter"),
                        "type": "pairplot"
                    })
                    plt.close(g.fig)
            except Exception:
                pass

        # 16. Correlation Network / Cluster Map (Dendrogram)
        if len(numeric_cols) >= 3:
            try:
                from scipy.cluster.hierarchy import dendrogram, linkage
                corr_matrix = df[numeric_cols].corr(method='pearson').fillna(0)
                
                fig, axes = plt.subplots(1, 2, figsize=(18, 8))
                
                # Clustered heatmap
                sns.clustermap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                              center=0, square=True, linewidths=0.5,
                              figsize=(10, 8), dendrogram_ratio=(0.15, 0.15),
                              cbar_pos=(0.02, 0.8, 0.03, 0.15))
                plt.close()
                # Manually re-create simpler version
                linked = linkage(corr_matrix, method='ward')
                dendro = dendrogram(linked, ax=axes[0], orientation='top', labels=corr_matrix.columns,
                                   leaf_font_size=9, color_threshold=0.7*np.max(linked[:, 2]))
                axes[0].set_title('Feature Dendrogram (Hierarchical Clustering)', fontweight='bold', fontsize=12)
                axes[0].set_ylabel('Distance')
                axes[0].grid(axis='y', alpha=0.3)
                
                # Correlation bar chart (top absolute correlations)
                corr_vals = corr_matrix.abs().unstack().sort_values(ascending=False)
                corr_vals = corr_vals[corr_vals < 1.0].drop_duplicates().head(10)
                colors_corr = ['#10b981' if v >= 0.7 else '#f59e0b' if v >= 0.5 else '#ef4444' for v in corr_vals.values]
                axes[1].barh(range(len(corr_vals)), corr_vals.values, color=colors_corr, edgecolor='black', linewidth=0.5)
                axes[1].set_yticks(range(len(corr_vals)))
                axes[1].set_yticklabels([f'{i[0]} ↔ {i[1]}' for i in corr_vals.index], fontsize=8)
                axes[1].set_xlabel('Absolute Correlation', fontsize=11)
                axes[1].set_title('Top 10 Strongest Correlations', fontweight='bold', fontsize=12)
                axes[1].grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                charts.append({
                    "title": "Hierarchical Clustering & Correlations",
                    "image": fig_to_base64(fig),
                    "filepath": save_chart(fig, "hierarchical_correlation"),
                    "type": "mixed"
                })
                plt.close(fig)
            except Exception:
                pass

        # 17. Bivariate KDE Contour Plot
        if len(numeric_cols) >= 2:
            try:
                n_bivar = min(len(numeric_cols), 4)
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                axes = axes.flatten()
                pairs = [(numeric_cols[i], numeric_cols[j]) for i in range(min(n_bivar, 4)) for j in range(i+1, min(n_bivar, 4))][:4]
                for i, (cx, cy) in enumerate(pairs):
                    data_x = df[cx].dropna()
                    data_y = df[cy].dropna()
                    common = data_x.index.intersection(data_y.index)
                    if len(common) >= 5:
                        sns.kdeplot(x=df.loc[common, cx], y=df.loc[common, cy], 
                                   ax=axes[i], cmap='viridis', fill=True, thresh=0.05, levels=10)
                        axes[i].set_title(f'{cx} vs {cy}', fontweight='bold', fontsize=11)
                        axes[i].grid(True, alpha=0.2)
                for j in range(i+1, len(axes)):
                    axes[j].set_visible(False)
                plt.suptitle('Bivariate Density Contours', fontsize=14, fontweight='bold')
                plt.tight_layout()
                charts.append({
                    "title": "Bivariate KDE Contours",
                    "image": fig_to_base64(fig),
                    "filepath": save_chart(fig, "bivariate_kde"),
                    "type": "contour"
                })
                plt.close(fig)
            except Exception:
                pass

        # 18. ECDF (Empirical Cumulative Distribution Function)
        if numeric_cols:
            try:
                n_ecdf = min(len(numeric_cols), 8)
                fig, ax = plt.subplots(figsize=(14, 7))
                colors_ecdf = sns.color_palette("husl", n_ecdf)
                for i, col in enumerate(numeric_cols[:n_ecdf]):
                    data_clean = df[col].dropna().sort_values()
                    if len(data_clean) > 5:
                        ecdf = np.arange(1, len(data_clean)+1) / len(data_clean)
                        ax.plot(data_clean.values, ecdf, linewidth=2, alpha=0.8, 
                               label=f'{col}', color=colors_ecdf[i])
                ax.set_xlabel('Value', fontsize=12)
                ax.set_ylabel('ECDF', fontsize=12)
                ax.set_title('Empirical Cumulative Distribution Functions', fontweight='bold', fontsize=13)
                ax.legend(loc='lower right', fontsize=9)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                charts.append({
                    "title": "ECDF Plots",
                    "image": fig_to_base64(fig),
                    "filepath": save_chart(fig, "ecdf_plots"),
                    "type": "line"
                })
                plt.close(fig)
            except Exception:
                pass

        # 19. Heatmap of top 10% highest magnitude values
        if len(numeric_cols) >= 2:
            try:
                n_heat = min(len(numeric_cols), 10)
                heat_df = df[numeric_cols[:n_heat]].select_dtypes(include=[np.number]).dropna()
                if len(heat_df) >= 5:
                    sample_n = min(200, len(heat_df))
                    heat_sample = heat_df.sample(sample_n, random_state=42)
                    # Z-score normalize
                    heat_norm = (heat_sample - heat_sample.mean()) / heat_sample.std()
                    fig, ax = plt.subplots(figsize=(max(10, n_heat*0.6), 10))
                    sns.heatmap(heat_norm.T, cmap='RdBu_r', center=0, ax=ax,
                               linewidths=0.1, linecolor='gray', cbar_kws={'label': 'Z-Score'})
                    ax.set_title('Feature Value Heatmap (Z-Score Normalized)', fontweight='bold', fontsize=13)
                    ax.set_xlabel('Sample Index')
                    plt.tight_layout()
                    charts.append({
                        "title": "Feature Value Heatmap",
                        "image": fig_to_base64(fig),
                        "filepath": save_chart(fig, "feature_heatmap"),
                        "type": "heatmap"
                    })
                    plt.close(fig)
            except Exception:
                pass

        # 20. Parallel Coordinates Plot for high-dimensional patterns
        if len(numeric_cols) >= 2:
            try:
                from pandas.plotting import parallel_coordinates
                pc_cols = numeric_cols[:6]
                class_col = None
                for c in categorical_cols:
                    if df[c].nunique() <= 5:
                        class_col = c
                        break
                if class_col is None and len(numeric_cols) > 0:
                    # Use binned version of first numeric col
                    class_col = numeric_cols[0] + '_bin'
                    df_temp = df[pc_cols].copy()
                    df_temp[class_col] = pd.qcut(df[pc_cols[0]], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
                else:
                    df_temp = df[pc_cols + [class_col]].copy()
                
                df_temp = df_temp.dropna().sample(min(300, len(df_temp)), random_state=42)
                if len(df_temp) >= 10:
                    fig, ax = plt.subplots(figsize=(14, 7))
                    pd.plotting.parallel_coordinates(df_temp, class_column=class_col, 
                                                     colormap='viridis', alpha=0.5, linewidth=1, ax=ax)
                    ax.set_title('Parallel Coordinates Plot', fontweight='bold', fontsize=13)
                    ax.grid(True, alpha=0.2)
                    ax.legend(fontsize=8, loc='upper right')
                    plt.tight_layout()
                    charts.append({
                        "title": "Parallel Coordinates",
                        "image": fig_to_base64(fig),
                        "filepath": save_chart(fig, "parallel_coordinates"),
                        "type": "line"
                    })
                    plt.close(fig)
            except Exception:
                pass

        # 21. Missing Data Analysis by Quartile / Group
        if df.isnull().sum().sum() > 0 and numeric_cols:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Missing by quartile of first numeric feature
                ref_col = numeric_cols[0]
                if df[ref_col].notna().sum() >= 10:
                    df_miss = df.copy()
                    df_miss['_missing_count'] = df_miss.isnull().sum(axis=1)
                    df_miss['_quartile'] = pd.qcut(df_miss[ref_col].fillna(df_miss[ref_col].median()), 
                                                   q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                    miss_by_q = df_miss.groupby('_quartile')['_missing_count'].mean()
                    miss_by_q.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black', alpha=0.7)
                    axes[0].set_title(f'Avg Missing Values by {ref_col} Quartile', fontweight='bold', fontsize=11)
                    axes[0].set_ylabel('Avg Missing Count')
                    axes[0].grid(axis='y', alpha=0.3)
                
                # Missing correlation heatmap (which columns tend to be missing together)
                missing_indicator = df.isnull().astype(int)
                if missing_indicator.sum().sum() > 0 and missing_indicator.shape[1] >= 2:
                    missing_corr = missing_indicator.corr()
                    sns.heatmap(missing_corr, annot=True, fmt='.2f', cmap='YlOrRd',
                               ax=axes[1], linewidths=0.5, vmin=0, vmax=1,
                               cbar_kws={'label': 'Co-missing Correlation'})
                    axes[1].set_title('Missing Value Co-occurrence', fontweight='bold', fontsize=11)
                
                plt.tight_layout()
                charts.append({
                    "title": "Advanced Missing Data Analysis",
                    "image": fig_to_base64(fig),
                    "filepath": save_chart(fig, "advanced_missing"),
                    "type": "mixed"
                })
                plt.close(fig)
            except Exception:
                pass

        # 22. Distribution comparison: Boxen (letter-value) plots for large datasets
        if numeric_cols:
            try:
                n_boxen = min(len(numeric_cols), 8)
                fig, ax = plt.subplots(figsize=(14, 7))
                boxen_data = [df[col].dropna().values for col in numeric_cols[:n_boxen]]
                bp = ax.boxplot(boxen_data, labels=numeric_cols[:n_boxen], patch_artist=True,
                               showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=5))
                colors_box = sns.color_palette("Set2", n_boxen)
                for patch, color in zip(bp['boxes'], colors_box):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_title('Enhanced Box Plot Comparison (with Means)', fontweight='bold', fontsize=13)
                ax.set_ylabel('Value')
                ax.grid(axis='y', alpha=0.3)
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()
                charts.append({
                    "title": "Enhanced Box Plot Comparison",
                    "image": fig_to_base64(fig),
                    "filepath": save_chart(fig, "enhanced_boxplot"),
                    "type": "boxplot"
                })
                plt.close(fig)
            except Exception:
                pass

        # 23. Pairwise Scatter Plot Matrices (if <=5 numeric cols)
        if len(numeric_cols) <= 5 and len(numeric_cols) >= 2:
            try:
                fig, axes = plt.subplots(len(numeric_cols), len(numeric_cols), figsize=(14, 14))
                for i, col_i in enumerate(numeric_cols):
                    for j, col_j in enumerate(numeric_cols):
                        ax = axes[i, j]
                        if i == j:
                            ax.hist(df[col_i].dropna(), bins=25, color='steelblue', alpha=0.7, edgecolor='black')
                            ax.set_title(col_i, fontsize=8)
                        else:
                            ax.scatter(df[col_j], df[col_i], alpha=0.4, s=10, c='steelblue', edgecolors='none')
                            # Add regression line
                            from numpy.polynomial.polynomial import polyfit, polyval
                            try:
                                valid = df[[col_j, col_i]].dropna()
                                if len(valid) > 5:
                                    coeffs = np.polyfit(valid[col_j], valid[col_i], 1)
                                    x_line = np.linspace(valid[col_j].min(), valid[col_j].max(), 50)
                                    ax.plot(x_line, np.polyval(coeffs, x_line), 'r-', linewidth=1.5, alpha=0.6)
                            except Exception:
                                pass
                        if i < len(numeric_cols)-1:
                            ax.set_xticks([])
                        else:
                            ax.set_xlabel(col_j, fontsize=8)
                        if j > 0:
                            ax.set_yticks([])
                        else:
                            ax.set_ylabel(col_i, fontsize=8)
                plt.suptitle('Full Pairwise Scatter Matrix with Regression Lines', fontweight='bold', fontsize=14, y=1.01)
                plt.tight_layout()
                charts.append({
                    "title": "Scatter Matrix with Trends",
                    "image": fig_to_base64(fig),
                    "filepath": save_chart(fig, "scatter_matrix_full"),
                    "type": "mixed"
                })
                plt.close(fig)
            except Exception:
                pass

        # 24. Categorical heatmap - count / proportion of top categories
        if categorical_cols:
            try:
                if len(categorical_cols) >= 2:
                    top_cats = {}
                    for c in categorical_cols:
                        top_vals = df[c].value_counts().head(5).index.tolist()
                        top_cats[c] = top_vals
                    # Create cross-tabulation for first 2 categorical columns
                    c1, c2 = categorical_cols[0], categorical_cols[1]
                    ct = pd.crosstab(df[c1].apply(lambda x: x if x in top_cats[c1][:5] else 'Other'),
                                    df[c2].apply(lambda x: x if x in top_cats[c2][:5] else 'Other'),
                                    normalize='index')
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.heatmap(ct, annot=True, fmt='.2%', cmap='YlOrRd', ax=ax, linewidths=0.5)
                    ax.set_title(f'Proportion Heatmap: {c1} vs {c2}', fontweight='bold', fontsize=13)
                    plt.tight_layout()
                    charts.append({
                        "title": "Categorical Crosstab Heatmap",
                        "image": fig_to_base64(fig),
                        "filepath": save_chart(fig, "crosstab_heatmap"),
                        "type": "heatmap"
                    })
                    plt.close(fig)
            except Exception:
                pass

        # 25. Outlier Summary Visualization
        if numeric_cols:
            outlier_summary = []
            for col in numeric_cols:
                data_clean = df[col].dropna()
                if len(data_clean) >= 4:
                    Q1 = data_clean.quantile(0.25)
                    Q3 = data_clean.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = data_clean[(data_clean < lower_bound) | (data_clean > upper_bound)]
                    outlier_pct = len(outliers) / len(data_clean) * 100
                    if outlier_pct > 0:
                        outlier_summary.append((col, outlier_pct))
            
            if outlier_summary:
                fig, ax = plt.subplots(figsize=(12, 7))
                cols_out, pcts_out = zip(*sorted(outlier_summary, key=lambda x: x[1], reverse=True))
                colors_out = ['#ef4444' if p > 10 else '#f59e0b' if p > 5 else '#10b981' for p in pcts_out]
                ax.barh(cols_out, pcts_out, color=colors_out, edgecolor='black', linewidth=1)
                ax.set_xlabel('Outlier Percentage (%)')
                ax.set_title('Outlier Summary (IQR Method)', fontweight='bold', fontsize=14)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                charts.append({
                    "title": "Outlier Summary",
                    "image": fig_to_base64(fig),
                    "filepath": save_chart(fig, "outlier_summary"),
                    "type": "bar"
                })
                plt.close(fig)
                insights.append(f"Outliers detected in {len(outlier_summary)} columns via IQR method")

        session.analysis_history.append({
            "type": "EDA",
            "timestamp": datetime.now().isoformat(),
            "charts_count": len(charts),
            "insights": insights,
            "advanced_stats": advanced_stats
        })
        
        logger.info(f"EDA completed successfully - Generated {len(charts)} charts, {len(insights)} insights")

        return {
            "success": True,
            "charts": charts,
            "insights": insights,
            "advanced_statistics": advanced_stats,
            "summary": {
                "total_charts": len(charts),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "datetime_columns": len(datetime_cols),
                "health_score": session.metadata["health_score"],
                "health_status": get_health_label(session.metadata["health_score"])
            }
        }
    except Exception as e:
        logger.exception(f"EDA failed for session {request.session_id}: {str(e)}")
        raise HTTPException(500, f"EDA failed: {str(e)}\n{traceback.format_exc()}")

# ==============================================================================
# PREPROCESSING ENDPOINT
# ==============================================================================
@app.post("/analysis/preprocess")
async def preprocess_data(request: PreprocessRequest):
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
    from scipy import stats
    from collections import Counter

    if request.session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[request.session_id]
    df = session.df.copy()
    report = {"steps": [], "before_shape": list(df.shape), "columns_affected": []}

    try:
        # 1. Missing Values
        if request.handle_missing != "none":
            missing_before = int(df.isnull().sum().sum())
            if request.handle_missing == "auto":
                for col in df.columns:
                    if df[col].isnull().sum() > 0:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            skewness = df[col].skew()
                            if abs(skewness) > 1:
                                df[col].fillna(df[col].median(), inplace=True)
                                report["steps"].append(f"{col}: Filled missing with MEDIAN (skew={skewness:.2f})")
                            else:
                                df[col].fillna(df[col].mean(), inplace=True)
                                report["steps"].append(f"{col}: Filled missing with MEAN")
                        else:
                            mode_val = df[col].mode()
                            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                            df[col].fillna(fill_val, inplace=True)
                            report["steps"].append(f"{col}: Filled missing with MODE ('{fill_val}')")
            elif request.handle_missing == "drop":
                rows_before = len(df)
                df.dropna(inplace=True)
                report["steps"].append(f"Dropped {rows_before - len(df)} rows with missing values")
            elif request.handle_missing in ["mean", "median", "mode"]:
                for col in df.select_dtypes(include=[np.number]).columns:
                    if df[col].isnull().sum() > 0:
                        if request.handle_missing == "mean":
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif request.handle_missing == "median":
                            df[col].fillna(df[col].median(), inplace=True)
                        report["steps"].append(f"{col}: Filled missing with {request.handle_missing.upper()}")
            elif request.handle_missing == "knn":
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty and numeric_df.isnull().sum().sum() > 0:
                    imputer = KNNImputer(n_neighbors=min(5, len(df)-1))
                    df[numeric_df.columns] = imputer.fit_transform(numeric_df)
                    report["steps"].append("Filled missing numeric values with KNN Imputer (k=5)")
            missing_after = int(df.isnull().sum().sum())
            report["missing_handled"] = missing_before - missing_after

        # 2. Outliers
        if request.handle_outliers != "none":
            outlier_count = 0
            for col in df.select_dtypes(include=[np.number]).columns:
                if request.handle_outliers == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                    df[col] = df[col].clip(lower, upper)
                    outlier_count += int(outliers)
                elif request.handle_outliers == "zscore":
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outliers = (z_scores > 3).sum()
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    df[col] = df[col].clip(mean_val - 3*std_val, mean_val + 3*std_val)
                    outlier_count += int(outliers)
            if outlier_count > 0:
                report["steps"].append(f"Capped {outlier_count} outliers using {request.handle_outliers.upper()} method")

        # 3. Encoding
        encoders = {}
        if request.encode_categorical != "none":
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if request.encode_categorical == "auto":
                for col in cat_cols:
                    unique_count = df[col].nunique()
                    if unique_count <= 10:
                        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                        df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                        report["steps"].append(f"{col}: One-hot encoded ({unique_count} categories)")
                    else:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        encoders[col] = le
                        report["steps"].append(f"{col}: Label encoded ({unique_count} unique values)")
            elif request.encode_categorical == "label":
                for col in cat_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    encoders[col] = le
                    report["steps"].append(f"{col}: Label encoded")
            elif request.encode_categorical == "onehot":
                for col in cat_cols:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                    df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                    report["steps"].append(f"{col}: One-hot encoded")

        # 4. Scaling
        scaler = None
        if request.scale_numeric != "none":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if request.scale_numeric == "standard":
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                report["steps"].append(f"StandardScaler applied to {len(numeric_cols)} numeric columns")
            elif request.scale_numeric == "minmax":
                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                report["steps"].append(f"MinMaxScaler applied to {len(numeric_cols)} numeric columns")
            elif request.scale_numeric == "robust":
                scaler = RobustScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                report["steps"].append(f"RobustScaler applied to {len(numeric_cols)} numeric columns")

        # 5. Imbalance
        if request.handle_imbalance != "none" and session.target_column and session.target_column in df.columns:
            target = df[session.target_column]
            if not pd.api.types.is_numeric_dtype(target) or target.nunique() <= 10:
                class_counts = Counter(target)
                total = sum(class_counts.values())
                class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
                report["class_weights"] = {str(k): round(v, 3) for k, v in class_weights.items()}
                report["steps"].append(f"Class weights computed: {report['class_weights']}")

        session.df_processed = df
        session.scaler = scaler
        session.label_encoders = encoders
        report["after_shape"] = list(df.shape)
        report["columns_after"] = list(df.columns)

        processed_path = os.path.join(OUTPUT_DIR, f"{request.session_id}_processed.csv")
        df.to_csv(processed_path, index=False)
        report["output_file"] = processed_path

        missing_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        new_issues = []
        for col, pct in missing_pct.items():
            if pct > 50: new_issues.append(f"🚨 {col}: {pct}% missing — CRITICAL")
            elif pct > 20: new_issues.append(f"⚠️ {col}: {pct}% missing — HIGH")
        new_health = max(0, 100 - len(new_issues) * 8)

        return {
            "success": True,
            "report": report,
            "preview": df.head(5).replace({np.nan: None}).to_dict(orient='records'),
            "shape": {"before": report["before_shape"], "after": list(df.shape)},
            "new_health_score": new_health,
            "message": f"Preprocessing complete! {report['before_shape']} → {list(df.shape)}"
        }
    except Exception as e:
        raise HTTPException(500, f"Preprocessing failed: {str(e)}\n{traceback.format_exc()}")

# ==============================================================================
# ML MODELING ENDPOINT
# ==============================================================================
@app.post("/analysis/model")
async def train_model(request: AnalysisRequest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                                  accuracy_score, f1_score, confusion_matrix,
                                  precision_score, recall_score, roc_auc_score, roc_curve)
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.svm import SVR, SVC
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from scipy import stats

    if request.session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[request.session_id]
    df = session.df_processed if session.df_processed is not None else session.df

    if not request.target_column or request.target_column not in df.columns:
        candidates = suggest_target_candidates(df)
        raise HTTPException(
            400,
            {
                "error": "target_column_required",
                "message": "You must provide a valid target_column for model training.",
                "provided": request.target_column,
                "candidates": candidates
            }
        )

    charts = []
    results = {}
    cv_results = {}

    try:
        feature_cols = request.feature_columns or [c for c in df.columns if c != request.target_column]
        feature_cols = [c for c in feature_cols if c in df.columns and c != request.target_column]
        if not feature_cols:
            raise HTTPException(400, "No valid feature columns found")

        X = df[feature_cols].copy().fillna(df[feature_cols].mean())
        y = df[request.target_column].copy()

        if request.problem_type == "auto":
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                problem_type = "regression"
            else:
                problem_type = "classification"
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))
        else:
            problem_type = request.problem_type
            if y.dtype == 'object' and problem_type == "classification":
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))

        stratify = y if problem_type == "classification" else None

        # ---- CLASSIFICATION SAFETY: ensure at least 2 classes in training ----
        # Some datasets (or user-chosen target) may contain only one class.
        # In that case, LogisticRegression/SVM/etc. will crash.
        if problem_type == "classification":
            unique_y = np.unique(y)
            if len(unique_y) < 2:
                raise HTTPException(
                    400,
                    {
                        "error": "single_class",
                        "message": "Target column contains only one class. Classification models require at least 2 classes.",
                        "provided_target": request.target_column,
                        "classes": [str(c) for c in unique_y.tolist()],
                        "next": ["Try another target column", "For single-class targets, use regression or clustering instead"],
                    },
                )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=42, stratify=stratify
        )

        if problem_type == "classification":
            # Also ensure split didn't accidentally create a single-class train/test.
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                # Fallback: redo split without stratification.
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=request.test_size, random_state=42, stratify=None
                )

            if len(np.unique(y_train)) < 2:
                raise HTTPException(
                    400,
                    {
                        "error": "single_class_after_split",
                        "message": "After train/test split, training data still has only one class. Adjust test_size or choose a different target.",
                        "provided_target": request.target_column,
                        "train_classes": [str(c) for c in np.unique(y_train).tolist()],
                        "test_classes": [str(c) for c in np.unique(y_test).tolist()],
                    },
                )


        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        session.feature_columns = feature_cols
        session.target_column = request.target_column

        # Get hyperparameters from request or use defaults
        hp = request.ml_hyperparameters or MLHyperparameters()
        
        models = {}
        if problem_type == "regression":
            from sklearn.linear_model import ElasticNet
            from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.neural_network import MLPRegressor
            
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=hp.ridge_alpha, random_state=42),
                "Lasso Regression": Lasso(alpha=hp.lasso_alpha, random_state=42),
                "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
                "Random Forest": RandomForestRegressor(
                    n_estimators=hp.n_estimators,
                    max_depth=hp.max_depth,
                    min_samples_split=hp.min_samples_split,
                    min_samples_leaf=hp.min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
                ),
                "Extra Trees": ExtraTreesRegressor(
                    n_estimators=hp.n_estimators,
                    max_depth=hp.max_depth,
                    random_state=42,
                    n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=hp.n_estimators,
                    max_depth=max(3, hp.max_depth // 4) if hp.max_depth else 5,
                    random_state=42
                ),
                "AdaBoost": AdaBoostRegressor(
                    n_estimators=hp.n_estimators,
                    learning_rate=0.1,
                    random_state=42
                ),
                "Decision Tree": DecisionTreeRegressor(
                    max_depth=hp.max_depth,
                    min_samples_split=hp.min_samples_split,
                    random_state=42
                ),
                "KNN": KNeighborsRegressor(
                    n_neighbors=hp.knn_neighbors,
                    weights=hp.knn_weights,
                    metric=hp.knn_metric
                ),
                "SVR": SVR(
                    kernel=hp.svm_kernel,
                    C=hp.svm_c,
                    gamma=hp.svm_gamma
                ),
                "MLP": MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42
                )
            }
            
            # XGBoost
            if request.model_type in ("xgboost", "all"):
                try:
                    from xgboost import XGBRegressor
                    models["XGBoost"] = XGBRegressor(
                        n_estimators=hp.n_estimators,
                        max_depth=hp.xgb_max_depth,
                        learning_rate=hp.xgb_learning_rate,
                        subsample=hp.xgb_subsample,
                        colsample_bytree=hp.xgb_colsample_bytree,
                        random_state=42
                    )
                except ImportError:
                    pass
            
            # LightGBM
            if request.model_type in ("lightgbm", "lgb", "all"):
                try:
                    from lightgbm import LGBMRegressor
                    models["LightGBM"] = LGBMRegressor(
                        num_leaves=hp.lgb_num_leaves,
                        learning_rate=hp.lgb_learning_rate,
                        n_estimators=hp.n_estimators,
                        feature_fraction=hp.lgb_feature_fraction,
                        random_state=42,
                        verbose=-1
                    )
                except ImportError:
                    pass
            
            # CatBoost
            if request.model_type in ("catboost", "cat", "all"):
                try:
                    from catboost import CatBoostRegressor
                    models["CatBoost"] = CatBoostRegressor(
                        iterations=hp.n_estimators,
                        depth=hp.max_depth or 6,
                        learning_rate=0.1,
                        random_state=42,
                        verbose=0
                    )
                except ImportError:
                    pass
                    
        else:  # Classification
            from sklearn.linear_model import ElasticNet
            from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.neural_network import MLPClassifier
            
            models = {
                "Logistic Regression": LogisticRegression(
                    penalty=hp.logistic_penalty,
                    C=hp.logistic_c,
                    solver=hp.logistic_solver,
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=hp.n_estimators,
                    max_depth=hp.max_depth,
                    min_samples_split=hp.min_samples_split,
                    min_samples_leaf=hp.min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
                ),
                "Extra Trees": ExtraTreesClassifier(
                    n_estimators=hp.n_estimators,
                    max_depth=hp.max_depth,
                    random_state=42,
                    n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=hp.n_estimators,
                    max_depth=max(3, hp.max_depth // 4) if hp.max_depth else 5,
                    random_state=42
                ),
                "AdaBoost": AdaBoostClassifier(
                    n_estimators=hp.n_estimators,
                    learning_rate=0.1,
                    random_state=42
                ),
                "Decision Tree": DecisionTreeClassifier(
                    max_depth=hp.max_depth,
                    min_samples_split=hp.min_samples_split,
                    random_state=42
                ),
                "KNN": KNeighborsClassifier(
                    n_neighbors=hp.knn_neighbors,
                    weights=hp.knn_weights,
                    metric=hp.knn_metric
                ),
                "Naive Bayes": GaussianNB(),
                "SVM": SVC(
                    kernel=hp.svm_kernel,
                    C=hp.svm_c,
                    gamma=hp.svm_gamma,
                    probability=True,
                    random_state=42
                ),
                "MLP": MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42
                )
            }
            
            # XGBoost
            if request.model_type in ("xgboost", "all"):
                try:
                    from xgboost import XGBClassifier
                    models["XGBoost"] = XGBClassifier(
                        n_estimators=hp.n_estimators,
                        max_depth=hp.xgb_max_depth,
                        learning_rate=hp.xgb_learning_rate,
                        subsample=hp.xgb_subsample,
                        colsample_bytree=hp.xgb_colsample_bytree,
                        random_state=42
                    )
                except ImportError:
                    pass
            
            # LightGBM
            if request.model_type in ("lightgbm", "lgb", "all"):
                try:
                    from lightgbm import LGBMClassifier
                    models["LightGBM"] = LGBMClassifier(
                        num_leaves=hp.lgb_num_leaves,
                        learning_rate=hp.lgb_learning_rate,
                        n_estimators=hp.n_estimators,
                        feature_fraction=hp.lgb_feature_fraction,
                        random_state=42,
                        verbose=-1
                    )
                except ImportError:
                    pass
            
            # CatBoost
            if request.model_type in ("catboost", "cat", "all"):
                try:
                    from catboost import CatBoostClassifier
                    models["CatBoost"] = CatBoostClassifier(
                        iterations=hp.n_estimators,
                        depth=hp.max_depth or 6,
                        learning_rate=0.1,
                        random_state=42,
                        verbose=0
                    )
                except ImportError:
                    pass

        if request.model_type != "all":
            key_map = {
                "linear": ["Linear Regression", "Logistic Regression", "Ridge Regression", "Lasso Regression", "ElasticNet"],
                "random_forest": ["Random Forest"],
                "extra_trees": ["Extra Trees"],
                "xgboost": ["XGBoost"],
                "lightgbm": ["LightGBM"],
                "lgb": ["LightGBM"],
                "catboost": ["CatBoost"],
                "cat": ["CatBoost"],
                "svr": ["SVR"],
                "svm": ["SVM"],
                "knn": ["KNN"],
                "decision_tree": ["Decision Tree"],
                "adaboost": ["AdaBoost"],
                "mlp": ["MLP"],
                "naive_bayes": ["Naive Bayes"]
            }
            allowed = key_map.get(request.model_type.lower(), list(models.keys()))
            models = {k: v for k, v in models.items() if k in allowed}

        model_scores = {}
        trained_models = {}
        best_model_name = None
        best_score = -9999

        for name, model in models.items():
            use_scaled = any(x in name for x in ["Linear", "Logistic", "Ridge", "Lasso", "SVR", "SVM"])
            X_tr = X_train_scaled if use_scaled else X_train
            X_te = X_test_scaled if use_scaled else X_test

            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te) if hasattr(model, "predict_proba") and problem_type == "classification" else None

            if problem_type == "regression":
                scores = {
                    "R2": round(r2_score(y_test, y_pred), 4),
                    "MAE": round(mean_absolute_error(y_test, y_pred), 4),
                    "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                }
                main_metric = scores["R2"]
            else:
                scores = {
                    "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "F1": round(f1_score(y_test, y_pred, average='weighted'), 4),
                    "Precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                    "Recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                }
                if y_proba is not None:
                    try:
                        if len(np.unique(y_test)) == 2:
                            scores["ROC-AUC"] = round(roc_auc_score(y_test, y_proba[:, 1]), 4)
                        else:
                            scores["ROC-AUC"] = round(roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'), 4)
                    except:
                        pass
                main_metric = scores["Accuracy"]

            try:
                cv = StratifiedKFold(n_splits=min(request.cv_folds, len(y_train)), shuffle=True, random_state=42) if problem_type == "classification" else KFold(n_splits=min(request.cv_folds, len(y_train)), shuffle=True, random_state=42)
                cv_metric = 'accuracy' if problem_type == "classification" else 'r2'
                cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring=cv_metric, n_jobs=-1)
                cv_results[name] = {
                    "mean": round(float(cv_scores.mean()), 4),
                    "std": round(float(cv_scores.std()), 4),
                    "folds": [round(float(s), 4) for s in cv_scores]
                }
            except Exception as e:
                cv_results[name] = {"error": str(e)}

            model_scores[name] = scores
            trained_models[name] = model

            if main_metric > best_score:
                best_score = main_metric
                best_model_name = name
                results["predictions"] = y_pred.tolist()
                results["actual"] = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)

        for name, model in trained_models.items():
            model_filename = f"{request.session_id}_{name.replace(' ', '_').lower()}.joblib"
            model_path = os.path.join(MODELS_DIR, model_filename)
            joblib.dump({"model": model, "scaler": scaler, "features": feature_cols, "problem_type": problem_type}, model_path)
            session.model_registry[name] = {
                "path": model_path,
                "type": problem_type,
                "scores": model_scores[name],
                "cv": cv_results.get(name, {})
            }

        session.model_results = {
            "problem_type": problem_type,
            "model_scores": model_scores,
            "best_model": best_model_name,
            "feature_columns": feature_cols,
            "target_column": request.target_column,
            "cv_results": cv_results
        }

        # Chart 1: Model Comparison
        fig, ax = plt.subplots(figsize=(12, 7))
        model_names = list(model_scores.keys())
        if problem_type == "regression":
            metric_vals = [model_scores[m]["R2"] for m in model_names]
            colors = ['#10b981' if v > 0.7 else '#f59e0b' if v > 0.5 else '#ef4444' for v in metric_vals]
            bars = ax.barh(model_names, metric_vals, color=colors, edgecolor='black', linewidth=1.5, height=0.6)
            ax.set_xlabel('R² Score', fontsize=12)
            ax.set_title('Model Comparison — R² Score (Higher is Better)', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.7, label='Good (0.7)')
            ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate (0.5)')
        else:
            metric_vals = [model_scores[m]["Accuracy"] for m in model_names]
            colors = ['#10b981' if v > 0.8 else '#f59e0b' if v > 0.6 else '#ef4444' for v in metric_vals]
            bars = ax.barh(model_names, metric_vals, color=colors, edgecolor='black', linewidth=1.5, height=0.6)
            ax.set_xlabel('Accuracy', fontsize=12)
            ax.set_title('Model Comparison — Accuracy (Higher is Better)', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Good (0.8)')
            ax.axvline(x=0.6, color='orange', linestyle='--', alpha=0.7, label='Moderate (0.6)')

        for bar, val in zip(bars, metric_vals):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                   ha='left', va='center', fontweight='bold', fontsize=10)
        ax.legend(loc='lower right')
        plt.tight_layout()
        charts.append({
            "title": "Model Comparison",
            "image": fig_to_base64(fig),
            "filepath": save_chart(fig, "model_comparison"),
            "type": "bar"
        })
        plt.close(fig)

        # Chart 2: CV Scores
        if cv_results:
            fig, ax = plt.subplots(figsize=(12, 6))
            cv_names = [n for n in model_names if n in cv_results and "mean" in cv_results[n]]
            if cv_names:
                cv_means = [cv_results[n]["mean"] for n in cv_names]
                cv_stds = [cv_results[n]["std"] for n in cv_names]
                x_pos = np.arange(len(cv_names))
                ax.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, color='steelblue', edgecolor='black', alpha=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(cv_names, rotation=30, ha='right')
                ax.set_ylabel('CV Score' if problem_type == "regression" else 'CV Accuracy')
                ax.set_title(f'{request.cv_folds}-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                charts.append({
                    "title": "Cross-Validation Scores",
                    "image": fig_to_base64(fig),
                    "filepath": save_chart(fig, "cv_scores"),
                    "type": "bar"
                })
                plt.close(fig)

        # Chart 3: Predictions vs Actual
        fig, ax = plt.subplots(figsize=(10, 8))
        if problem_type == "regression":
            ax.scatter(results["actual"], results["predictions"], alpha=0.6, c='steelblue', edgecolors='black', linewidth=0.5, s=50)
            min_val = min(min(results["actual"]), min(results["predictions"]))
            max_val = max(max(results["actual"]), max(results["predictions"]))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Values', fontsize=12)
            ax.set_ylabel('Predicted Values', fontsize=12)
            ax.set_title(f'Predictions vs Actual — {best_model_name}', fontsize=14, fontweight='bold')
            ax.legend()
        else:
            cm = confusion_matrix(y_test, results["predictions"])
            labels = sorted(set(y_test))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=labels, yticklabels=labels, linewidths=0.5)
            ax.set_title(f'Confusion Matrix — {best_model_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        plt.tight_layout()
        charts.append({
            "title": "Predictions vs Actual",
            "image": fig_to_base64(fig),
            "filepath": save_chart(fig, "predictions_vs_actual"),
            "type": "scatter" if problem_type == "regression" else "heatmap"
        })
        plt.close(fig)

        # Chart 4: Feature Importance
        for name, model in trained_models.items():
            if hasattr(model, 'feature_importances_'):
                fig, ax = plt.subplots(figsize=(10, max(6, len(feature_cols) * 0.45)))
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                colors = sns.color_palette("viridis", len(feature_cols))
                sns.barplot(x=importances[indices], y=[feature_cols[i] for i in indices],
                           palette=colors, ax=ax, edgecolor='black', linewidth=0.5)
                ax.set_title(f'Feature Importance — {name}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Importance Score')
                plt.tight_layout()
                charts.append({
                    "title": f"Feature Importance — {name}",
                    "image": fig_to_base64(fig),
                    "filepath": save_chart(fig, f"feature_importance_{name.replace(' ', '_').lower()}"),
                    "type": "bar"
                })
                plt.close(fig)
                break

        # Chart 5: Residuals
        if problem_type == "regression":
            residuals = np.array(results["actual"]) - np.array(results["predictions"])
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            axes[0].scatter(results["predictions"], residuals, alpha=0.6, c='coral', edgecolors='black', linewidth=0.5, s=50)
            axes[0].axhline(y=0, color='red', linestyle='--', lw=2)
            axes[0].set_xlabel('Predicted Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residual Plot', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[1].hist(residuals, bins=35, edgecolor='black', color='skyblue', alpha=0.7, density=True)
            mu, std = np.mean(residuals), np.std(residuals)
            x = np.linspace(mu - 3*std, mu + 3*std, 100)
            axes[1].plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label='Normal fit')
            axes[1].axvline(x=0, color='red', linestyle='--', lw=2)
            axes[1].set_xlabel('Residual Value')
            axes[1].set_ylabel('Density')
            axes[1].set_title('Residual Distribution', fontweight='bold')
            axes[1].legend()
            plt.tight_layout()
            charts.append({
                "title": "Residual Analysis",
                "image": fig_to_base64(fig),
                "filepath": save_chart(fig, "residuals"),
                "type": "scatter"
            })
            plt.close(fig)

        # Chart 6: ROC Curves
        if problem_type == "classification" and len(np.unique(y_test)) == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            for name, model in trained_models.items():
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(X_test_scaled if any(x in name for x in ["Linear", "Logistic", "SVM"]) else X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, proba)
                        auc = roc_auc_score(y_test, proba)
                        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f})')
                    except:
                        pass
            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            charts.append({
                "title": "ROC Curves",
                "image": fig_to_base64(fig),
                "filepath": save_chart(fig, "roc_curves"),
                "type": "line"
            })
            plt.close(fig)

        # Chart 7: Prediction Heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        n_samples = min(50, len(results["actual"]))
        pred_matrix = np.array([results["actual"][:n_samples], results["predictions"][:n_samples]])
        sns.heatmap(pred_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=[f'S{i+1}' for i in range(n_samples)],
                   yticklabels=['Actual', 'Predicted'], ax=ax, linewidths=0.5)
        ax.set_title(f'Prediction Heatmap (Top {n_samples} Samples)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        charts.append({
            "title": "Prediction Heatmap",
            "image": fig_to_base64(fig),
            "filepath": save_chart(fig, "prediction_heatmap"),
            "type": "heatmap"
        })
        plt.close(fig)

        return {
            "success": True,
            "problem_type": problem_type,
            "model_scores": model_scores,
            "cv_results": cv_results,
            "best_model": best_model_name,
            "charts": charts,
            "feature_count": len(feature_cols),
            "sample_count": len(y_test),
            "models_saved": list(session.model_registry.keys()),
            "insights": [
                f"Best Model: {best_model_name} ({'R²=' if problem_type == 'regression' else 'Acc='}{best_score:.3f})",
                f"Problem Type: {problem_type.title()}",
                f"Features Used: {len(feature_cols)} columns",
                f"Test Samples: {len(y_test)} ({request.test_size*100:.0f}% split)",
                f"Cross-Validation: {request.cv_folds}-fold"
            ]
        }
    except Exception as e:
        # Return detailed error so the UI can show the real cause (not just 500).
        tb = traceback.format_exc()
        raise HTTPException(500, f"Model training failed: {str(e)}\n{tb}")

# ==============================================================================
# ANN ENDPOINT
# ==============================================================================
@app.post("/analysis/ann")
async def train_ann(request: AnalysisRequest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                                  accuracy_score, f1_score, confusion_matrix, precision_score, recall_score)

    if request.session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[request.session_id]
    df = session.df_processed if session.df_processed is not None else session.df

    if not request.target_column or request.target_column not in df.columns:
        candidates = suggest_target_candidates(df)
        raise HTTPException(
            400,
            {
                "error": "target_column_required",
                "message": "You must provide a valid target_column for ANN training.",
                "provided": request.target_column,
                "candidates": candidates
            }
        )

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        from tensorflow.keras.optimizers import AdamW
        from tensorflow.keras.regularizers import l2

        tf.get_logger().setLevel('ERROR')

        feature_cols = request.feature_columns or [c for c in df.columns if c != request.target_column]
        feature_cols = [c for c in feature_cols if c in df.columns and c != request.target_column]
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[request.target_column].copy()

        # ──────────────────────────────────────────────────────────────
        # Output layer rules (applied for both auto-detect and explicit):
        #
        #   REGRESSION:
        #     - 1 output neuron, linear activation (no squashing)
        #     - loss: MSE (mean squared error)
        #
        #   BINARY CLASSIFICATION:
        #     - 1 output neuron, sigmoid activation (probability in [0,1])
        #     - loss: binary_crossentropy
        #     - threshold at 0.5 for class prediction
        #
        #   MULTI-CLASS CLASSIFICATION:
        #     - n output neurons (one per class), softmax activation
        #     - loss: sparse_categorical_crossentropy
        #     - argmax over outputs for class prediction
        # ──────────────────────────────────────────────────────────────
        if request.problem_type == "auto":
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                problem_type = "regression"
                output_units = 1
                output_activation = "linear"       # no activation = regression output
                loss = "mse"
                metrics = ["mae"]
            else:
                problem_type = "classification"
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))
                n_classes = len(np.unique(y))
                if n_classes > 2:
                    output_units = n_classes      # one neuron per class
                    output_activation = "softmax"  # produces class probabilities
                    loss = "sparse_categorical_crossentropy"
                else:
                    output_units = 1              # single neuron for binary
                    output_activation = "sigmoid"  # probability in [0,1]
                    loss = "binary_crossentropy"
                metrics = ["accuracy"]
        else:
            problem_type = request.problem_type
            if problem_type == "regression":
                output_units = 1
                output_activation = "linear"
                loss = "mse"
                metrics = ["mae"]
            else:
                n_classes = len(np.unique(y))
                if n_classes > 2:
                    output_units = n_classes
                    output_activation = "softmax"
                    loss = "sparse_categorical_crossentropy"
                else:
                    output_units = 1
                    output_activation = "sigmoid"
                    loss = "binary_crossentropy"
                metrics = ["accuracy"]

        stratify = y if problem_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=request.test_size, random_state=42, stratify=stratify)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        input_dim = X_train.shape[1]
        
        # Get ANN configuration from request or use defaults
        ann_cfg = request.ann_config or ANNConfig()
        
        # Auto-detect architecture if needed
        if ann_cfg.auto_architecture or not ann_cfg.hidden_layers:
            if input_dim <= 10:
                hidden_layers = [64, 32]
                dropout_rates = [0.2, 0.1]
            elif input_dim <= 50:
                hidden_layers = [128, 64, 32]
                dropout_rates = [0.3, 0.2, 0.1]
            elif input_dim <= 100:
                hidden_layers = [256, 128, 64, 32]
                dropout_rates = [0.4, 0.3, 0.2, 0.1]
            else:
                hidden_layers = [512, 256, 128, 64]
                dropout_rates = [0.4, 0.3, 0.2, 0.1]
        else:
            # Use user-specified architecture
            hidden_layers = ann_cfg.hidden_layers
            if ann_cfg.dropout_rates and len(ann_cfg.dropout_rates) >= len(hidden_layers):
                dropout_rates = ann_cfg.dropout_rates[:len(hidden_layers)]
            else:
                # Generate default dropout rates if not specified
                dropout_rates = [0.3] * len(hidden_layers)
        
        # Override output activation and loss if user specified
        if ann_cfg.output_activation:
            output_activation = ann_cfg.output_activation
        if ann_cfg.loss:
            loss = ann_cfg.loss

        # Build model
        model = Sequential()
        model.add(Dense(
            hidden_layers[0],
            input_shape=(input_dim,),
            kernel_regularizer=l2(ann_cfg.l2_regularization)
        ))
        if ann_cfg.use_batch_normalization:
            model.add(BatchNormalization())
        model.add(Activation(ann_cfg.activation))
        model.add(Dropout(dropout_rates[0]))

        for units, drop in zip(hidden_layers[1:], dropout_rates[1:]):
            model.add(Dense(units, kernel_regularizer=l2(ann_cfg.l2_regularization)))
            if ann_cfg.use_batch_normalization:
                model.add(BatchNormalization())
            model.add(Activation(ann_cfg.activation))
            model.add(Dropout(drop))

        model.add(Dense(output_units, activation=output_activation))

        # Configure optimizer
        if ann_cfg.optimizer.lower() == "adamw":
            optimizer = keras.optimizers.AdamW(
                learning_rate=ann_cfg.learning_rate,
                weight_decay=ann_cfg.weight_decay
            )
        elif ann_cfg.optimizer.lower() == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=ann_cfg.learning_rate)
        elif ann_cfg.optimizer.lower() == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=ann_cfg.learning_rate, momentum=0.9)
        elif ann_cfg.optimizer.lower() == "rmsprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=ann_cfg.learning_rate)
        else:
            optimizer = keras.optimizers.AdamW(learning_rate=ann_cfg.learning_rate, weight_decay=ann_cfg.weight_decay)
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        model_path = os.path.join(MODELS_DIR, f"{request.session_id}_ann_best.keras")
        
        # Configure callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=ann_cfg.early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=ann_cfg.reduce_lr_patience,
                min_lr=ann_cfg.min_learning_rate,
                verbose=0
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]

        # Auto-detect batch size if not specified
        if ann_cfg.batch_size:
            batch_size = ann_cfg.batch_size
        else:
            batch_size = min(64, max(16, len(X_train) // 32))

        history = model.fit(
            X_train, y_train,
            validation_split=ann_cfg.validation_split,
            epochs=ann_cfg.epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

        y_pred = model.predict(X_test, verbose=0)

        if problem_type == "regression":
            y_pred_flat = y_pred.flatten()
            scores = {
                "R2": round(r2_score(y_test, y_pred_flat), 4),
                "MAE": round(mean_absolute_error(y_test, y_pred_flat), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_flat)), 4),
            }
        else:
            if output_units > 1:
                y_pred_classes = np.argmax(y_pred, axis=1)
            else:
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            scores = {
                "Accuracy": round(accuracy_score(y_test, y_pred_classes), 4),
                "F1": round(f1_score(y_test, y_pred_classes, average='weighted'), 4),
                "Precision": round(precision_score(y_test, y_pred_classes, average='weighted', zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred_classes, average='weighted', zero_division=0), 4),
            }

        charts = []

        # Chart 1: Training History
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        axes[0].set_title('Training & Validation Loss', fontweight='bold', fontsize=13)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        metric_key = 'mae' if problem_type == 'regression' else 'accuracy'
        val_metric_key = f'val_{metric_key}'
        if metric_key in history.history:
            axes[1].plot(history.history[metric_key], label=f'Training {metric_key.upper()}', color='green', linewidth=2)
            axes[1].plot(history.history[val_metric_key], label=f'Validation {metric_key.upper()}', color='red', linewidth=2)
            axes[1].set_title(f'Training & Validation {metric_key.upper()}', fontweight='bold', fontsize=13)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel(metric_key.upper())
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        charts.append({
            "title": "ANN Training History",
            "image": fig_to_base64(fig),
            "filepath": save_chart(fig, "ann_training_history"),
            "type": "line"
        })
        plt.close(fig)

        # Chart 2: Architecture
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.text(0.5, 0.95, 'Neural Network Architecture', ha='center', va='top',
               fontsize=18, fontweight='bold', transform=ax.transAxes)
        layers_info = [f"Input Layer: {input_dim} neurons"]
        for idx, (units, drop) in enumerate(zip(hidden_layers, dropout_rates)):
            layers_info.append(f"Hidden {idx+1}: {units} neurons (ReLU + BatchNorm + Dropout {drop})")
        layers_info.append(f"Output Layer: {output_units} neuron(s) ({output_activation})")

        for i, layer in enumerate(layers_info):
            color = '#e0e7ff' if 'Input' in layer else '#c7d2fe' if 'Hidden' in layer else '#a5b4fc'
            ax.text(0.5, 0.82 - i*0.12, layer, ha='center', va='center',
                   fontsize=11, bbox=dict(boxstyle='round,pad=0.6', facecolor=color, edgecolor='#6366f1', linewidth=1.5),
                   transform=ax.transAxes)
            if i < len(layers_info) - 1:
                ax.annotate('', xy=(0.5, 0.82 - (i+1)*0.12 + 0.03), xytext=(0.5, 0.82 - i*0.12 - 0.03),
                           arrowprops=dict(arrowstyle='->', color='#6366f1', lw=2), transform=ax.transAxes)

        ax.text(0.5, 0.05, f'Optimizer: AdamW (lr=0.001, wd=0.004) | Loss: {loss} | Epochs: {len(history.history["loss"])}',
               ha='center', va='center', fontsize=10, transform=ax.transAxes, style='italic')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.tight_layout()
        charts.append({
            "title": "ANN Architecture",
            "image": fig_to_base64(fig),
            "filepath": save_chart(fig, "ann_architecture"),
            "type": "diagram"
        })
        plt.close(fig)

        # Chart 3: Predictions
        fig, ax = plt.subplots(figsize=(10, 8))
        if problem_type == "regression":
            ax.scatter(y_test, y_pred_flat, alpha=0.6, c='mediumseagreen', edgecolors='black', linewidth=0.5, s=50)
            min_val = min(min(y_test), min(y_pred_flat))
            max_val = max(max(y_test), max(y_pred_flat))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
            ax.set_xlabel('Actual', fontsize=12)
            ax.set_ylabel('Predicted', fontsize=12)
            ax.set_title('ANN: Predictions vs Actual', fontweight='bold', fontsize=14)
            ax.legend()
        else:
            if output_units > 1:
                y_pred_classes = np.argmax(y_pred, axis=1)
            else:
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            cm = confusion_matrix(y_test, y_pred_classes)
            labels = sorted(set(y_test))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, linewidths=0.5,
                       xticklabels=labels, yticklabels=labels)
            ax.set_title('ANN Confusion Matrix', fontweight='bold', fontsize=14)
        plt.tight_layout()
        charts.append({
            "title": "ANN Predictions",
            "image": fig_to_base64(fig),
            "filepath": save_chart(fig, "ann_predictions"),
            "type": "scatter" if problem_type == "regression" else "heatmap"
        })
        plt.close(fig)

        # Chart 4: Prediction Heatmap
        fig, ax = plt.subplots(figsize=(16, 5))
        n_samples = min(30, len(y_test))
        if problem_type == "regression":
            pred_matrix = np.array([y_test[:n_samples], y_pred_flat[:n_samples]])
            fmt = '.2f'
        else:
            pred_matrix = np.array([y_test[:n_samples], y_pred_classes[:n_samples]])
            fmt = 'd'
        sns.heatmap(pred_matrix, annot=True, fmt=fmt, cmap='RdYlGn',
                   xticklabels=[f'S{i+1}' for i in range(n_samples)],
                   yticklabels=['Actual', 'Predicted'], ax=ax, linewidths=0.5)
        ax.set_title(f'ANN Prediction Heatmap (Top {n_samples})', fontweight='bold', fontsize=14)
        plt.tight_layout()
        charts.append({
            "title": "ANN Prediction Heatmap",
            "image": fig_to_base64(fig),
            "filepath": save_chart(fig, "ann_prediction_heatmap"),
            "type": "heatmap"
        })
        plt.close(fig)

        # Chart 5: LR Schedule
        if 'lr' in history.history:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history.history['lr'], color='purple', linewidth=2)
            ax.set_title('Learning Rate Schedule', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            charts.append({
                "title": "Learning Rate Schedule",
                "image": fig_to_base64(fig),
                "filepath": save_chart(fig, "ann_lr_schedule"),
                "type": "line"
            })
            plt.close(fig)

        session.ann_model_path = model_path
        session.ann_metadata = {
            "problem_type": problem_type,
            "scores": scores,
            "architecture": {
                "input_dim": input_dim,
                "hidden_layers": hidden_layers,
                "output_units": output_units,
                "activations": ["relu"] * len(hidden_layers) + [output_activation],
                "dropout": dropout_rates,
                "epochs_trained": len(history.history['loss']),
                "final_loss": round(float(history.history['loss'][-1]), 6),
                "final_val_loss": round(float(history.history['val_loss'][-1]), 6),
                "best_epoch": int(np.argmin(history.history['val_loss']) + 1)
            }
        }

        session.model_registry["ANN"] = {
            "path": model_path,
            "type": problem_type,
            "scores": scores,
            "framework": "keras"
        }

        return {
            "success": True,
            "problem_type": problem_type,
            "scores": scores,
            "architecture": session.ann_metadata["architecture"],
            "charts": charts,
            "model_path": model_path,
            "insights": [
                f"ANN trained for {len(history.history['loss'])} epochs (best epoch {session.ann_metadata['architecture']['best_epoch']})",
                f"Architecture: {input_dim} → {' → '.join(map(str, hidden_layers))} → {output_units}",
                f"Regularization: L2(0.001) + BatchNorm + Dropout {dropout_rates}",
                f"Optimizer: AdamW with ReduceLROnPlateau",
                f"Model saved to: {model_path}"
            ]
        }
    except ImportError:
        raise HTTPException(500, "TensorFlow/Keras not installed. Install with: pip install tensorflow")
    except Exception as e:
        raise HTTPException(500, f"ANN training failed: {str(e)}\n{traceback.format_exc()}")

# ==============================================================================
# CLUSTERING ENDPOINT
# ==============================================================================
@app.post("/analysis/clustering")
async def run_clustering(request: AnalysisRequest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    if request.session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[request.session_id]
    df = session.df_processed if session.df_processed is not None else session.df
    numeric_cols = session.metadata["numeric_columns"]
    if len(numeric_cols) < 2:
        raise HTTPException(400, "Need at least 2 numeric columns for clustering")

    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    charts = []
    insights = []

    try:
        inertias = []
        silhouettes = []
        K_range = range(2, min(11, len(X)))
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels))

        best_k = list(K_range)[np.argmax(silhouettes)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia (WCSS)')
        axes[0].set_title('Elbow Method', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
        axes[0].legend()

        axes[1].plot(list(K_range), silhouettes, 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
        axes[1].legend()
        plt.tight_layout()
        charts.append({
            "title": "Clustering Diagnostics",
            "image": fig_to_base64(fig),
            "filepath": save_chart(fig, "clustering_diagnostics"),
            "type": "line"
        })
        plt.close(fig)

        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        df['Cluster'] = cluster_labels

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
        centers = pca.transform(kmeans.cluster_centers_)
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, edgecolors='black', linewidth=2, label='Centroids')
        ax.set_title(f'K-Means Clustering (k={best_k}) — PCA Projection', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.legend()
        plt.tight_layout()
        charts.append({
            "title": "Cluster Visualization (PCA)",
            "image": fig_to_base64(fig),
            "filepath": save_chart(fig, "cluster_pca"),
            "type": "scatter"
        })
        plt.close(fig)

        cluster_profiles = df.groupby('Cluster')[numeric_cols].mean().round(2).to_dict()
        insights.append(f"Optimal clusters: k={best_k} (silhouette={max(silhouettes):.3f})")
        insights.append(f"PCA explains {sum(pca.explained_variance_ratio_):.1%} of variance in 2D")
        for c in range(best_k):
            size = int((cluster_labels == c).sum())
            insights.append(f"Cluster {c}: {size} samples ({size/len(df):.1%})")

        return {
            "success": True,
            "optimal_k": int(best_k),
            "silhouette_score": round(float(max(silhouettes)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(X_scaled, cluster_labels)), 2),
            "cluster_profiles": cluster_profiles,
            "charts": charts,
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(500, f"Clustering failed: {str(e)}\n{traceback.format_exc()}")

# ==============================================================================
# PREDICTION ENDPOINTS
# ==============================================================================
@app.post("/predict")
async def predict(request: PredictRequest):
    if request.session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[request.session_id]

    try:
        if request.model_type == "best" or request.model_type == "ann":
            if session.ann_model_path and os.path.exists(session.ann_model_path):
                import tensorflow as tf
                model = tf.keras.models.load_model(session.ann_model_path)
                model_kind = "ann"
            else:
                if not session.model_registry:
                    raise HTTPException(400, "No trained models found. Train a model first.")
                best_name = session.model_results.get("best_model", list(session.model_registry.keys())[0])
                model_data = joblib.load(session.model_registry[best_name]["path"])
                model = model_data["model"]
                model_kind = "sklearn"
        else:
            if request.model_type in session.model_registry:
                model_data = joblib.load(session.model_registry[request.model_type]["path"])
                model = model_data["model"]
                model_kind = "sklearn"
            else:
                raise HTTPException(400, f"Model '{request.model_type}' not found. Available: {list(session.model_registry.keys())}")

        features = session.feature_columns
        if not features:
            raise HTTPException(400, "No feature columns defined. Train a model first.")

        if request.input_data:
            input_df = pd.DataFrame([request.input_data])
            for feat in features:
                if feat not in input_df.columns:
                    input_df[feat] = 0
            input_df = input_df[features]

            if model_kind == "sklearn" and session.scaler:
                input_scaled = session.scaler.transform(input_df)
            else:
                input_scaled = input_df.values

            if model_kind == "ann":
                pred = model.predict(input_scaled, verbose=0)
                if session.ann_metadata and session.ann_metadata.get("problem_type") == "classification":
                    if pred.shape[1] > 1:
                        pred_class = int(np.argmax(pred, axis=1)[0])
                        confidence = float(np.max(pred, axis=1)[0])
                    else:
                        pred_class = int((pred > 0.5).astype(int).flatten()[0])
                        confidence = float(pred.flatten()[0])
                    result = {"prediction": pred_class, "confidence": round(confidence, 4), "model": "ANN"}
                else:
                    result = {"prediction": round(float(pred.flatten()[0]), 4), "model": "ANN"}
            else:
                pred = model.predict(input_scaled)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_scaled)
                    confidence = float(np.max(proba, axis=1)[0])
                    result = {"prediction": pred[0] if hasattr(pred, '__iter__') else pred,
                             "confidence": round(confidence, 4), "model": request.model_type}
                else:
                    result = {"prediction": pred[0] if hasattr(pred, '__iter__') else pred, "model": request.model_type}

            return {"success": True, "result": result, "features_used": features}

        elif request.input_file:
            pred_df = session.df_processed if session.df_processed is not None else session.df
            pred_df = pred_df[features].fillna(0)

            if model_kind == "sklearn" and session.scaler:
                pred_scaled = session.scaler.transform(pred_df)
            else:
                pred_scaled = pred_df.values

            if model_kind == "ann":
                preds = model.predict(pred_scaled, verbose=0)
                if session.ann_metadata and session.ann_metadata.get("problem_type") == "classification":
                    if preds.shape[1] > 1:
                        pred_classes = np.argmax(preds, axis=1).tolist()
                        confidences = np.max(preds, axis=1).tolist()
                    else:
                        pred_classes = (preds > 0.5).astype(int).flatten().tolist()
                        confidences = preds.flatten().tolist()
                    results = [{"prediction": c, "confidence": round(conf, 4)} for c, conf in zip(pred_classes, confidences)]
                else:
                    results = [{"prediction": round(float(p), 4)} for p in preds.flatten()]
            else:
                preds = model.predict(pred_scaled)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(pred_scaled)
                    results = [{"prediction": p, "confidence": round(float(np.max(pr)), 4)} for p, pr in zip(preds, proba)]
                else:
                    results = [{"prediction": p} for p in preds]

            return {
                "success": True,
                "predictions": results[:100],
                "total_predictions": len(results),
                "model": request.model_type,
                "features_used": features
            }
        else:
            raise HTTPException(400, "Provide either input_data or input_file")
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}\n{traceback.format_exc()}")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    if request.session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[request.session_id]
    try:
        model_type = request.model_type
        if model_type == "best":
            model_type = session.model_results.get("best_model", list(session.model_registry.keys())[0])

        if model_type not in session.model_registry:
            raise HTTPException(400, f"Model '{model_type}' not found")

        model_data = joblib.load(session.model_registry[model_type]["path"])
        model = model_data["model"]
        features = model_data.get("features", session.feature_columns)

        input_df = pd.DataFrame(request.data)
        for feat in features:
            if feat not in input_df.columns:
                input_df[feat] = 0
        input_df = input_df[features].fillna(0)

        if session.scaler:
            input_scaled = session.scaler.transform(input_df)
        else:
            input_scaled = input_df.values

        preds = model.predict(input_scaled)
        results = []
        for i, p in enumerate(preds):
            entry = {"index": i, "prediction": p}
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_scaled[i:i+1])
                entry["confidence"] = round(float(np.max(proba)), 4)
            results.append(entry)

        return {"success": True, "predictions": results, "model": model_type, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, f"Batch prediction failed: {str(e)}")

# ==============================================================================
# MULTI-AGENT CHAT  (MoE Edition)
# ==============================================================================
_SWARM_CACHE: Dict[str, Any] = {}

def _provider_cache_key(provider_cfg: Dict[str, Any]) -> str:
    try:
        return json.dumps(provider_cfg, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(provider_cfg)


def _get_swarm(provider_cfg: Dict[str, Any]):
    """Returns a cached AgentSwarm for the given provider config."""
    from agent import AgentSwarm
    key = _provider_cache_key(provider_cfg)
    if key not in _SWARM_CACHE:
        _SWARM_CACHE[key] = AgentSwarm(provider_cfg)
    return _SWARM_CACHE[key]


def _auto_detect_web_search(message: str) -> bool:
    """
    Detect if the user's message implies a web search request.
    Runs before calling the LLM — zero latency.
    """
    triggers = [
        "search", "find online", "look up", "latest", "current", "today",
        "news", "recent", "trend", "what is", "who is", "how does",
        "جستجو", "سرچ", "اینترنت", "بیا پیدا کن", "جدیدترین", "اخبار",
        "تحقیق کن", "find info", "tell me about",
    ]
    msg_lower = message.lower()
    return any(t in msg_lower for t in triggers)


def _build_provider_cfg(request: "ChatRequest") -> Dict[str, Any]:
    """Build LLM provider config from request fields."""
    cfg: Dict[str, Any] = {"type": request.llm_provider}
    pt = request.llm_provider.lower()
    if pt in ["llamacpp", "llama-cpp", "gguf", "llama_cpp"]:
        cfg["model_path"] = request.gguf_model_path or MODEL_PATH
        cfg["n_ctx"]      = int(os.environ.get("N_CTX", 2048))
        cfg["n_threads"]  = int(os.environ.get("N_THREADS", os.cpu_count() or 8))
    elif pt == "ollama":
        cfg["ollama_base_url"] = request.ollama_base_url
        cfg["ollama_model"]    = request.ollama_model
    elif pt in ["openai", "chatgpt"]:
        if not request.openai_api_key:
            raise HTTPException(400, "openai_api_key required for openai/chatgpt provider")
        cfg["openai_api_base"] = request.openai_api_base
        cfg["openai_model"]    = request.openai_model
        cfg["openai_api_key"]  = request.openai_api_key
    return cfg


@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    # ── Session guard ─────────────────────────────────────────────────────────
    session = sessions.get(request.session_id)
    if session is None:
        # Allow chat without a dataset (intent router will handle casual mode)
        logger.warning(f"No session found for {request.session_id} — running in no-data mode")

    # ── Auto-detect web search intent ─────────────────────────────────────────
    do_search = request.search_web or _auto_detect_web_search(request.message)

    # ── Web search ────────────────────────────────────────────────────────────
    search_results_text: Optional[str] = None
    web_search_obj: Optional[Dict] = None
    citations: List[Dict[str, str]] = []

    if do_search:
        try:
            from web_search import search_web, format_search_for_llm
            web_search_obj = search_web(
                query=request.message,
                max_results=5,
                fetch_top=True,          # fetch top-2 pages for richer context
                max_chars_per_page=2000,
            )
            # search_results is already a formatted string — do NOT join chars
            search_results_text = format_search_for_llm(web_search_obj, max_results=3)

            # Build structured citations for the UI
            for r in (web_search_obj.get("results") or [])[:5]:
                url = r.get("url", "")
                title = r.get("title", url)
                snippet = r.get("snippet", "")
                if url:
                    citations.append({"title": title, "url": url, "snippet": snippet})

            logger.info(f"Web search completed: {len(citations)} results for '{request.message[:60]}'")
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            search_results_text = f"[Web search unavailable: {str(e)}]"

    # ── Build session data context ────────────────────────────────────────────
    session_data: Optional[Dict[str, Any]] = None
    if session:
        session_data = {
            "filename":            session.filename,
            "shape":               session.metadata.get("shape"),
            "columns":             session.metadata.get("columns"),
            "numeric_columns":     session.metadata.get("numeric_columns"),
            "categorical_columns": session.metadata.get("categorical_columns"),
            "datetime_columns":    session.metadata.get("datetime_columns"),
            "health_score":        session.metadata.get("health_score"),
            "health_issues":       session.metadata.get("health_issues"),
            "missing_values":      session.metadata.get("missing_values"),
            "duplicate_rows":      session.metadata.get("duplicate_rows"),
            "pii_detected":        session.metadata.get("pii_detected", []),
            "analysis_history":    session.analysis_history,
            "model_registry":      list(session.model_registry.keys()),
            "best_model":          session.model_results.get("best_model"),
            "ann_model_path":      session.ann_model_path,
        }

    # ── Execute MoE swarm ─────────────────────────────────────────────────────
    try:
        provider_cfg = _build_provider_cfg(request)
        swarm = _get_swarm(provider_cfg)

        result = swarm.execute(
            request.message,
            session_data=session_data,
            web_search_results=search_results_text,
        )

        return {
            "success":             True,
            "session_id":          request.session_id,
            "message":             request.message,
            "intent":              result.get("intent", "unknown"),
            "active_agents":       result.get("active_agents", []),
            "execution_plan":      result.get("execution_plan", []),
            "agent_outputs":       result.get("agent_outputs", {}),
            "response":            result.get("synthesis", ""),
            "search_enabled":      do_search,
            "search_results_raw":  search_results_text,
            "citations":           citations,           # structured [{title,url,snippet}]
            "available_tools": [
                "run_eda", "preprocess_data", "train_ml_model",
                "train_ann", "predict", "generate_report", "search_web",
            ],
            "suggested_next_steps": [
                {"action": "POST /analysis/eda",        "description": "Run full EDA"},
                {"action": "POST /analysis/preprocess", "description": "Clean & encode data"},
                {"action": "POST /analysis/model",      "description": "Train ML models"},
                {"action": "POST /analysis/ann",        "description": "Train neural network"},
            ],
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Chat error: {tb}")
        raise HTTPException(500, f"Chat failed: {str(e)}\n{tb}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    result = await chat_with_agent(request)
    return result

# ==============================================================================
# REPORTING & EXPORTS
# ==============================================================================
@app.get("/sessions/{session_id}/export/report")
async def export_report(session_id: str, format: str = "markdown"):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[session_id]
    report = f"""# 📊 Dr. Data Analysis Report
**Session ID:** `{session_id}`  
**Dataset:** {session.filename}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Executive Summary

Dataset **{session.filename}** contains **{session.metadata['shape'][0]} rows** and **{session.metadata['shape'][1]} columns**.

**Data Health Score:** {session.metadata['health_score']}/100 — *{get_health_label(session.metadata['health_score'])}*

### Key Findings
- **Numeric Features:** {len(session.metadata['numeric_columns'])}
- **Categorical Features:** {len(session.metadata['categorical_columns'])}
- **Missing Values:** {sum(session.metadata['missing_values'].values())} total
- **Duplicate Rows:** {session.metadata['duplicate_rows']}

---

## 2. Data Quality Assessment

"""
    if session.metadata["health_issues"]:
        report += "### Issues Detected\n"
        for issue in session.metadata["health_issues"]:
            report += f"- {issue}\n"
    else:
        report += "✅ No critical data quality issues detected.\n"

    if session.metadata.get("pii_detected"):
        report += "\n### Privacy & Security\n"
        for pii in session.metadata["pii_detected"]:
            report += f"- {pii}\n"

    report += f"\n\n## 3. Column Profiles\n\n| Column | Type | Missing | Unique | Summary |\n|--------|------|---------|--------|---------|\n"
    for col in session.metadata["columns"]:
        dtype = session.metadata["dtypes"][col]
        missing = session.metadata["missing_values"][col]
        unique = session.df[col].nunique()
        if col in session.metadata["numeric_summary"]:
            ns = session.metadata["numeric_summary"][col]
            summary = f"μ={ns.get('mean', 'N/A'):.2f}, σ={ns.get('std', 'N/A'):.2f}"
        else:
            summary = f"Top: {session.metadata.get('categorical_summary', {}).get(col, {}).get('top', 'N/A')}"
        report += f"| {col} | {dtype} | {missing} | {unique} | {summary} |\n"

    if session.analysis_history:
        report += "\n\n## 4. Analysis History\n"
        for idx, hist in enumerate(session.analysis_history, 1):
            report += f"\n### Analysis {idx}: {hist.get('type', 'Unknown')}\n"
            report += f"- **Timestamp:** {hist.get('timestamp', 'N/A')}\n"
            report += f"- **Charts Generated:** {hist.get('charts_count', 0)}\n"
            if hist.get('insights'):
                report += "- **Key Insights:**\n"
                for ins in hist['insights'][:5]:
                    report += f"  - {ins}\n"

    if session.model_registry:
        report += "\n\n## 5. Model Performance\n"
        for name, info in session.model_registry.items():
            report += f"\n### {name}\n"
            report += f"- **Type:** {info.get('type', 'N/A')}\n"
            report += f"- **Framework:** {info.get('framework', 'sklearn')}\n"
            report += "- **Scores:**\n"
            for metric, val in info.get('scores', {}).items():
                report += f"  - {metric}: {val}\n"
            if info.get('cv'):
                report += f"- **CV Mean:** {info['cv'].get('mean', 'N/A')}, **CV Std:** {info['cv'].get('std', 'N/A')}\n"

    report += """
---

## 6. Recommendations

1. **Data Quality:** Address all flagged issues before modeling
2. **Feature Engineering:** Consider log-transform for highly skewed features
3. **Model Selection:** Compare cross-validation scores across algorithms
4. **Validation:** Use stratified K-Fold for classification, time-series split for temporal data
5. **Deployment:** Export best model via `/predict` endpoint for real-time inference

---

*Report generated by Dr. Data Multi-Agent Swarm v2.1*
"""

    report_path = os.path.join(REPORTS_DIR, f"{session_id}_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return FileResponse(report_path, media_type='text/markdown', filename=f"dr_data_report_{session_id}.md")

@app.get("/sessions/{session_id}/download/{chart_name}")
async def download_chart(session_id: str, chart_name: str):
    filepath = os.path.join(CHARTS_DIR, chart_name)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type='image/png', filename=chart_name)
    raise HTTPException(404, "Chart not found")

@app.get("/models/{session_id}/{model_name}")
async def download_model(session_id: str, model_name: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[session_id]
    if model_name not in session.model_registry:
        raise HTTPException(404, "Model not found")
    path = session.model_registry[model_name]["path"]
    return FileResponse(path, media_type='application/octet-stream', filename=f"{model_name}.joblib")

@app.get("/sessions/{session_id}/export/csv")
async def export_csv(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    df = sessions[session_id].df_processed or sessions[session_id].df
    filepath = os.path.join(OUTPUT_DIR, f"{session_id}_export.csv")
    df.to_csv(filepath, index=False)
    return FileResponse(filepath, media_type='text/csv', filename=f"processed_data.csv")

@app.get("/health")
async def health_check():
    import psutil
    return {
        "status": "healthy",
        "version": "2.1.0",
        "active_sessions": len(sessions),
        "model_loaded": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH,
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_free_gb": round(psutil.disk_usage('/').free / 1024**3, 2),
        },
        "endpoints": {
            "upload": "POST /upload",
            "eda": "POST /analysis/eda",
            "preprocess": "POST /analysis/preprocess",
            "model": "POST /analysis/model",
            "ann": "POST /analysis/ann",
            "clustering": "POST /analysis/clustering",
            "predict": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "chat": "POST /chat",
            "chat_stream": "POST /chat/stream",
            "report": "GET /sessions/{id}/export/report",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    print("=" * 80)
    print("  Dr. Data - Multi-Agent Data Science Swarm v2.1")
    print("=" * 80)
    print(f"  Uploads:   {UPLOAD_DIR}")
    print(f"  Outputs:   {OUTPUT_DIR}")
    print(f"  Models:    {MODELS_DIR}")
    print(f"  Reports:   {REPORTS_DIR}")
    print(f"  LLM:       {MODEL_PATH}")
    print(f"  Dashboard: http://localhost:8000")
    print(f"  API Docs:  http://localhost:8000/docs")
    print("=" * 80)
    uvicorn.run(app, port=8000, log_level="info")