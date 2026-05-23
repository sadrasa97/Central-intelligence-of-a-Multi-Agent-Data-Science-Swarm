# 🧠 Dr. Data — Multi-Agent Data Science Swarm v2.1

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Seven specialized AI agents collaborate to ingest, analyze, model, visualize, and predict on your datasets — powered by **FastAPI** + a local LLM (GGUF via `llama-cpp-python`) with optional cloud provider support (OpenAI, Ollama).

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Configuration](#️-configuration)
- [LLM Providers](#-llm-providers)
- [API Reference](#-api-reference)
- [Multi-Agent Architecture](#-multi-agent-architecture)
- [Web Search Integration](#-web-search-integration)
- [Performance Tuning](#-performance-tuning)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 🔭 Overview

Dr. Data is an **all-in-one data science platform** that brings the power of large language models (LLMs) together with classical ML, deep learning, and rich visualizations. Upload any dataset (CSV, XLSX, Parquet, JSON) and get instant access to:

- Automated **Exploratory Data Analysis** (25+ chart types)
- Intelligent **Preprocessing** (missing values, outliers, encoding, scaling)
- **Machine Learning** model training and comparison (15+ algorithms)
- **Deep Learning** with auto-architecture ANN
- **Clustering** with elbow/silhouette analysis
- **Single & batch prediction** with confidence scores
- **Multi-agent chat** powered by LLM (local or cloud)
- **Web search** for domain context enrichment
- **Report generation** in Markdown

---

## ✨ Features

### 📤 **Upload & Explore**
| Format | Support | Details |
|--------|---------|---------|
| CSV | ✅ Full | Auto-detect encoding, low-memory mode |
| XLSX / XLS | ✅ Full | Multi-sheet support via openpyxl |
| Parquet | ✅ Full | Columnar storage, fast I/O |
| JSON | ✅ Full | Structured JSON parsing |

After upload, Dr. Data automatically extracts rich metadata:
- **Shape**, **dtypes** per column
- **Numeric**, **categorical**, **datetime** column classification
- **Missing value** analysis (count + percentage)
- **Duplicate row** detection
- **PII detection** (emails, phone numbers, SSNs, credit cards)
- **Data health score** (0–100) with actionable issues
- **Target column suggestions** via heuristic scoring

### 📊 **EDA — 25+ Chart Types**
- Missing Values Heatmap
- Numeric Distributions (histogram + KDE)
- Box Plots (outlier detection)
- Violin Plots (distribution shape)
- Categorical Bar Charts
- Pearson & Spearman Correlation Heatmaps
- Pair Plot / Pairwise Scatter Matrix
- Q-Q Plots (normality assessment)
- KDE Overlay (multi-variable density)
- Categorical Pie Charts
- PCA — 2D Projection & Explained Variance
- Time Series Overview
- Feature Dendrogram (hierarchical clustering)
- Bivariate KDE Contour Plots
- ECDF Plots
- Skewness & Kurtosis Bar Charts
- Parallel Coordinates Plot
- Enhanced Box Plot Comparison
- Feature Value Heatmap (Z-score normalized)
- Missing Data Analysis by Quartile
- Categorical Crosstab Heatmap
- Outlier Summary Bar Chart

### 🧹 **Preprocessing — Smart Data Cleaning**
| Operation | Methods | Description |
|-----------|---------|-------------|
| **Missing Values** | `auto`, `drop`, `mean`, `median`, `mode`, `knn` | Auto mode uses median for skewed, mean for symmetric, mode for categorical |
| **Outlier Handling** | `none`, `iqr`, `zscore` | IQR caps at 1.5×IQR; Z-score caps at ±3σ |
| **Categorical Encoding** | `auto`, `label`, `onehot` | Auto uses one-hot for ≤10 categories, label for high-cardinality |
| **Numeric Scaling** | `none`, `standard`, `minmax`, `robust` | StandardScaler, MinMaxScaler, or RobustScaler |
| **Class Imbalance** | `none`, `weight` | Computes class weights for balanced training |

### 🤖 **Machine Learning — 15+ Algorithms**
#### Regression
- Linear Regression
- Ridge Regression (L2)
- Lasso Regression (L1)
- ElasticNet
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- Decision Tree Regressor
- KNN Regressor
- SVR (RBF/Linear/Poly/Sigmoid)
- MLP Regressor
- **XGBoost Regressor** (if installed)
- **LightGBM Regressor** (if installed)
- **CatBoost Regressor** (if installed)

#### Classification
- Logistic Regression (L1/L2/ElasticNet)
- Random Forest Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Decision Tree Classifier
- KNN Classifier
- Naive Bayes (Gaussian)
- SVM (with probability=True)
- MLP Classifier
- **XGBoost Classifier** (if installed)
- **LightGBM Classifier** (if installed)
- **CatBoost Classifier** (if installed)

All models report: **R²/Accuracy**, **MAE/RMSE/F1**, **Precision/Recall**, **ROC-AUC**, and **K-Fold Cross-Validation** scores with visualizations (bar charts, confusion matrices, residual plots, ROC curves, feature importance, prediction heatmaps).

### 🧠 **Deep Learning — ANN with Auto-Architecture**
- **Auto-architecture detection** based on input dimensionality:
  - ≤10 features → [64, 32] layers
  - ≤50 features → [128, 64, 32]
  - ≤100 features → [256, 128, 64, 32]
  - >100 features → [512, 256, 128, 64]
- **Optimizers**: AdamW (default), Adam, SGD (+momentum), RMSprop
- **Regularization**: L2 kernel regularization, Batch Normalization, Dropout, Early Stopping, ReduceLROnPlateau
- **Activation functions**: relu, tanh, sigmoid, elu, selu
- **Auto-detection** of problem type (regression/binary/multi-class) with appropriate loss functions and output activations
- **Interactive UI** (via `ann_config.js`) with live architecture visualization, layer add/remove, preset architectures (Shallow/Medium/Deep/Wide)

### 🔗 **Clustering**
- K-Means with elbow method & silhouette analysis
- Optimal k auto-selection
- PCA 2D projection with cluster centroids
- Cluster profile summaries (mean per numeric column)
- Calinski-Harabasz index

### 🔮 **Prediction**
- **Single prediction**: Pass `input_data` dict for instant inference
- **Batch prediction**: Pass array of input dicts
- **Model selection**: Choose by name or use `"best"` / `"ann"`
- **Confidence scores**: For classification models with `predict_proba`
- **Pre/post-processing**: Automatic feature alignment and scaling

### 💬 **Multi-Agent Chat — MoE Architecture**
8 specialized agents in a Mixture-of-Experts pattern:

| Agent | Role |
|-------|------|
| 🧠 **Orchestrator** | Decomposes requests into sub-tasks |
| 📊 **Data Analyst** | Dataset statistics, correlations, data quality |
| 🧹 **Preprocessor** | Missing values, outliers, encoding, scaling |
| ⚙️ **ML Engineer** | Model training, comparison, CV, overfitting detection |
| 🧬 **Deep Learning** | ANN design, training, regularization |
| 🎨 **Visualizer** | Chart recommendations, colorblind-friendly |
| 🔮 **Predictor** | Single/batch inference, confidence, drift detection |
| 📝 **Reporter** | Professional report generation (Persian/English) |

The **Intent Router** uses a two-stage classification:
1. **Fast keyword scan** (zero LLM calls, ~1ms) covering English & Persian
2. **LLM fallback** for ambiguous queries

### 🌐 **Web Search**
- Multi-backend DuckDuckGo search (html → lite → bing → raw fallback)
- Auto-detect search intent from message keywords
- Fetches top-2 result pages for richer LLM context
- Structured citations returned to UI
- Rate-limit handling with UA rotation & exponential backoff

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
# Install Python dependencies
pip install -r requirements.txt

# Optional: Install huggingface_hub for auto-download
pip install huggingface_hub

# Optional: Install additional ML frameworks
pip install xgboost lightgbm catboost
```

**Core dependencies:**
| Package | Purpose |
|---------|---------|
| `fastapi`, `uvicorn` | API server framework |
| `llama-cpp-python` | Local GGUF model inference |
| `pandas`, `numpy` | Data processing |
| `scikit-learn`, `scipy` | ML algorithms & statistics |
| `matplotlib`, `seaborn` | Visualization |
| `tensorflow`, `keras` | Deep learning (ANN) |
| `duckduckgo-search` | Web search backend |
| `psutil` | System resource monitoring |

### 2. Get a GGUF Model

Place a GGUF model in `./models/` directory, or set the `MODEL_PATH` environment variable:

```bash
# Recommended: Qwen3.5-2B (fast, good for data science tasks)
# Auto-downloads from HuggingFace if missing
```

The system checks these locations in order:
1. `D:\models\Qwen3.5-2B-Q4_K_S.gguf`
2. `./models/Qwen3.5-2B-Q4_K_S.gguf`
3. `./Qwen3.5-2B-Q4_K_S.gguf`
4. Auto-downloads to `./models/Qwen3.5-2B-Q4_K_S.gguf`

**Auto-download fallback models** (tried in order):
- `Qwen/Qwen3.5-2B-GGUF` → `qwen3-5-2b-q4_k_s.gguf`
- `Qwen/Qwen3.5-1.5B-GGUF` → `qwen3-5-1.5b-q4_k_s.gguf`

Pattern-based detection also supports Gemma, Llama-3, Mistral, and Phi-3 models.

### 3. Run

```bash
# Start the server (default: http://localhost:8000)
python main.py
```

```bash
# Or with custom port
python main.py --port 8080
```

Open your browser at **http://localhost:8000** for the dashboard UI.

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | Auto-detected | Path to GGUF model file |
| `N_CTX` | `2048` | Context window size (lower = faster, less memory) |
| `N_THREADS` | CPU count | CPU threads for LLM inference |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable debug logging |
| `UPLOAD_DIR` | `%TEMP%/dr_data/uploads` | Dataset upload directory |
| `OPENAI_API_KEY` | — | API key for OpenAI provider |

### `.env` File

Create a `.env` file from `env.example`:

```bash
cp env.example .env
```

---

## 🔌 LLM Providers

### 1. **llamacpp** (default — local, private, free)
```json
{
  "llm_provider": "llamacpp",
  "gguf_model_path": "./models/qwen3.5-2b-q4_0.gguf"
}
```
- Uses `llama-cpp-python` for GGUF inference
- **Zero cost**, fully offline, no data leaves your machine
- Configurable via `N_CTX`, `N_THREADS`

### 2. **ollama** (connect to local Ollama)
```json
{
  "llm_provider": "ollama",
  "ollama_base_url": "http://localhost:11434",
  "ollama_model": "llama3"
}
```
- Connects to a running [Ollama](https://ollama.ai/) instance
- Supports any model you've pulled (llama3, mistral, phi, etc.)

### 3. **openai** (cloud — OpenAI / OpenRouter / any OpenAI-compatible API)
```json
{
  "llm_provider": "openai",
  "openai_api_base": "https://api.openai.com",
  "openai_model": "gpt-4o-mini",
  "openai_api_key": "sk-..."
}
```
- Supports any OpenAI-compatible API (OpenAI, OpenRouter, Groq, etc.)
- Best quality responses; requires API key

---

## 📡 API Reference

### Dashboard & Docs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard UI (index.html) |
| `GET` | `/docs` | Swagger UI API documentation |
| `GET` | `/health` | System health + active sessions |

### Upload & Session Management

#### `POST /upload`
Upload a dataset file.

**Request:** `multipart/form-data` with a `file` field (CSV, XLSX, Parquet, JSON)

**Response:**
```json
{
  "success": true,
  "session_id": "sess_20260522_123456_abcd1234",
  "filename": "data.csv",
  "metadata": {
    "shape": [1000, 15],
    "columns": ["age", "income", "churn", ...],
    "numeric_columns": ["age", "income"],
    "categorical_columns": ["churn"],
    "missing_values": {"age": 5, "income": 0},
    "health_score": 85,
    "health_issues": ["⚠️ age: 5% missing — MODERATE"],
    "pii_detected": ["🔒 email: Potential EMAIL detected"],
    "target_candidates": [
      {"column": "churn", "dtype": "object", "unique": 2, "hint_score": 5}
    ]
  }
}
```

#### `GET /sessions/{session_id}/preview?rows=10`
Preview first N rows of the dataset.

#### `GET /sessions/{session_id}/columns`
Get detailed column information (dtype, missing stats, unique values, numeric stats).

#### `GET /sessions/{session_id}/profile`
Get full data profile including memory usage, constant columns, high-cardinality columns.

### Analysis Endpoints

#### `POST /analysis/eda`
Run comprehensive EDA (25+ chart types).

**Request:**
```json
{
  "session_id": "sess_...",
  "analysis_type": "eda"
}
```

**Response:** Returns array of charts (base64 images + file paths), text insights, advanced statistics (Shapiro-Wilk normality test), and summary metrics.

#### `POST /analysis/preprocess`
Clean and transform data.

**Request:**
```json
{
  "session_id": "sess_...",
  "handle_missing": "auto",
  "handle_outliers": "none",
  "encode_categorical": "auto",
  "scale_numeric": "standard",
  "handle_imbalance": "none"
}
```

**Response:** Processing report with step-by-step changes, before/after shape, preview of cleaned data.

#### `POST /analysis/model`
Train and compare ML models.

**Request:**
```json
{
  "session_id": "sess_...",
  "target_column": "price",
  "model_type": "all",
  "test_size": 0.2,
  "problem_type": "auto",
  "cv_folds": 5,
  "ml_hyperparameters": {
    "n_estimators": 200,
    "max_depth": 20
  }
}
```

**Response:** Model comparison scores, cross-validation results, best model selection, charts (model comparison, CV scores, predictions vs actual, feature importance, residuals, ROC curves, prediction heatmap).

#### `POST /analysis/ann`
Train a neural network with auto-architecture.

**Request:**
```json
{
  "session_id": "sess_...",
  "target_column": "price",
  "ann_config": {
    "auto_architecture": true,
    "epochs": 150,
    "optimizer": "adamw",
    "learning_rate": 0.001
  }
}
```

**Response:** Training history charts, architecture diagram, confusion matrix/prediction plot, learning rate schedule, final scores.

#### `POST /analysis/clustering`
Run K-Means clustering with elbow & silhouette analysis.

**Request:**
```json
{
  "session_id": "sess_...",
  "analysis_type": "clustering"
}
```

**Response:** Optimal k, silhouette score, cluster profiles, PCA visualization.

### Prediction Endpoints

#### `POST /predict`
Single or batch prediction.

**Request (single):**
```json
{
  "session_id": "sess_...",
  "model_type": "best",
  "input_data": {"age": 35, "income": 75000}
}
```

**Request (batch on loaded data):**
```json
{
  "session_id": "sess_...",
  "model_type": "best",
  "input_file": "self"
}
```

**Response:** Prediction(s) with confidence scores where available.

#### `POST /predict/batch`
Batch prediction with explicit data array.

**Request:**
```json
{
  "session_id": "sess_...",
  "model_type": "best",
  "data": [{"age": 35}, {"age": 42}]
}
```

### Chat & Multi-Agent

#### `POST /chat`
Interact with the multi-agent swarm.

**Request:**
```json
{
  "session_id": "sess_...",
  "message": "What are the most important features for predicting churn?",
  "search_web": false,
  "agent_mode": true,
  "llm_provider": "llamacpp"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "sess_...",
  "intent": "ml_training",
  "active_agents": ["ml_engineer"],
  "execution_plan": [{"agent": "ml_engineer", "intent": "ml_training"}],
  "agent_outputs": {
    "ml_engineer": {"task": "...", "output": "..."}
  },
  "response": "Based on feature importance analysis...",
  "search_enabled": false,
  "citations": [],
  "available_tools": ["run_eda", "preprocess_data", "train_ml_model", "train_ann", "predict", "generate_report", "search_web"]
}
```

#### `POST /chat/stream`
Alias for `/chat` (accepts same request format, returns same response).

### Export & Download

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/sessions/{id}/export/report` | Download Markdown analysis report |
| `GET` | `/sessions/{id}/export/csv` | Download processed data as CSV |
| `GET` | `/sessions/{id}/download/{chart_name}` | Download chart PNG |
| `GET` | `/models/{session_id}/{model_name}` | Download trained model (.joblib) |

### Pydantic Model Schemas

#### `ANNConfig`
| Field | Default | Description |
|-------|---------|-------------|
| `auto_architecture` | `true` | Auto-detect layers based on data |
| `hidden_layers` | `null` | Custom layer sizes, e.g. `[128, 64, 32]` |
| `dropout_rates` | `null` | Per-layer dropout, e.g. `[0.3, 0.2, 0.1]` |
| `activation` | `"relu"` | Activation: relu, tanh, sigmoid, elu, selu |
| `output_activation` | `null` | Auto-detect if null |
| `optimizer` | `"adamw"` | adamw, adam, sgd, rmsprop |
| `learning_rate` | `0.001` | Optimizer learning rate |
| `weight_decay` | `0.004` | AdamW weight decay |
| `epochs` | `150` | Max training epochs |
| `validation_split` | `0.2` | Fraction for validation |
| `early_stopping_patience` | `20` | Early stopping patience |
| `reduce_lr_patience` | `7` | LR reduction patience |

#### `MLHyperparameters`
| Field | Default | Description |
|-------|---------|-------------|
| `n_estimators` | `200` | Trees for RF/GB/XGBoost/LightGBM |
| `max_depth` | `20` | Max tree depth |
| `svm_kernel` | `"rbf"` | SVM kernel: rbf, linear, poly, sigmoid |
| `svm_c` | `1.0` | SVM regularization |
| `knn_neighbors` | `5` | KNN neighbors |
| `ridge_alpha` | `1.0` | Ridge regularization |
| `lasso_alpha` | `0.1` | Lasso regularization |
| `xgb_learning_rate` | `0.1` | XGBoost learning rate |
| `lgb_num_leaves` | `31` | LightGBM leaves |

---

## 🏗️ Multi-Agent Architecture

### System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Server (main.py)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  Upload &   │  │   Analysis   │  │  Chat & Prediction     │  │
│  │  Session    │  │  Endpoints   │  │  Endpoints             │  │
│  │  Management │  │  (EDA/ML/    │  │  (/chat, /predict,     │  │
│  │  (/upload,  │  │  ANN/Cluster │  │   /predict/batch)      │  │
│  │  /sessions) │  │  /Preproc)   │  │                        │  │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬────────────┘  │
│         │                │                      │              │
│  ┌──────▼────────────────▼──────────────────────▼───────────┐  │
│  │                  Agent Swarm (agent.py)                    │  │
│  │                                                           │  │
│  │    ┌──────────┐ ┌────────────┐ ┌─────────────┐           │  │
│  │    │Intent    │ │ Orchestrator│ │ Synthesis   │           │  │
│  │    │Router    │ │ Agent      │ │ Engine      │           │  │
│  │    └────┬─────┘ └─────┬──────┘ └──────┬──────┘           │  │
│  │         │              │               │                  │  │
│  │    ┌────▼──────────────▼───────────────▼──────────────┐  │  │
│  │    │           Specialized Agents (MoE)                │  │  │
│  │    │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │  │  │
│  │    │  │Data    │ │Preproc │ │ML      │ │Deep    │    │  │  │
│  │    │  │Analyst │ │essor   │ │Engineer│ │Learn   │    │  │  │
│  │    │  └────────┘ └────────┘ └────────┘ └────────┘    │  │  │
│  │    │  ┌────────┐ ┌────────┐ ┌────────┐               │  │  │
│  │    │  │Visual  │ │Predict│ │Report  │               │  │  │
│  │    │  │izer    │ │or     │ │er      │               │  │  │
│  │    │  └────────┘ └────────┘ └────────┘               │  │  │
│  │    └──────────────────────────────────────────────────┘  │  │
│  │                           │                               │
│  │  ┌────────────────────────▼───────────────────────────┐  │
│  │  │           LLM Provider Layer                        │  │
│  │  │    ┌──────────┐  ┌────────┐  ┌─────────────┐      │  │
│  │  │    │llamacpp  │  │ Ollama │  │ OpenAI/     │      │  │
│  │  │    │(local)   │  │(local) │  │ Compatible  │      │  │
│  │  │    └──────────┘  └────────┘  └─────────────┘      │  │
│  │  └─────────────────────────────────────────────────────┘  │
│  └────────────────────────────────────────────────────────────┘
└──────────────────────────────────────────────────────────────────┘
```

### Agent Workflow

When you send a message via `/chat`:

1. **Intent Classification** — The Intent Router quickly classifies your request into one of 10 intents using **keyword matching** (sub-millisecond) or **LLM fallback** for ambiguous queries.

2. **Agent Activation** — Only the relevant agents are activated (Mixture-of-Experts pattern). For example, "train a random forest" activates only the ML Engineer agent, not the Visualizer or Reporter.

3. **Context Injection** — Session data (column info, health scores, model registry) and web search results (if enabled) are injected into agent prompts.

4. **Execution** — Activated agents generate their responses in parallel (order determined by priority).

5. **Synthesis** — If multiple agents were activated, the Orchestrator synthesizes their outputs into a single coherent response. Single-agent responses pass through directly.

### Intent Routing Table

| Intent | Activated Agents | Keywords (English) | Keywords (فارسی) |
|--------|-----------------|-------------------|-------------------|
| `casual_chat` | Chat | — | — |
| `data_qa` | Data Analyst | column, row, dataset, data | ستون, ردیف, داده |
| `eda` | Data Analyst, Visualizer | eda, exploratory, analyze, statistics, correlation | آنالیز, تحلیل, آمار |
| `preprocessing` | Preprocessor | clean, preprocess, missing, outlier, encode, scale | پیش‌پردازش, تمیز, نرمال |
| `ml_training` | ML Engineer | train, model, random forest, xgboost, classification | آموزش مدل, یادگیری ماشین |
| `deep_learning` | Deep Learning | neural, ann, deep learning, keras, epoch | شبکه عصبی, یادگیری عمیق |
| `visualization` | Visualizer | chart, plot, graph, visualize, heatmap, scatter | نمودار, رسم, چارت |
| `prediction` | Predictor | predict, inference, forecast | پیش‌بینی, پیشگویی |
| `report` | Reporter, Data Analyst | report, summary, export | گزارش, خلاصه |
| `web_search` | Data Analyst | search, find online, look up, latest news | جستجو, سرچ, اینترنت |

### Thinking Suppression

For models that emit internal reasoning (e.g., Qwen, DeepSeek), Dr. Data injects a special prefix (`<think>\n\n</think>\n\n`) that satisfies the model's thinking requirement while keeping the actual output clean. A post-processing regex strip removes any leaked `<think>` blocks.

---

## 🌐 Web Search Integration

The web search module (`web_search.py`) provides robust internet search capabilities with automatic failover:

### Search Backend Chain
1. **DuckDuckGo HTML** — `ddgs._text_html` (direct POST, bypasses bing-only lock)
2. **DuckDuckGo Lite** — `ddgs._text_lite`
3. **DuckDuckGo Bing** — `ddgs._text_bing` (default, may rate-limit)
4. **Raw urllib fallback** — POST directly to `lite.duckduckgo.com`

### Rate-Limit Protection
- **User-Agent rotation** across 4 realistic browser profiles
- **Exponential backoff** between retry attempts (1.5s, 2.5s, 4.5s)
- **Request timeout**: 15 seconds per attempt

### Auto-Detect Search Intent
The `/chat` endpoint automatically detects search intent from message keywords:
- English: "search", "find online", "latest", "news", "current", "trend"
- Persian: "جستجو", "سرچ", "اینترنت", "بیا پیدا کن", "جدیدترین", "اخبار"

### Context Enrichment
When search is enabled, the system:
1. Searches DuckDuckGo (up to 5 results)
2. Fetches the top 2 pages for deeper context (up to 2000 chars each)
3. Injects results into the LLM prompt as structured context
4. Returns structured citations `[{title, url, snippet}]` to the UI

---

## ⚡ Performance Tuning

### For faster responses (local GGUF models):

| Parameter | Default | Faster | Why |
|-----------|---------|--------|-----|
| `N_CTX` | 2048 | **1024** | Smaller context = less processing |
| `N_THREADS` | CPU count | **CPU count** | More threads = faster token gen |
| `n_batch` | 256 | **128** | Smaller batches = less memory |
| `use_mmap` | true | — | Memory-mapped loading (faster startup) |

Set via environment or `.env` file:

```bash
export N_CTX=1024
export N_THREADS=8
```

### Recommended small/fast GGUF models:
| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| **Qwen3.5-1.5B** (Q4_K_M) | ~1GB | ⚡ Very Fast | Good for basic tasks |
| **Qwen3.5-2B** (Q4_K_M) | ~1.3GB | ⚡ Fast | Good balance |
| **Phi-3-mini-4k** (Q4) | ~2.3GB | 🐢 Moderate | Strong reasoning |
| **Llama-3-8B** (Q4_K_M) | ~4.5GB | 🐢 Moderate | Best quality |

### Memory Usage Tips
- **Upload directory**: Uses system temp directory by default (`%TEMP%/dr_data`) to avoid permission issues
- **CSV loading**: Uses `low_memory=False` for mixed-type column detection
- **TensorFlow**: Set `TF_CPP_MIN_LOG_LEVEL=3` to suppress TF warnings
- **Matplotlib**: Uses `Agg` backend (non-interactive, memory-efficient)

---

## 📁 Project Structure

```
.
├── main.py                  # FastAPI server: all REST endpoints, session management, EDA/ML/ANN/Clustering/Prediction
├── agent.py                 # Multi-agent swarm: Intent Router, 8 specialized agents, LLM providers
├── config.py                # Configuration management, logging setup, model download utilities
├── web_search.py            # Web search module: multi-backend DuckDuckGo search with UA rotation & page fetching
├── system_prompt.txt        # (optional) Custom system prompts for agents
├── env.example              # Environment variable template
├── requirements.txt         # Python dependencies
│
├── index.html               # Dashboard UI: TailwindCSS glassmorphism design, chat interface, analysis panels
├── static/                  # Frontend assets
│   ├── ann_config.js        # ANN configurator UI: live architecture viz, layer editor, preset architectures
│   └── theme.js             # Theme configuration
│
├── models/                  # GGUF model files (auto-downloaded here)
├── uploads/                 # Uploaded datasets (temp directory)
├── outputs/                 # Generated outputs
│   ├── charts/              # PNG charts from EDA, ML, ANN, clustering
│   ├── models/              # Saved models (.joblib for sklearn, .keras for TF)
│   └── reports/             # Generated Markdown reports
└── logs/                    # Application logs (daily rotation: dr_data_YYYYMMDD.log)
    ├── dr_data_20260522.log
    └── dr_data_20260523.log
```

### File Details

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | ~3200 | All API endpoints, session management, EDA (25+ chart types), preprocessing, ML training, ANN, clustering, prediction, chat, report generation |
| `agent.py` | ~700 | AgentSwarm class, IntentRouter, 8 specialized agents (BaseAgent subclasses), 3 LLM providers (LlamaCpp, Ollama, OpenAI), think suppression, model auto-download |
| `config.py` | ~190 | Config class (paths, directories, model fallback), logging setup, HuggingFace download utility |
| `web_search.py` | ~315 | Multi-backend DDG search, raw urllib fallback, page fetcher, citation formatter |
| `index.html` | ~1220 | Single-page dashboard: upload, EDA, preprocessing, ML, ANN, clustering, chat, model download, report |
| `static/ann_config.js` | ~517 | ANN architecture configurator: canvas visualization, layer editing, presets, training progress |
| `requirements.txt` | — | Python package dependencies |

---

## 📄 License

MIT

---

*Dr. Data — Multi-Agent Data Science Swarm v2.1. Built with FastAPI, Python, and LLMs.*