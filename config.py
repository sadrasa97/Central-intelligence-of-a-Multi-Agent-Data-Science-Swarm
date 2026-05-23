"""
================================================================================
Configuration and Logging Setup for Dr. Data Platform
================================================================================
"""
import os
import logging
from pathlib import Path
from datetime import datetime

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Create logs directory
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
def setup_logging():
    """Setup comprehensive logging for the application"""
    log_filename = LOGS_DIR / f"dr_data_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simple)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress noisy libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    logging.info("="*80)
    logging.info("Dr. Data Platform - Logging Initialized")
    logging.info(f"Log file: {log_filename}")
    logging.info("="*80)


def download_model_from_hf(repo_id: str, filename: str, local_path: Path) -> bool:
    """
    Download a GGUF model from HuggingFace Hub
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "Qwen/Qwen3.5-2B-GGUF")
        filename: Model filename in the repo
        local_path: Where to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import hf_hub_download
        logging.info(f"Downloading model from HuggingFace: {repo_id}/{filename}")
        logging.info(f"Target location: {local_path}")
        
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_path.parent,
            local_dir_use_symlinks=False
        )
        
        # Move to expected location if needed
        if Path(downloaded_path) != local_path:
            import shutil
            shutil.move(downloaded_path, local_path)
        
        logging.info(f"Model downloaded successfully to {local_path}")
        return True
        
    except ImportError:
        logging.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        logging.error(f"Failed to download model: {str(e)}")
        return False


def ensure_model_available(model_path: str, auto_download: bool = True) -> tuple[bool, str]:
    """
    Ensure a GGUF model is available, downloading if necessary
    
    Args:
        model_path: Path to the model
        auto_download: Whether to auto-download if missing
        
    Returns:
        (success: bool, message: str)
    """
    path = Path(model_path)
    
    if path.exists():
        return True, f"Model found at {path}"
    
    if not auto_download:
        return False, f"Model not found at {path} and auto-download disabled"
    
    logging.warning(f"Model not found at {path}, attempting auto-download...")
    
    # Try to download the default model
    success = download_model_from_hf(
        repo_id=Config.DEFAULT_HF_REPO,
        filename=Config.DEFAULT_HF_FILENAME,
        local_path=path
    )
    
    if success:
        return True, f"Model downloaded successfully to {path}"
    else:
        return False, (
            f"Model not found and auto-download failed. "
            f"Please download manually from https://huggingface.co/{Config.DEFAULT_HF_REPO} "
            f"or install huggingface_hub: pip install huggingface_hub"
        )

# Application configuration
class Config:
    # Model configuration - Check multiple locations
    MODELS_STORAGE_DIR = BASE_DIR / "models"
    MODELS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Default model path - check D:\models first, then local models folder
    DEFAULT_MODEL_PATHS = [
        Path(r"D:\models\Qwen3.5-2B-Q4_K_S.gguf"),
        MODELS_STORAGE_DIR / "Qwen3.5-2B-Q4_K_S.gguf",
        BASE_DIR / "Qwen3.5-2B-Q4_K_S.gguf",
    ]
    
    MODEL_PATH = os.environ.get("MODEL_PATH", "")
    if not MODEL_PATH:
        for path in DEFAULT_MODEL_PATHS:
            if path.exists():
                MODEL_PATH = str(path)
                break
        else:
            # Use the local models folder as default download location
            MODEL_PATH = str(MODELS_STORAGE_DIR / "Qwen3.5-2B-Q4_K_S.gguf")
    
    # HuggingFace model fallback configuration
    DEFAULT_HF_REPO = "Qwen/Qwen3.5-2B-GGUF"
    DEFAULT_HF_FILENAME = "qwen3-5-2b-q4_k_s.gguf"
    
    # Directories - Use temp directory for uploads to avoid permission issues
    TEMP_BASE = Path(os.environ.get("TEMP", os.environ.get("TMP", "C:\\Temp"))) / "dr_data"
    UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", str(TEMP_BASE / "uploads")))
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "outputs"))
    CHARTS_DIR = OUTPUT_DIR / "charts"
    MODELS_DIR = OUTPUT_DIR / "models"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    
    # Ensure directories exist
    for directory in [UPLOAD_DIR, OUTPUT_DIR, CHARTS_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Server configuration
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 8000))
    DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
    
    # ML/ANN Configuration
    ANN_MAX_LAYERS = 10
    ANN_MAX_NEURONS_PER_LAYER = 1024
    ANN_MIN_NEURONS_PER_LAYER = 8
    
    # API Configuration
    MAX_UPLOAD_SIZE_MB = 500
    SESSION_TIMEOUT_HOURS = 24
