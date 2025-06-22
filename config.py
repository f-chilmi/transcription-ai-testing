# config.py - Configuration for Audio Transcription Testing
import os
os.environ['USE_NNPACK'] = '0'
# =============================================================================
# REQUIRED SETTINGS - UPDATE THESE
# =============================================================================

# Your Hugging Face token (required for diarization)
# Get it from: https://huggingface.co/settings/tokens
from datetime import datetime
import torch
torch.backends.nnpack.enabled = False

HUGGING_FACE_TOKEN = "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"

# Your audio file paths (update these with your actual file paths)
AUDIO_FILES = {
    "mono": "audio_mono_arabic.mp3",        # Your mono-speaker audio file
    # "multi": "audio_multi_arabic.mp3",      # Your multi-speaker audio file  
    # "noisy": "audio_noisy_arabic.mp3"       # Your noisy audio file
}

# =============================================================================
# OPTIONAL SETTINGS - MODIFY AS NEEDED
# =============================================================================

# Test configurations
TEST_CONFIG = {
    # Thread counts to test
    "thread_counts": [1, 2, 4, 6],
    
    # Whisper model sizes to test (smaller = faster, larger = more accurate)
    "whisper_models": ["base"],  # Options: tiny, base, small, medium, large-v2
    
    # Batch sizes to test
    "batch_sizes": [2, 4],
    
    # Compute types to test  
    "compute_types": ["int8"],  # Options: int8, float16, float32
    
    # Which tests to run (set to False to skip)
    "run_baseline": True,
    "run_whisper_only": True, 
    "run_hybrid": True,
    "run_thread_scaling": True,
    "run_batch_processing": True,
}

# Resource monitoring settings
MONITORING_CONFIG = {
    "monitor_interval": 0.5,  # seconds between resource measurements
    "save_detailed_logs": True,
    "include_memory_usage": True,
    "include_cpu_per_core": True
}

# Output settings
OUTPUT_CONFIG = {
    "results_filename": f"transcription_test_results_{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}.json",
    "save_segments_sample": 3,  # Number of segments to save for quality checking
    "print_progress": True,
    "save_intermediate_results": True
}

# Performance targets (for comparison)
PERFORMANCE_TARGETS = {
    "max_processing_time_per_minute": 60,  # Max seconds to process 1 minute of audio
    "min_cpu_utilization": 50,  # Minimum expected CPU usage %
    "max_memory_usage": 80,  # Maximum acceptable memory usage %
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config():
    """Validate configuration before running tests"""
    issues = []
    
    # Check Hugging Face token
    if not HUGGING_FACE_TOKEN or HUGGING_FACE_TOKEN == "hf_your_token_here":
        issues.append("❌ Hugging Face token not set")
    
    # Check audio files
    import os
    for name, path in AUDIO_FILES.items():
        if not os.path.exists(path):
            issues.append(f"❌ Audio file not found: {name} -> {path}")
    
    # Check thread counts
    import psutil
    max_cores = psutil.cpu_count()
    if max(TEST_CONFIG["thread_counts"]) > max_cores:
        issues.append(f"⚠️  Thread count {max(TEST_CONFIG['thread_counts'])} > available cores {max_cores}")
    
    return issues

def print_config():
    """Print current configuration"""
    print("📋 Current Configuration:")
    print(f"   Audio files: {len(AUDIO_FILES)} files")
    print(f"   Thread counts: {TEST_CONFIG['thread_counts']}")
    print(f"   Whisper models: {TEST_CONFIG['whisper_models']}")
    print(f"   Batch sizes: {TEST_CONFIG['batch_sizes']}")
    
    enabled_tests = [k for k, v in TEST_CONFIG.items() if k.startswith('run_') and v]
    print(f"   Enabled tests: {len(enabled_tests)}")

if __name__ == "__main__":
    # Validate configuration when run directly
    issues = validate_config()
    
    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ Configuration is valid!")
        print_config()