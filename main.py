# Quick Test Runner - Run this to start testing immediately
import os

from services.audio_transcription import AudioTranscription
from services.audio_transcription_tester import AudioTranscriptionTest
os.environ['USE_NNPACK'] = '0'
import torch
torch.backends.nnpack.enabled = False
from config import AUDIO_FILES, HUGGING_FACE_TOKEN, OUTPUT_CONFIG



def check_files():
    """Check if audio files exist"""
    missing_files = []
    for name, path in AUDIO_FILES.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("❌ Missing audio files:")
        for missing in missing_files:
            print(f"   - {missing}")
        print("\n💡 Please update the AUDIO_FILES dictionary with correct paths")
        return False
    
    print("✅ All audio files found")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    packages = [
        "whisperx",
        "psutil", 
        "torch",
        "torchaudio",
        "pyannote.audio",
        "openai-whisper",
        "silero-vad"
    ]
    
    os.system("pip install https://github.com/alunkingusw/pyannote-whisper/archive/main.zip")
    
    for package in packages:
        os.system(f"pip install {package}")
    
    print("✅ Packages installed")

def run_quick_test():
    """Run a quick test on one file to verify setup"""
    print("🧪 Running quick verification test...")
    
    try:
        
        
        tester = AudioTranscription(HUGGING_FACE_TOKEN)
        
        # Test with the first available file
        first_audio = list(AUDIO_FILES.values())[0]
        print(f"Testing with: {first_audio}")
        
        # Quick whisper-only test
        result = tester.test_whisper_tiny(first_audio, threads=2)
        
        if result['success']:
            print(f"✅ Quick test successful! Processed in {result['processing_time']:.1f}s")
            print(f"   Found {result['segments_count']} segments")
            return True
        else:
            print(f"❌ Quick test failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during quick test: {str(e)}")
        return False

def run_full_test():
    """Run the complete test suite"""
    print("🚀 Starting full test suite...")
    
    try:
        
        tester = AudioTranscriptionTest(HUGGING_FACE_TOKEN)
        
        # Run comprehensive test
        results = tester.run_comparison_test(list(AUDIO_FILES.values())[0], language="en")

        print(90, 'run_full_test')
        
        # Save and display results
        tester.save_results(results)
        print(94, 'run_full_test')
        # tester.print_summary(results)

        print(97, 'run_full_test')
        
        print(f"\n🎉 Full test completed! Results saved to: {OUTPUT_CONFIG['results_filename']}")
        
    except Exception as e:
        print(f"❌ Error during full test: {str(e)}")

def run_performance_test():
    """Run performance test suite"""
    print("🚀 Starting performance test suite...")
    
    try:
        
        tester = AudioTranscriptionTest(HUGGING_FACE_TOKEN)
        
        # Run comprehensive test
        results = tester.run_performance_test(AUDIO_FILES)

        print(115, 'run_performance_test')
        
        # Save and display results
        tester.save_results(results)
        print(119, 'run_performance_test')
        tester.print_summary(results)

        # print(122, 'run_performance_test')
        
        print(f"\n🎉 Performance test completed! Results saved to: {OUTPUT_CONFIG['results_filename']}")
        
    except Exception as e:
        print(f"❌ Error during performance test: {str(e)}")

def main():
    print("🎯 Audio Transcription Testing Suite")
    print("=" * 50)
    
    # Check if token is set
    if HUGGING_FACE_TOKEN == "hf_your_token_here":
        print("❌ Please set your Hugging Face token in HUGGING_FACE_TOKEN variable")
        return
    
    # Check files
    if not check_files():
        return
    
    print("\nSelect test mode:")
    print("1. Quick test (fast verification)")
    print("2. Full test suite (comprehensive)")
    print("3. Install requirements only")
    print("4. Run performance test")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        if run_quick_test():
            print("\n🎯 Quick test passed! Ready for full testing.")
        
    elif choice == "2":
        run_full_test()
        
    elif choice == "3":
        install_requirements()

    elif choice == "4":
        run_performance_test()
        
    else:
        print("Invalid choice. Please run again.")

if __name__ == "__main__":
    main()