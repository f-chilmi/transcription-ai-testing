# Quick Test Runner - Run this to start testing immediately
import gc
import os
import time

import whisperx
from typing import Dict, Any

from services.audio_transcription import AudioTranscription
from services.audio_transcription_tester import AudioTranscriptionTest
from utils.formatter import format_transcript_file
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
        results = tester.run_performance_test(AUDIO_FILES, 'whisperx_tiny')

        print(115, 'run_performance_test')
        
        # Save and display results
        tester.save_results(results)
        print(119, 'run_performance_test')
        tester.print_summary(results)

        # print(122, 'run_performance_test')
        
        print(f"\n🎉 Performance test completed! Results saved to: {OUTPUT_CONFIG['results_filename']}")
        
    except Exception as e:
        print(f"❌ Error during performance test: {str(e)}")

def format_response(): 
    formatted = format_transcript_file(
        file_path="transcription_test_results_27-06-2025 07:54:24.json",
        output_path="formatted_conversation.txt",
        # speaker_names={'SPEAKER_00': 'Phil', 'SPEAKER_01': 'Georgie'}
    )
    print(formatted)
    return formatted


def test_transcription_diarization() -> Dict[str, Any]:
    """Test: test_whisperx_models (no diarization)"""
    # logger.info(f"Testing test_whisperx_models with {threads} threads model {model}")
    
    threads = 6
    model = 'tiny'
    audio_path = 'audio_mono_swedish.mp3'
    language = 'sv'

    os.environ["OMP_NUM_THREADS"] = str(threads)
    # monitor = ResourceMonitor()
    # monitor.start_monitoring()
    
    start_time = time.time()
    
    try:
        device = "cpu"
        compute_type = "int8"
        
        model_a = whisperx.load_model(model, device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_path)
        result = model_a.transcribe(audio, language=language, batch_size=4)
        print(165, result["segments"])
        
        del model_a
        gc.collect()

        # 2. Align whisper output
        model_b, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(result["segments"], model_b, metadata, audio, device, return_char_alignments=False)
        # self.results = result
        print(178, result["segments"]) # after alignment

        del model_b
        gc.collect()
        print(59)
        end_time_transcription = time.time()
        # monitor.stop_monitoring()
        print(62)

        model = whisperx.load_model("tiny", device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_path)

        # Diarization
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=HUGGING_FACE_TOKEN,
            device=device)
        print(111, diarize_model)
        diarize_segments = diarize_model(audio)
        print(113, diarize_segments)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print(11562, result)
        
        # Cleanup
        del model, diarize_model
        gc.collect()
        
        end_time = time.time()
        # monitor.stop_monitoring()

        tester = AudioTranscriptionTest(HUGGING_FACE_TOKEN)

        final_result = {
            'method': 'test_whisperx',
            'threads': threads,
            'processing_time_transcription': end_time_transcription - start_time,
            'processing_time_diarozation': end_time - end_time_transcription,
            'segments_count': len(result['segments']),
            'speakers_detected': len(set(seg.get('speaker', 'Unknown') for seg in result['segments'])),
            # 'resource_usage': monitor.get_summary(),
            'success': True,
            'segments': result,
        }
        tester.save_results(final_result)
        
        return final_result
            
        
        # return {
        #     'method': 'test_whisperx_models',
        #     'threads': threads,
        #     'model': model,
        #     'processing_time': end_time - start_time,
        #     'segments_count': len(result['segments']),
        #     'speakers_detected': len(set(seg.get('speaker', 'Unknown') for seg in result['segments'])),
        #     # 'resource_usage': monitor.get_summary(),
        #     'success': True,
        #     'segments': result['segments'],
        #     'result': result
        # }
        
    except Exception as e:
        # monitor.stop_monitoring()
        return {
            'method': 'whisper_only',
            'threads': threads,
            'processing_time': time.time() - start_time,
            'error': str(e),
            'success': False,
            # 'resource_usage': monitor.get_summary()
        }
        

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
    
    test_transcription_diarization()
    
    # print("\nSelect test mode:")
    # print("1. Quick test (fast verification)")
    # print("2. Full test suite (comprehensive)")
    # print("3. Install requirements only")
    # print("4. Run performance test")
    # print("5. Format")
    
    # choice = input("\nEnter choice (1-4): ").strip()
    
    # if choice == "1":
    #     if run_quick_test():
    #         print("\n🎯 Quick test passed! Ready for full testing.")
        
    # elif choice == "2":
    #     run_full_test()
        
    # elif choice == "3":
    #     install_requirements()

    # elif choice == "4":
    #     run_performance_test()
    # elif choice == "5":
    #     format_response()
        
    # else:
    #     print("Invalid choice. Please run again.")

if __name__ == "__main__":
    main()