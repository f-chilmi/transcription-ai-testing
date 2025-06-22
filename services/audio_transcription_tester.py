import os
os.environ['USE_NNPACK'] = '0'
import time
import json
import logging
import psutil
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import whisperx
import gc
import torch
torch.backends.nnpack.enabled = False

from config import OUTPUT_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitors CPU, memory, and other system resources during processing"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'timestamps': []
        }
    
    def start_monitoring(self):
        self.monitoring = True
        self.stats = {'cpu_usage': [], 'memory_usage': [], 'timestamps': []}
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def _monitor(self):
        while self.monitoring:
            self.stats['cpu_usage'].append(psutil.cpu_percent(percpu=True))
            self.stats['memory_usage'].append(psutil.virtual_memory().percent)
            self.stats['timestamps'].append(time.time())
            time.sleep(0.5)  # Monitor every 0.5 seconds
    
    def get_summary(self):
        if not self.stats['cpu_usage']:
            return {}
        
        avg_cpu_per_core = [sum(core_usage)/len(core_usage) for core_usage in zip(*self.stats['cpu_usage'])]
        max_cpu_per_core = [max(core_usage) for core_usage in zip(*self.stats['cpu_usage'])]
        
        return {
            'avg_cpu_per_core': avg_cpu_per_core,
            'max_cpu_per_core': max_cpu_per_core,
            'avg_memory': sum(self.stats['memory_usage']) / len(self.stats['memory_usage']),
            'max_memory': max(self.stats['memory_usage']),
            'total_cpu_cores': len(avg_cpu_per_core),
            'monitoring_duration': self.stats['timestamps'][-1] - self.stats['timestamps'][0] if self.stats['timestamps'] else 0
        }

class AudioTranscriptionTester:
    """Comprehensive testing suite for different transcription methods"""
    
    def __init__(self, hugging_face_token: str):
        self.hf_token = hugging_face_token
        self.results = {}
        
    def test_baseline_full_whisperx(self, audio_path: str, threads: int = 6) -> Dict[str, Any]:
        """Test 1: Full WhisperX + Diarization (Baseline)"""
        logger.info(f"Testing Baseline Full WhisperX with {threads} threads")
        
        os.environ["OMP_NUM_THREADS"] = str(threads)
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            device = "cpu"
            compute_type = "int8"
            
            # Load model and transcribe
            model = whisperx.load_model("base", device, compute_type=compute_type)  # Using base for speed
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, batch_size=4)

            print(90, result)
            
            # Align
            model_a, metadata = whisperx.load_align_model(language_code="ar", device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device)
            
            print(96)

            # Diarization
            diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=self.hf_token, device=device)
            print(100)
            diarize_segments = diarize_model(audio)
            print(102)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print(104)
            
            # Cleanup
            del model, model_a, diarize_model
            gc.collect()
            
            end_time = time.time()
            monitor.stop_monitoring()
            
            return {
                'method': 'baseline_full_whisperx',
                'threads': threads,
                'processing_time': end_time - start_time,
                'segments_count': len(result['segments']),
                'speakers_detected': len(set(seg.get('speaker', 'Unknown') for seg in result['segments'])),
                'resource_usage': monitor.get_summary(),
                'success': True,
                'segments': result['segments'][:3]  # Store first 3 segments for quality check
            }
            
        except Exception as e:
            monitor.stop_monitoring()
            return {
                'method': 'baseline_full_whisperx',
                'threads': threads,
                'processing_time': time.time() - start_time,
                'error': str(e),
                'success': False,
                'resource_usage': monitor.get_summary()
            }
    
    def test_whisper_only(self, audio_path: str, threads: int = 6) -> Dict[str, Any]:
        """Test 2: WhisperX only (no diarization)"""
        logger.info(f"Testing WhisperX only with {threads} threads")
        
        os.environ["OMP_NUM_THREADS"] = str(threads)
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            device = "cpu"
            compute_type = "int8"
            
            model = whisperx.load_model("base", device, compute_type=compute_type)
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, batch_size=4)
            
            # Simple speaker labeling (heuristic)
            for i, segment in enumerate(result['segments']):
                segment['speaker'] = f"Speaker_{(i // 3) % 2 + 1}"  # Alternate speakers every 3 segments
            
            del model
            gc.collect()
            
            end_time = time.time()
            monitor.stop_monitoring()
            
            return {
                'method': 'whisper_only',
                'threads': threads,
                'processing_time': end_time - start_time,
                'segments_count': len(result['segments']),
                'speakers_detected': len(set(seg.get('speaker', 'Unknown') for seg in result['segments'])),
                'resource_usage': monitor.get_summary(),
                'success': True,
                'segments': result['segments'][:3]
            }
            
        except Exception as e:
            monitor.stop_monitoring()
            return {
                'method': 'whisper_only',
                'threads': threads,
                'processing_time': time.time() - start_time,
                'error': str(e),
                'success': False,
                'resource_usage': monitor.get_summary()
            }
    
    def test_hybrid_pipeline(self, audio_path: str, whisper_threads: int = 4, diarize_threads: int = 2) -> Dict[str, Any]:
        """Test 3: Hybrid Pipeline (Whisper first, then diarization)"""
        logger.info(f"Testing Hybrid Pipeline - Whisper: {whisper_threads}, Diarize: {diarize_threads} threads")
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            device = "cpu"
            compute_type = "int8"
            
            # Step 1: Whisper transcription
            os.environ["OMP_NUM_THREADS"] = str(whisper_threads)
            model = whisperx.load_model("base", device, compute_type=compute_type)
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, batch_size=4)
            
            whisper_time = time.time()
            del model
            gc.collect()
            
            # Step 2: Diarization (can be done offline)
            os.environ["OMP_NUM_THREADS"] = str(diarize_threads)
            diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=self.hf_token, device=device)
            diarize_segments = diarize_model(audio)
            
            # Simple assignment (normally you'd do proper alignment)
            for segment in result['segments']:
                segment['speaker'] = 'Speaker_1'  # Simplified for testing
            
            del diarize_model
            gc.collect()
            
            end_time = time.time()
            monitor.stop_monitoring()
            
            return {
                'method': 'hybrid_pipeline',
                'whisper_threads': whisper_threads,
                'diarize_threads': diarize_threads,
                'processing_time': end_time - start_time,
                'whisper_time': whisper_time - start_time,
                'diarize_time': end_time - whisper_time,
                'segments_count': len(result['segments']),
                'speakers_detected': len(set(seg.get('speaker', 'Unknown') for seg in result['segments'])),
                'resource_usage': monitor.get_summary(),
                'success': True,
                'segments': result['segments'][:3]
            }
            
        except Exception as e:
            monitor.stop_monitoring()
            return {
                'method': 'hybrid_pipeline',
                'whisper_threads': whisper_threads,
                'diarize_threads': diarize_threads,
                'processing_time': time.time() - start_time,
                'error': str(e),
                'success': False,
                'resource_usage': monitor.get_summary()
            }
    
    def test_thread_scaling(self, audio_path: str) -> Dict[str, Any]:
        """Test 4: Thread scaling analysis"""
        logger.info("Testing thread scaling (1, 2, 4, 6 threads)")
        
        thread_results = {}
        
        for threads in [1, 2, 4, 6]:
            logger.info(f"Testing with {threads} threads")
            os.environ["OMP_NUM_THREADS"] = str(threads)
            
            monitor = ResourceMonitor()
            monitor.start_monitoring()
            
            start_time = time.time()
            
            try:
                device = "cpu"
                model = whisperx.load_model("base", device, compute_type="int8")
                audio = whisperx.load_audio(audio_path)
                result = model.transcribe(audio, batch_size=2)
                
                del model
                gc.collect()
                
                end_time = time.time()
                monitor.stop_monitoring()
                
                thread_results[threads] = {
                    'processing_time': end_time - start_time,
                    'segments_count': len(result['segments']),
                    'resource_usage': monitor.get_summary(),
                    'success': True
                }
                
            except Exception as e:
                monitor.stop_monitoring()
                thread_results[threads] = {
                    'processing_time': time.time() - start_time,
                    'error': str(e),
                    'success': False,
                    'resource_usage': monitor.get_summary()
                }
        
        return {
            'method': 'thread_scaling',
            'results': thread_results
        }
    
    def test_batch_processing(self, audio_files: List[str], batch_size: int = 4, threads: int = 6) -> Dict[str, Any]:
        """Test 5: Batch processing multiple files"""
        logger.info(f"Testing batch processing with {len(audio_files)} files")
        
        os.environ["OMP_NUM_THREADS"] = str(threads)
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            device = "cpu"
            model = whisperx.load_model("base", device, compute_type="int8")
            
            results = []
            for audio_file in audio_files:
                audio = whisperx.load_audio(audio_file)
                result = model.transcribe(audio, batch_size=batch_size)
                results.append({
                    'file': audio_file,
                    'segments_count': len(result['segments'])
                })
            
            del model
            gc.collect()
            
            end_time = time.time()
            monitor.stop_monitoring()
            
            return {
                'method': 'batch_processing',
                'batch_size': batch_size,
                'threads': threads,
                'files_processed': len(audio_files),
                'processing_time': end_time - start_time,
                'avg_time_per_file': (end_time - start_time) / len(audio_files),
                'resource_usage': monitor.get_summary(),
                'results': results,
                'success': True
            }
            
        except Exception as e:
            monitor.stop_monitoring()
            return {
                'method': 'batch_processing',
                'batch_size': batch_size,
                'threads': threads,
                'files_processed': len(audio_files),
                'processing_time': time.time() - start_time,
                'error': str(e),
                'success': False,
                'resource_usage': monitor.get_summary()
            }
    
    def run_comprehensive_test(self, audio_files: Dict[str, str]) -> Dict[str, Any]:
        """Run all tests on all audio files"""
        logger.info("Starting comprehensive test suite")
        
        all_results = {
            'test_start_time': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': os.sys.version
            },
            'audio_files': audio_files,
            'results': {}
        }
        
        # Test each audio file with different methods
        for audio_name, audio_path in audio_files.items():
            logger.info(f"Testing audio file: {audio_name}")
            
            file_results = {}
            
            # Test 1: Baseline (6 threads)
            file_results['baseline_6_threads'] = self.test_baseline_full_whisperx(audio_path, threads=6)
            
            # Test 2: Baseline (1 thread for comparison)
            file_results['baseline_1_thread'] = self.test_baseline_full_whisperx(audio_path, threads=1)
            
            # Test 3: WhisperX only
            file_results['whisper_only'] = self.test_whisper_only(audio_path, threads=6)
            
            # Test 4: Hybrid pipeline
            file_results['hybrid_pipeline'] = self.test_hybrid_pipeline(audio_path, whisper_threads=4, diarize_threads=2)
            
            # Test 5: Thread scaling (only for mono audio to save time)
            if 'mono' in audio_name.lower():
                file_results['thread_scaling'] = self.test_thread_scaling(audio_path)
            
            all_results['results'][audio_name] = file_results
        
        # Test 6: Batch processing (if multiple files)
        if len(audio_files) > 1:
            all_results['batch_processing'] = self.test_batch_processing(list(audio_files.values()))
        
        all_results['test_end_time'] = datetime.now().isoformat()
        
        return all_results
    
    # def save_results(self, results: Dict[str, Any], filename: str = OUTPUT_CONFIG.results_filename):
    #     """Save test results to JSON file"""
    #     with open(filename, 'w', encoding='utf-8') as f:
    #         json.dump(results, f, indent=2, ensure_ascii=False)
    #     logger.info(f"Results saved to {filename}")
    def save_results(self, results: Dict[str, Any], filename: str = OUTPUT_CONFIG["results_filename"]):
        """Save test results to JSON file"""

        # Remove 'words' from each segment (if exists)
        for test in results.get("results", {}).values():
            for variant in test.values():
                for segment in variant.get("segments", []):
                    segment.pop("words", None)

        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filename}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of test results"""
        print("\n" + "="*60)
        print("TRANSCRIPTION TESTING SUMMARY")
        print("="*60)
        
        print(f"System: {results['system_info']['cpu_count']} CPU cores, "
              f"{results['system_info']['memory_total_gb']:.1f}GB RAM")
        
        for audio_name, file_results in results['results'].items():
            print(f"\n📁 {audio_name.upper()}")
            print("-" * 40)
            
            for test_name, test_result in file_results.items():
                if test_result.get('success', False):
                    time_taken = test_result.get('processing_time', 0)
                    segments = test_result.get('segments_count', 0)
                    speakers = test_result.get('speakers_detected', 0)
                    
                    cpu_avg = test_result.get('resource_usage', {}).get('avg_cpu_per_core', [])
                    cpu_usage = f"{sum(cpu_avg)/len(cpu_avg):.1f}%" if cpu_avg else "N/A"
                    
                    print(f"✅ {test_name}: {time_taken:.1f}s | {segments} segments | "
                          f"{speakers} speakers | CPU: {cpu_usage}")
                else:
                    print(f"❌ {test_name}: FAILED - {test_result.get('error', 'Unknown error')}")
        
        # Performance comparison
        print(f"\n📊 PERFORMANCE COMPARISON")
        print("-" * 40)
        
        for audio_name, file_results in results['results'].items():
            print(f"\n{audio_name}:")
            times = []
            for test_name, result in file_results.items():
                if result.get('success') and 'processing_time' in result:
                    times.append((test_name, result['processing_time']))
            
            if times:
                times.sort(key=lambda x: x[1])
                fastest = times[0]
                print(f"  🏆 Fastest: {fastest[0]} ({fastest[1]:.1f}s)")
                
                if len(times) > 1:
                    slowest = times[-1]
                    speedup = slowest[1] / fastest[1]
                    print(f"  🐌 Slowest: {slowest[0]} ({slowest[1]:.1f}s) - {speedup:.1f}x slower")

# Usage example
if __name__ == "__main__":
    # Initialize tester
    HF_TOKEN = "your_hugging_face_token_here"  # Replace with your token
    tester = AudioTranscriptionTester(HF_TOKEN)
    
    # Define your audio files
    audio_files = {
        "mono": "audio_mono.wav",      # Replace with your file paths
        "multi": "audio_multi.wav",    # Replace with your file paths
        "noisy": "audio_noisy.wav"     # Replace with your file paths
    }
    
    # Run comprehensive test
    results = tester.run_comprehensive_test(audio_files)
    
    # Save and display results
    tester.save_results(results)
    tester.print_summary(results)
    
    print(f"\n📄 Detailed results saved to: {OUTPUT_CONFIG.results_filename}")