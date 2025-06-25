import os

from services.audio_diarization import AudioDiarization
from services.audio_transcription import AudioTranscription
from utils.utils import diarize_text, serialize_diarization_result
os.environ['USE_NNPACK'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
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
import torchaudio
from collections import OrderedDict
from pyannote.core import Segment

from config import OUTPUT_CONFIG, HUGGING_FACE_TOKEN

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiarizationEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Segment):
            return {'start': float(obj.start), 'end': float(obj.end)}
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

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

class AudioTranscriptionTest:
    """Comprehensive testing suite for different transcription methods"""
    
    def __init__(self, hugging_face_token: str):
        self.hf_token = hugging_face_token
        self.results = {}
        self.transcription_service = AudioTranscription()
        self.diarization_service = AudioDiarization(hugging_face_token)
   
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
    
    def save_results(self, results: Dict[str, Any], filename: str = OUTPUT_CONFIG['results_filename']):
        """Save test results to JSON file"""
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

    def run_comparison_test(self, audio_path: str):
        """Run comprehensive comparison test"""
        results = {}
        
        # Test 1: VAD
        results['vad'] = self.transcription_service.test_vad(audio_path)
        
        # Test 2: Whisper
        results['whisper'] = self.transcription_service.test_whisper_tiny(audio_path)
        
        # Test 3: WhisperX diarization
        self.diarization_service.set_transcription_results(self.transcription_service.results)
        results['whisperx_diarization'] = self.diarization_service.test_whisperx(audio_path)
        
        # Test 4: Pyannote diarization
        self.diarization_service.set_transcription_results(self.transcription_service.results)
        results['pyannote_diarization'] = self.diarization_service.test_pyannote(audio_path)
        
        return results
    
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
            
            file_results['whisper_only'] = self.test_whisper_only(audio_path, threads=6)

            # file_results['pyannote_diarization'] = self.test_pyannote_diarization(audio_path, threads=6)

            # Baseline (6 threads)
            # file_results['baseline_6_threads'] = self.test_baseline_full_whisperx(audio_path, threads=6)

            # Baseline (1 thread for comparison)
            # file_results['baseline_1_thread'] = self.test_baseline_full_whisperx(audio_path, threads=1)
            
            # Hybrid pipeline
            file_results['hybrid_pipeline'] = self.test_hybrid_pipeline(audio_path, whisper_threads=4, diarize_threads=2)

            # file_results['silero_vad'] = self.test_silero_vad_transcription(audio_path, threads=6)
            
            # Thread scaling (only for mono audio to save time)
            # if 'mono' in audio_name.lower():
            #     file_results['thread_scaling'] = self.test_thread_scaling(audio_path)
            
            all_results['results'][audio_name] = file_results
        
        # Test 6: Batch processing (if multiple files)
        if len(audio_files) > 1:
            all_results['batch_processing'] = self.test_batch_processing(list(audio_files.values()))
        
        all_results['test_end_time'] = datetime.now().isoformat()
        
        return all_results
  