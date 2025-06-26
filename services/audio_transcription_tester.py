import os

from services.audio_diarization import AudioDiarization
from services.audio_transcription import AudioTranscription
from services.resource_monitor import ResourceMonitor
os.environ['USE_NNPACK'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import time
import json
import logging
import psutil
from datetime import datetime
from typing import Dict, List, Any
import torch
torch.backends.nnpack.enabled = False

from config import OUTPUT_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioTranscriptionTest:
    """Comprehensive testing suite for different transcription methods"""
    
    def __init__(self, hugging_face_token: str):
        self.hf_token = hugging_face_token
        self.results = {}
        self.transcription_service = AudioTranscription()
        self.diarization_service = AudioDiarization(hugging_face_token)
   
    def test_thread_scaling(self, audio_path: str, transcription_method: str = 'whisperx_tiny') -> Dict[str, Any]:
        """Test thread scaling with existing transcription methods"""
        logger.info(f"Testing thread scaling with {transcription_method}")
        
        thread_results = {}
        
        for threads in [1, 2, 4, 6, 8]:
            logger.info(f"Testing with {threads} threads")
            
            if transcription_method == 'whisperx_tiny':
                result = self.transcription_service.test_whisperx_models('tiny', audio_path, threads)
            elif transcription_method == 'faster_whisper_tiny':
                result = self.transcription_service.test_faster_whisper_models('tiny', audio_path, threads)
            elif transcription_method == 'whisper_tiny':
                result = self.transcription_service.test_whisper_models('tiny', audio_path, threads)

            print(50, 'transcription done')

            self.diarization_service.set_transcription_results(result)
            diarization = self.diarization_service.test_whisperx(audio_path, threads)

            print(57, 'diarization done')
            
            # Store both results together
            thread_results[threads] = {
                'transcription': result,
                'diarization': diarization,
                'total_time': result.get('processing_time', 0) + diarization.get('processing_time', 0)
            }
        
        return {
            'method': 'thread_scaling',
            'transcription_method': transcription_method,
            'results': thread_results
        }
    
    def test_batch_processing(self, audio_files: List[str], transcription_method: str = 'whisperx_tiny') -> Dict[str, Any]:
        """Test batch processing using existing transcription methods"""
        logger.info(f"Testing batch processing with {transcription_method}")
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            results = []
            for audio_file in audio_files:
                if transcription_method == 'whisperx_tiny':
                    transcription = self.transcription_service.test_whisperx_models('tiny', audio_file)
                elif transcription_method == 'faster_whisper_tiny':
                    transcription = self.transcription_service.test_faster_whisper_models('tiny', audio_file)
                elif transcription_method == 'whisper_tiny':
                    transcription = self.transcription_service.test_whisper_models('tiny', audio_file)
                
                print('transcription done for', audio_file)
                
                self.diarization_service.set_transcription_results(transcription)
                diarization = self.diarization_service.test_whisperx(audio_file)
                
                print('diarization done for', audio_file)
                
                results.append({
                    'file': audio_file,
                    'transcription': transcription,
                    'diarization': diarization,
                    'total_time': transcription.get('processing_time', 0) + diarization.get('processing_time', 0)
                })
            
            end_time = time.time()
            monitor.stop_monitoring()
            
            return {
                'method': 'batch_processing',
                'transcription_method': transcription_method,
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
                'transcription_method': transcription_method,
                'processing_time': time.time() - start_time,
                'error': str(e),
                'success': False,
                'resource_usage': monitor.get_summary()
            }
        
    def test_concurrent_processing(self, audio_files: List[str], transcription_method: str = 'whisperx_tiny') -> Dict[str, Any]:
        """Test concurrent processing using existing transcription methods"""
        logger.info(f"Testing concurrent processing with {transcription_method}")
        
        import concurrent.futures
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        def process_single_file(audio_file):
            try:
                if transcription_method == 'whisperx_tiny':
                    transcription = self.transcription_service.test_whisperx_models('tiny', audio_file)
                elif transcription_method == 'faster_whisper_tiny':
                    transcription = self.transcription_service.test_faster_whisper_models('tiny', audio_file)
                elif transcription_method == 'whisper_tiny':
                    transcription = self.transcription_service.test_whisper_models('tiny', audio_file)
                
                print('transcription done for', audio_file)
                
                self.diarization_service.set_transcription_results(transcription)
                diarization = self.diarization_service.test_whisperx(audio_file)
                
                print('diarization done for', audio_file)
                
                return {
                    'file': audio_file,
                    'transcription': transcription,
                    'diarization': diarization,
                    'total_time': transcription.get('processing_time', 0) + diarization.get('processing_time', 0),
                    'success': True
                }
            except Exception as e:
                return {
                    'file': audio_file,
                    'error': str(e),
                    'success': False
                }
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(process_single_file, audio_files))
            
            end_time = time.time()
            monitor.stop_monitoring()
            
            return {
                'method': 'concurrent_processing',
                'transcription_method': transcription_method,
                'files_processed': len(audio_files),
                'processing_time': end_time - start_time,
                'resource_usage': monitor.get_summary(),
                'results': results,
                'success': True
            }
            
        except Exception as e:
            monitor.stop_monitoring()
            return {
                'method': 'concurrent_processing',
                'transcription_method': transcription_method,
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

        results['whisperx_tiny'] = self.transcription_service.test_whisperx_models('tiny', audio_path)
        results['whisperx_base'] = self.transcription_service.test_whisperx_models('base', audio_path)
        results['whisperx_small'] = self.transcription_service.test_whisperx_models('small', audio_path)
        results['whisperx_medium'] = self.transcription_service.test_whisperx_models('medium', audio_path)
        results['whisperx_large'] = self.transcription_service.test_whisperx_models('large', audio_path)
        results['whisperx_turbo'] = self.transcription_service.test_whisperx_models('turbo', audio_path)

        results['faster_whisper_tiny'] = self.transcription_service.test_faster_whisper_models('tiny', audio_path)
        results['faster_whisper_base'] = self.transcription_service.test_faster_whisper_models('base', audio_path)
        results['faster_whisper_small'] = self.transcription_service.test_faster_whisper_models('small', audio_path)
        results['faster_whisper_medium'] = self.transcription_service.test_faster_whisper_models('medium', audio_path)
        results['faster_whisper_large'] = self.transcription_service.test_faster_whisper_models('large', audio_path)
        results['faster_whisper_turbo'] = self.transcription_service.test_faster_whisper_models('turbo', audio_path)

        results['faster_whisper_vad_tiny'] = self.transcription_service.test_faster_whisper_vad_models('tiny', audio_path)
        results['faster_whisper_vad_base'] = self.transcription_service.test_faster_whisper_vad_models('base', audio_path)
        results['faster_whisper_vad_small'] = self.transcription_service.test_faster_whisper_vad_models('small', audio_path)
        results['faster_whisper_vad_medium'] = self.transcription_service.test_faster_whisper_vad_models('medium', audio_path)
        results['faster_whisper_vad_large'] = self.transcription_service.test_faster_whisper_vad_models('large', audio_path)
        results['faster_whisper_vad_turbo'] = self.transcription_service.test_faster_whisper_vad_models('turbo', audio_path)

        # # Test 1: VAD
        # results['vad'] = self.transcription_service.test_vad(audio_path)
        
        # # Test 2: Whisper
        # results['whisper'] = self.transcription_service.test_whisper_tiny(audio_path)
        
        # # Test 3: WhisperX diarization
        # self.diarization_service.set_transcription_results(self.transcription_service.results)
        # results['whisperx_diarization'] = self.diarization_service.test_whisperx(audio_path)
        
        # # Test 4: Pyannote diarization
        # self.diarization_service.set_transcription_results(self.transcription_service.results)
        # results['pyannote_diarization'] = self.diarization_service.test_pyannote(audio_path)
        
        return results
    
    def run_performance_test(self, audio_files: Dict[str, str], transcription_method: str = 'whisperx_tiny'):
        """Run performance tests using existing transcription methods"""
        results = {}
        
        first_audio = list(audio_files.values())[0]
        
        # Thread scaling test
        results['thread_scaling'] = self.test_thread_scaling(first_audio, transcription_method)
        
        # Batch and concurrent processing
        if len(audio_files) > 1:
            audio_list = list(audio_files.values())
            results['batch_processing'] = self.test_batch_processing(audio_list, transcription_method)
            results['concurrent_processing'] = self.test_concurrent_processing(audio_list, transcription_method)
        
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
  