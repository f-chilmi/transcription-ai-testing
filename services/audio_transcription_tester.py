import os

from utils.utils import diarize_text
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

from config import OUTPUT_CONFIG, HUGGING_FACE_TOKEN

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
        
        # os.environ["OMP_NUM_THREADS"] = str(threads)
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            device = "cpu"
            compute_type = "int8"
            
            # Load model and transcribe
            model = whisperx.load_model("tiny", device, compute_type=compute_type)
            audio = whisperx.load_audio(audio_path)
            # result = model.transcribe(audio, batch_size=1)

            result = self.results
            print(90, result)
            
            # # Align
            # model_a, metadata = whisperx.load_align_model(language_code="ar", device=device)
            # result = whisperx.align(result["segments"], model_a, metadata, audio, device)
            
            # print(96)

            # Diarization
            diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=device)
            print(111, diarize_model)
            diarize_segments = diarize_model(audio)
            print(113, diarize_segments)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print(115, result)
            
            # Cleanup
            del model, diarize_model
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
                'segments': result['segments'],
                'result': result,
                'diarize_segments': json.loads(json.dumps(diarize_segments, default=str)),
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

            self.results = {}
            
            model = whisperx.load_model("tiny", device, compute_type=compute_type)
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, language="en", batch_size=4)
            print(165, result["segments"])
            
            # Simple speaker labeling (heuristic)
            # for i, segment in enumerate(result['segments']):
            #     segment['speaker'] = f"Speaker_{(i // 3) % 2 + 1}"  # Alternate speakers every 3 segments
            
            del model
            gc.collect()

            # 2. Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            self.results = result
            print(178, result["segments"]) # after alignment

            del model_a
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
                'segments': result['segments'],
                'result': result
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
        """Test 3: Hybrid Pipeline (Whisper transcription + Pyannote diarization)"""
        logger.info(f"Testing Hybrid Pipeline - Whisper: {whisper_threads}, Pyannote: {diarize_threads} threads")
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            device = "cpu"
            compute_type = "int8"
            
            # Step 1: Whisper transcription
            # os.environ["OMP_NUM_THREADS"] = str(whisper_threads)

            # whisper_model = whisperx.load_model("medium", device, compute_type=compute_type)
            
            whisper_start = time.time()
            transcription_result = self.results
            # transcription_result = whisper_model.transcribe(audio_path)
            whisper_time = time.time() - whisper_start
            print(217, transcription_result)
            
            # del whisper_model
            # gc.collect()
            
            # Step 2: Pyannote diarization (separate process) 
            # os.environ["OMP_NUM_THREADS"] = str(diarize_threads)
            from pyannote.audio import Pipeline
            
            diarize_start = time.time()
            try:
                print(246)
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", 
                    use_auth_token=self.hf_token
                )
            except Exception:
                # Fallback to older version
                print(253)
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization", 
                    use_auth_token=self.hf_token
                )
            print(258, diarization_pipeline)
            diarization_result = diarization_pipeline(audio_path)
            print(260, diarization_result)
            diarize_time = time.time() - diarize_start
            
            del diarization_pipeline
            gc.collect()
            
            # Step 3: Check if diarization_result is valid
            if diarization_result is None:
                raise Exception("Diarization returned None - check audio file and HF token")
            
            final_result = diarize_text(transcription_result, diarization_result)

            print(270, final_result)

            final_result_serialized = json.loads(json.dumps(final_result, default=str))
            print(273, final_result_serialized)
            
            # waveform, sample_rate = torchaudio.load(audio_path)
            # if sample_rate != 16000:
            #     waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            
            # whisper_model = whisperx.load_model("medium", device, compute_type=compute_type)
            # segments_text = []
            # for segment, _, speaker in diarization_result.itertracks(yield_label=True):
            #     try:
            #         # Extract segment audio
            #         start, end = segment.start, segment.end
            #         segment_audio = waveform[:, int(start * 16000): int(end * 16000)]
            #         print(284, segment_audio)

            #         # Transcribe the segment
            #         transcription = whisper_model.transcribe(segment_audio, language="en")["text"]
            #         print(291, f"Transcription for {speaker}: {transcription}")

            #         temp_segment_path = f'process/segment_{speaker}_{start:.2f}_{end:.2f}_{audio_path}'
            #         print(295,' - temp_segment_path -> ', temp_segment_path )
            #         torchaudio.save(temp_segment_path, segment_audio, 16000)
            #         print(296, 'saved')

            #         # Append results
            #         segments_text.append(f"{speaker}: {transcription}")

            #     except Exception as e:
            #         print(f"Error processing segment {speaker} ({start:.2f}-{end:.2f}): {e}")
            #         # Continue with next segment instead of failing completely
            #         continue
            
            # print(301, segments_text)
            # # Clean up model
            # del whisper_model
            # gc.collect()

            # # Step 4: Print combined results
            # for text in segments_text:
            #     print(text)
            # # Manual speaker assignment based on time overlap
            # segments = transcription_result.get('segments', [])
            # if not segments:
            #     raise Exception("No transcription segments found")
            
            # Convert diarization to list for easier processing
            # diarization_segments = []
            # try:
            #     for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            #         diarization_segments.append({
            #             'start': turn.start,
            #             'end': turn.end, 
            #             'speaker': speaker
            #         })
            # except Exception as e:
            #     logger.warning(f"Diarization processing failed: {e}")
            #     # Fallback: assign simple speakers
            #     for i, segment in enumerate(segments):
            #         segment['speaker'] = f'Speaker_{i % 2 + 1}'
                
            #     speakers_detected = 2
            # else:
            #     # Normal alignment process
            #     for segment in segments:
            #         seg_start = segment['start']
            #         seg_end = segment['end']
                    
            #         # Find speaker from diarization that overlaps most
            #         best_speaker = 'Unknown'
            #         max_overlap = 0
                    
            #         for diar_seg in diarization_segments:
            #             overlap_start = max(seg_start, diar_seg['start'])
            #             overlap_end = min(seg_end, diar_seg['end'])
            #             overlap_duration = max(0, overlap_end - overlap_start)
                        
            #             if overlap_duration > max_overlap:
            #                 max_overlap = overlap_duration
            #                 best_speaker = diar_seg['speaker']
                    
            #         segment['speaker'] = best_speaker if best_speaker != 'Unknown' else 'Speaker_1'
                
            #     # Count unique speakers
            #     speakers_detected = len(set(seg.get('speaker', 'Unknown') for seg in segments))
            
            end_time = time.time()
            monitor.stop_monitoring()
            
            return {
                'method': 'hybrid_pipeline',
                'whisper_threads': whisper_threads,
                'diarize_threads': diarize_threads,
                'processing_time': end_time - start_time,
                'whisper_time': whisper_time,
                'diarize_time': diarize_time,
                # 'segments_count': len(segments),
                # 'speakers_detected': speakers_detected,
                'resource_usage': monitor.get_summary(),
                'success': True,
                # 'segments': segments_text,
                'result': self.results,
                'diarization_result': json.loads(json.dumps(diarization_result, default=str)),
                'final_result_serialized': final_result_serialized
            }
            
        except Exception as e:
            monitor.stop_monitoring()
            logger.error(f"Hybrid pipeline error: {str(e)}")
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

    def test_pyannote_diarization(self, audio_path: str, threads: int = 6) -> Dict[str, Any]:
        """Test using pure pyannote diarization (like the reference code)"""
        logger.info(f"Testing Pure Pyannote Diarization with {threads} threads")
        
        os.environ["OMP_NUM_THREADS"] = str(threads)
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            device = "cpu"
            compute_type = "int8"
            
            # Step 1: Load models separately (like reference code)
            from pyannote.audio import Pipeline
            print(480)
            # Pyannote diarization
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            print(486)
            whisper_model = whisperx.load_model("base", device, compute_type=compute_type)
            print(488)
            # Step 2: Run diarization
            diarization_start = time.time()
            diarization_result = diarization_pipeline(audio_path)
            print(492, diarization_result)
            diarization_time = time.time() - diarization_start
            print(494)
            # Step 3: Run transcription
            transcription_start = time.time() 
            transcription_result = whisper_model.transcribe(audio_path)
            print(498)
            transcription_time = time.time() - transcription_start

            for turn, _, speaker in transcription_result.itertracks(yield_label=True):
                print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
                        
            # Step 4: Merge results (need pyannote_whisper utils)
            try:
                from pyannote.audio import Pipeline
                from pyannote_whisper.utils import diarize_text
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", use_auth_token=HUGGING_FACE_TOKEN
                )
                diarization_result = pipeline()
                final_result = diarize_text(transcription_result, diarization_result)
                segments = []
                
                for seg, spk, sent in final_result:
                    segments.append({
                        'start': seg.start,
                        'end': seg.end, 
                        'speaker': spk,
                        'text': sent
                    })
            except ImportError:
                # Fallback: simple merge
                segments = transcription_result['segments']
                for segment in segments:
                    segment['speaker'] = 'Speaker_1'  # Simple fallback
            
            # Count unique speakers
            speakers_detected = len(set(seg.get('speaker', 'Unknown') for seg in segments))
            
            del diarization_pipeline, whisper_model
            gc.collect()
            
            end_time = time.time()
            monitor.stop_monitoring()
            
            return {
                'method': 'pyannote_diarization',
                'threads': threads,
                'processing_time': end_time - start_time,
                'diarization_time': diarization_time,
                'transcription_time': transcription_time,
                'segments_count': len(segments),
                'speakers_detected': speakers_detected,
                'resource_usage': monitor.get_summary(),
                'success': True,
                'segments': segments[:3]
            }
            
        except Exception as e:
            monitor.stop_monitoring()
            return {
                'method': 'pyannote_diarization', 
                'threads': threads,
                'processing_time': time.time() - start_time,
                'error': str(e),
                'success': False,
                'resource_usage': monitor.get_summary()
            }

    def test_silero_vad_transcription(self, audio_path: str, threads: int = 6) -> Dict[str, Any]:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        """Test using Silero VAD + Whisper (heuristic approach)"""
        logger.info(f"Testing Silero VAD + Whisper with {threads} threads")
        
        os.environ["OMP_NUM_THREADS"] = str(threads)
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            device = "cpu"
            
            # Step 1: Load Silero VAD (cleaner approach)
            vad_start = time.time()
            from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
            
            model = load_silero_vad()
            wav = read_audio(audio_path)
            speech_timestamps = get_speech_timestamps(
                wav,
                model,
                return_seconds=True,  # Return speech timestamps in seconds
            )
            vad_time = time.time() - vad_start
            
            # Step 2: Load Whisper
            whisper_start = time.time()
            import whisper
            whisper_model = whisper.load_model("base")
            
            # Step 3: Process speech chunks
            segments = []
            speaker_counter = 0
            
            for i, timestamp in enumerate(speech_timestamps):
                # Simple speaker assignment (heuristic)
                if i % 4 == 0:  # Change speaker every 4 segments
                    speaker_counter += 1
                
                # Create chunk info for whisper
                start_time_chunk = timestamp['start']
                end_time_chunk = timestamp['end']
                
                # Transcribe full audio but we'll use timestamps for segmentation
                if i == 0:  # Only transcribe once
                    full_result = whisper_model.transcribe(audio_path)
                
                # Find overlapping segments from whisper result
                chunk_text = ""
                for seg in full_result['segments']:
                    seg_start = seg['start']
                    seg_end = seg['end']
                    
                    # Check if whisper segment overlaps with VAD chunk
                    if (seg_start < end_time_chunk and seg_end > start_time_chunk):
                        chunk_text += " " + seg['text']
                
                if chunk_text.strip():  # Only add if there's text
                    segments.append({
                        'start': start_time_chunk,
                        'end': end_time_chunk,
                        'text': chunk_text.strip(),
                        'speaker': f'Speaker_{speaker_counter % 2 + 1}'  # Alternate between 2 speakers
                    })
            
            whisper_time = time.time() - whisper_start
            
            # Count speakers
            speakers_detected = len(set(seg['speaker'] for seg in segments))
            
            del model, whisper_model
            gc.collect()
            
            end_time = time.time()
            monitor.stop_monitoring()
            
            return {
                'method': 'silero_vad_transcription',
                'threads': threads,
                'processing_time': end_time - start_time,
                'vad_time': vad_time,
                'whisper_time': whisper_time,
                'speech_chunks_found': len(speech_timestamps),
                'segments_count': len(segments),
                'speakers_detected': speakers_detected,
                'resource_usage': monitor.get_summary(),
                'success': True,
                'segments': segments[:3]
            }
            
        except Exception as e:
            monitor.stop_monitoring()
            return {
                'method': 'silero_vad_transcription',
                'threads': threads,
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
            
            file_results['whisper_only'] = self.test_whisper_only(audio_path, threads=6)

            # file_results['silero_vad'] = self.test_silero_vad_transcription(audio_path, threads=6)

            # file_results['pyannote_diarization'] = self.test_pyannote_diarization(audio_path, threads=6)

            # Baseline (6 threads)
            file_results['baseline_6_threads'] = self.test_baseline_full_whisperx(audio_path, threads=6)
            
            # Baseline (1 thread for comparison)
            # file_results['baseline_1_thread'] = self.test_baseline_full_whisperx(audio_path, threads=1)
            
            # Hybrid pipeline
            # file_results['hybrid_pipeline'] = self.test_hybrid_pipeline(audio_path, whisper_threads=4, diarize_threads=2)
            
            # Thread scaling (only for mono audio to save time)
            # if 'mono' in audio_name.lower():
            #     file_results['thread_scaling'] = self.test_thread_scaling(audio_path)
            
            all_results['results'][audio_name] = file_results
        
        # Test 6: Batch processing (if multiple files)
        if len(audio_files) > 1:
            all_results['batch_processing'] = self.test_batch_processing(list(audio_files.values()))
        
        all_results['test_end_time'] = datetime.now().isoformat()
        
        return all_results