import os

from services.resource_monitor import ResourceMonitor

os.environ['USE_NNPACK'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import time
import logging
from typing import Dict, Any
import whisperx
import gc
import torch
torch.backends.nnpack.enabled = False


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioTranscription:
    """Comprehensive testing suite for different transcription methods"""
    
    def __init__(self):
        self.results = {}
    
    def test_whisper_tiny(self, audio_path: str, threads: int = 6) -> Dict[str, Any]:
        """Test: WhisperX tiny (no diarization)"""
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
            
            del model
            gc.collect()

            # 2. Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            self.results = result
            print(178, result["segments"]) # after alignment

            del model_a
            gc.collect()
            print(59)
            end_time = time.time()
            monitor.stop_monitoring()
            print(62)
            
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
        
    def test_vad(self, audio_path: str, threads: int = 6) -> Dict[str, Any]:
        import ssl
        from IPython.display import Audio
        ssl._create_default_https_context = ssl._create_unverified_context
        """Test using Silero VAD + Whisper (heuristic approach)"""
        logger.info(f"Testing Silero VAD + Whisper with {threads} threads")

        SAMPLING_RATE = 16000
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            
            from silero_vad import (load_silero_vad,
                          read_audio,
                          get_speech_timestamps,
                          save_audio,
                          collect_chunks)
            
            model = load_silero_vad()

            wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
            print(589, wav)
            predicts = model.audio_forward(wav, sr=SAMPLING_RATE)
            print(571, 'predicts', predicts)
            
            speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
            print(592, speech_timestamps)

            save_audio('only_speech.wav',
                collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE) 
            print(597)
            Audio('only_speech.wav')
            print(599)

            del model
            gc.collect()

            print(128)

            end_time = time.time()
            monitor.stop_monitoring()
            print(131)
            transcription = self.test_whisper_tiny('only_speech.wav', threads)
            print(135)
            
            return {
                'method': 'test_vad',
                'threads': threads,
                'processing_time': end_time - start_time,
                'speech_chunks_found': len(speech_timestamps),
                'resource_usage': monitor.get_summary(),
                'success': True,
                'transcription': transcription,
                'result': transcription,
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