import os

from services.resource_monitor import ResourceMonitor
from utils.utils import diarize_text, serialize_diarization_result
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

class AudioDiarization:
    """Comprehensive testing suite for different transcription methods"""
    
    def __init__(self, hugging_face_token: str):
        self.hf_token = hugging_face_token
        self.results = {}
        self.transcription_service = None
    
    def set_transcription_results(self, results):
        """Set transcription results from AudioTranscription"""
        self.results = results

    def test_whisperx(self, audio_path: str, threads: int = 6) -> Dict[str, Any]:
        """Test: Full WhisperX + Diarization (Baseline)"""
        logger.info(f"Testing Baseline Full WhisperX with {threads} threads")
        
        # os.environ["OMP_NUM_THREADS"] = str(threads)
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            device = "cpu"
            compute_type = "int8"

            result = self.results
            print(90, result)
            
            # Load model and transcribe
            model = whisperx.load_model("tiny", device, compute_type=compute_type)
            audio = whisperx.load_audio(audio_path)

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
                'method': 'test_whisperx',
                'threads': threads,
                'processing_time': end_time - start_time,
                'segments_count': len(result['segments']),
                'speakers_detected': len(set(seg.get('speaker', 'Unknown') for seg in result['segments'])),
                'resource_usage': monitor.get_summary(),
                'success': True,
                'segments': result['segments'],
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
    
    def test_pyannote(self, audio_path: str, whisper_threads: int = 4, diarize_threads: int = 2) -> Dict[str, Any]:
        """Test: Hybrid Pipeline (Whisper transcription + Pyannote diarization)"""
        logger.info(f"Testing Hybrid Pipeline - Whisper: {whisper_threads}, Pyannote: {diarize_threads} threads")
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            transcription_result = self.results
            print(217, transcription_result)
            
            from pyannote.audio import Pipeline
            
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", 
                use_auth_token=self.hf_token
            )
            print(258, diarization_pipeline)
            diarization_result = diarization_pipeline(audio_path)
            print(260, diarization_result)
            
            del diarization_pipeline
            gc.collect()
            
            if diarization_result is None:
                raise Exception("Diarization returned None - check audio file and HF token")
            
            final_result = diarize_text(transcription_result, diarization_result)

            print(270, final_result)

            final_result_serialized = serialize_diarization_result(final_result)
            print(273, final_result_serialized)
     
            end_time = time.time()
            monitor.stop_monitoring()
            
            return {
                'method': 'hybrid_pipeline',
                'whisper_threads': whisper_threads,
                'diarize_threads': diarize_threads,
                'processing_time': end_time - start_time,
                'resource_usage': monitor.get_summary(),
                'success': True,
                'result': self.results,
                'final_result_serialized': final_result_serialized,
                'speakers_detected': len(set(seg.get('speaker', 'Unknown') for seg in final_result_serialized)),
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
