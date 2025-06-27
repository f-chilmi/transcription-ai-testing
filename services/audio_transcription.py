import json
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
import whisper
from faster_whisper import WhisperModel


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioTranscription:
    """Comprehensive testing suite for different transcription methods"""
    
    def __init__(self):
        self.results = {}
    
    def test_vad(self, audio_path: str, threads: int = 6) -> Dict[str, Any]:
        import ssl
        from IPython.display import Audio
        ssl._create_default_https_context = ssl._create_unverified_context
        """Test using Silero VAD + Whisper (heuristic approach)"""
        logger.info(f"Testing Silero VAD + Whisper with {threads} threads")

        SAMPLING_RATE = 16000
        
        os.environ["OMP_NUM_THREADS"] = str(threads)
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
            transcription = self.test_vad('only_speech.wav', threads)
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
    
    def test_whisperx_models(self, model: str, audio_path: str, threads: int = 6, language: str = "en") -> Dict[str, Any]:
        """Test: test_whisperx_models (no diarization)"""
        logger.info(f"Testing test_whisperx_models with {threads} threads model {model}")
        
        # os.environ["OMP_NUM_THREADS"] = str(threads)
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            device = "cpu"
            compute_type = "int8"

            self.results = {}
            
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
            end_time = time.time()
            monitor.stop_monitoring()
            print(62)
            
            return {
                'method': 'test_whisperx_models',
                'threads': threads,
                'model': model,
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
         
    def test_faster_whisper_models(self, model: str, audio_path: str, threads: int = 6, language: str = "en") -> Dict[str, Any]:
        
        """Test: WhisperX (no diarization)"""
        logger.info(f"Testing test_faster_whisper_models with {threads} threads model {model}")
        
        # os.environ["OMP_NUM_THREADS"] = str(threads)
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            device = "cpu"
            compute_type = "int8"

            self.results = {}
            
            model_a = WhisperModel(model, device=device, compute_type=compute_type)
            segments, info = model_a.transcribe(audio_path, beam_size=5, word_timestamps=True, language=language)
    
            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

            segments_list = list(segments)
            print("Segments count:", len(segments_list))
            print("First segment type:", type(segments_list[0]) if segments_list else "No segments")

            for segment in segments_list:
                for word in segment.words:
                    print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
            print(256)
            del model_a
            gc.collect()

         
            print(59, info)
            end_time = time.time()
            monitor.stop_monitoring()
            print(62)
            
            return {
                'method': 'faster_whisper',
                'threads': threads,
                'model': model,
                'processing_time': end_time - start_time,
                'resource_usage': monitor.get_summary(),
                'success': True,
                'segments':  [
                    {
                        'id': segment.id,
                        'start': float(segment.start),
                        'end': float(segment.end),
                        'text': segment.text,
                        'words': [
                            {
                                'start': float(word.start),
                                'end': float(word.end),
                                'word': word.word,
                                'probability': float(word.probability)
                            } for word in segment.words
                        ] if hasattr(segment, 'words') else []
                    } for segment in segments_list
                ],
                'info': json.loads(json.dumps(info, default=str)),
            }
            
        except Exception as e:
            monitor.stop_monitoring()
            return {
                'method': 'faster_whisper',
                'threads': threads,
                'processing_time': time.time() - start_time,
                'error': str(e),
                'success': False,
                'resource_usage': monitor.get_summary()
            }
        
    def test_faster_whisper_vad_models(self, model: str, audio_path: str, threads: int = 6, language: str = "en") -> Dict[str, Any]:
        
        """Test: test_faster_whisper_vad_models (no diarization)"""
        logger.info(f"Testing test_faster_whisper_vad_models with {threads} threads model {model}")
        
        # os.environ["OMP_NUM_THREADS"] = str(threads)
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            device = "cpu"
            compute_type = "int8"

            self.results = {}
            
            model_a = WhisperModel(model, device=device, compute_type=compute_type)
            segments, info = model_a.transcribe(audio_path, beam_size=5, word_timestamps=True, vad_filter=True,language=language)
            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            
            segments_list = list(segments)
            for segment in segments:
                for word in segment.words:
                    print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

            del model_a
            gc.collect()

            whisperx_segments = []
            for seg in segments_list:
                if hasattr(seg, 'words') and seg.words:
                    # Use word-level data instead of segment-level
                    segment_words = []
                    for word in seg.words:
                        segment_words.append({
                            'start': float(word.start),
                            'end': float(word.end),
                            'text': word.word.strip(),
                            'score': float(word.probability)
                        })
                    
                    whisperx_segments.append({
                        'start': float(seg.start),
                        'end': float(seg.end),
                        'text': seg.text,
                        'words': segment_words
                    })
                else:
                    # Fallback to segment-level if no words
                    whisperx_segments.append({
                        'start': float(seg.start),
                        'end': float(seg.end),
                        'text': seg.text
                    })

            audio = whisperx.load_audio(audio_path)
            model_align, metadata = whisperx.load_align_model(language_code=language, device="cpu")
            aligned_result = whisperx.align(whisperx_segments, model_align, metadata, audio, "cpu")
            del model_align

            word_segments = []
            for i, word_data in enumerate(aligned_result['word_segments']):
                word_segments.append({
                    'start': word_data['start'],
                    'end': word_data['end'],
                    'text': word_data['word'].strip(),
                    'words': [{
                        'start': word_data['start'],
                        'end': word_data['end'],
                        'text': word_data['word'].strip(),
                        'score': word_data.get('score', 1.0)
                    }]
                })

            self.results = word_segments

            print(59)
            end_time = time.time()
            monitor.stop_monitoring()
            print(62, info)
            
            return {
                'method': 'test_faster_whisper_vad_models',
                'threads': threads,
                'model': model,
                'processing_time': end_time - start_time,
                'resource_usage': monitor.get_summary(),
                'success': True,
                'segments':  [
                    {
                        'id': segment.id,
                        'start': float(segment.start),
                        'end': float(segment.end),
                        'text': segment.text,
                        'words': [
                            {
                                'start': float(word.start),
                                'end': float(word.end),
                                'word': word.word,
                                'probability': float(word.probability)
                            } for word in segment.words
                        ] if hasattr(segment, 'words') else []
                    } for segment in segments_list
                ],
                'aligned_result': aligned_result,
                'result': word_segments,
            }
            
            
        except Exception as e:
            monitor.stop_monitoring()
            return {
                'method': 'test_faster_whisper_vad_models',
                'threads': threads,
                'processing_time': time.time() - start_time,
                'error': str(e),
                'success': False,
                'resource_usage': monitor.get_summary()
            }