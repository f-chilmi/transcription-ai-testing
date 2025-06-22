import logging
import whisperx

logger = logging.getLogger(__name__)



class STTService:
    def __init__(self):
        pass
    
    def getStt(self): 
        device = "cpu"
        # audio_file = "audio2.mp3"
        audio_file = "audio3.mp4"
        batch_size = 4 # reduce if low on GPU mem
        compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

        # 1. Transcribe with original whisper (batched)
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        print(21, model)

        # save model to local path (optional)
        # model_dir = "/path/"
        # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

        audio = whisperx.load_audio(audio_file)
        print(28, audio)
        result = model.transcribe(audio, batch_size=batch_size)
        print(30)
        print(result["segments"]) # before alignment

        # delete model if low on GPU resources
        # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code="ar", device=device)
        # model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        print(38)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        print(result["segments"]) # after alignment

        # delete model if low on GPU resources
        # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

        # 3. Assign speaker labels
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=settings.hugging_face_api_key, device=device)
        print(48,)

        # add min/max number of speakers if known
        diarize_segments = diarize_model(audio)
        print(52)
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

        result = whisperx.assign_word_speakers(diarize_segments, result)
        print(diarize_segments)
        print(result["segments"]) # segments are now assigned speaker IDs
        return result
            