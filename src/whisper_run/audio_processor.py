import time
from typing import List, Dict
import os
from .audio_converter import AudioConverter
from .diarization_pipeline import DiarizationPipeline
from .transcription_pipeline import TranscriptionPipeline
class AudioProcessor:
    def __init__(self, file_path: str, device: str,hf_auth_token :str,model_name:str) -> None:
        self.file_path = file_path
        self.device = device
        self.hf_auth_token =hf_auth_token 
        self.pyannote_model_name = "pyannote/speaker-diarization-3.1"
        self.whisper_model_name = model_name

    def process(self) -> Dict[str, List[Dict[str, float]]]:
        total_start_time = time.time()

        self.file_path = AudioConverter.convert_to_wav(self.file_path)

        diarization_pipeline = DiarizationPipeline(self.pyannote_model_name, self.hf_auth_token, self.device)
        diarization_segments = diarization_pipeline.run(self.file_path)

        transcription_pipeline = TranscriptionPipeline(self.whisper_model_name, self.device)
        transcription_segments = transcription_pipeline.run(self.file_path)
        import json

        for segment in transcription_segments:
            closest_segment = min(
                diarization_segments,
                key=lambda x: min(abs(x['start'] - segment['start']), abs(x['end'] - segment['end']))
            )
            segment['speaker'] = closest_segment['speaker'] if closest_segment else None

        final_result = {
            'text': ' '.join([seg['text'] for seg in transcription_segments]),
            'segments': transcription_segments
        }

        total_elapsed_time = time.time() - total_start_time
        print(f"Total processing time: {total_elapsed_time:.2f} seconds")
           
        return final_result