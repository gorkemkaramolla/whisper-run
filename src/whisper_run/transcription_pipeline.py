import time
import json
from faster_whisper import WhisperModel
from typing import Dict, List, Any

class TranscriptionPipeline:
    def __init__(self, model_size: str, device: str) -> None:
        self.model = WhisperModel(model_size, device=device, compute_type="int8")

    def run(self, file_path: str) -> List[Dict[str, Any]]:
        start_time = time.time()
        segments, info = self.model.transcribe(file_path, beam_size=5)
        end_time = time.time()

        total_runtime = end_time - start_time
        print(f"Transcription completed in {total_runtime:.2f} seconds")

        transcription_json_start_time = time.time()

        transcription_result = [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments
        ]

        transcription_json_end_time = time.time()
        transcription_json_total = transcription_json_end_time - transcription_json_start_time
        print(f"Transcription JSON preparation {transcription_json_total:.2f} seconds")

        return transcription_result


