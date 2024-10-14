import time
import json
from typing import List, Dict, Optional
from .audio_converter import AudioConverter
from .diarization_pipeline import DiarizationPipeline
from .transcription_pipeline import TranscriptionPipeline


class AudioProcessor:
    def __init__(
        self,
        file_path: str,
        device: str,
        model_name: str,
        diarization: bool = True,  # Add diarization flag
        **kwargs,
    ) -> None:
        self.file_path = file_path
        self.device = device
        self.diarization = diarization  # Store diarization preference
        self.whisper_model_name = model_name

        # Store transcription parameters (kwargs)
        self.transcription_params = kwargs

    def process(self) -> Dict[str, List[Dict[str, float]]]:
        total_start_time = time.time()

        # Convert to WAV format
        self.file_path = AudioConverter.convert_to_wav(self.file_path)

        diarization_segments = None

        # Diarization Process (only if diarization=True)
        if self.diarization:
            print("Starting diarization process...")
            diarization_pipeline = DiarizationPipeline(self.device)
            diarization_segments = diarization_pipeline.run(self.file_path)

        # Transcription Process
        print("Starting transcription process...")
        transcription_pipeline = TranscriptionPipeline(
            self.whisper_model_name, self.device
        )
        transcription_json = transcription_pipeline.run(
            self.file_path, **self.transcription_params  # Pass kwargs here
        )

        transcription_segments = json.loads(transcription_json)

        # Match diarization segments with transcription segments (only if diarization=True)
        if self.diarization and diarization_segments:
            for segment in transcription_segments:
                if (
                    isinstance(segment, dict)
                    and "start" in segment
                    and "end" in segment
                ):
                    closest_segment = min(
                        diarization_segments,
                        key=lambda x: min(
                            abs(x["start"] - segment["start"]),
                            abs(x["stop"] - segment["end"]),
                        ),
                    )
                    segment["speaker"] = (
                        closest_segment["speaker"] if closest_segment else None
                    )
                else:
                    print(f"Unexpected segment format: {segment}")

        final_result = {
            "text": " ".join(
                [
                    seg["text"]
                    for seg in transcription_segments
                    if isinstance(seg, dict) and "text" in seg
                ]
            ),
            "segments": transcription_segments,
        }

        total_elapsed_time = time.time() - total_start_time
        print(f"Total processing time: {total_elapsed_time:.2f} seconds")

        return final_result
