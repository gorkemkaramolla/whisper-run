from typing import Dict, List, Any, NamedTuple
import orjson
from faster_whisper import WhisperModel
from whisper_run.utils import measure_time
import multiprocessing


class Segment(NamedTuple):
    start: float
    end: float
    text: str


def prepare_segment(segment) -> Dict[str, Any]:
    return {"start": segment.start, "end": segment.end, "text": segment.text}


class TranscriptionPipeline:
    def __init__(self, model_size: str, device: str) -> None:
        self.model = WhisperModel(model_size, device=device, compute_type="int8")

    def run(self, file_path: str) -> str:
        def transcribe(file_path: str) -> List:
            return self.model.transcribe(file_path, beam_size=5)

        (segments, info), total_runtime = measure_time(transcribe, file_path)
        print(f"Transcription completed in {total_runtime:.2f} seconds")

        def prepare_json(segments: List) -> List[Dict[str, Any]]:
            with multiprocessing.Pool() as pool:
                return pool.map(prepare_segment, segments)

        transcription_result, json_total_runtime = measure_time(prepare_json, segments)
        print(
            f"Transcription JSON preparation completed in {json_total_runtime:.2f} seconds"
        )

        # Use orjson for faster JSON encoding
        json_string, encoding_runtime = measure_time(orjson.dumps, transcription_result)
        print(f"JSON encoding completed in {encoding_runtime:.2f} seconds")

        return json_string.decode(
            "utf-8"
        )  # orjson returns bytes, so we decode to string
