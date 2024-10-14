import torch
from typing import Dict, List, Any, NamedTuple
import orjson
from faster_whisper import WhisperModel
from whisper_run.utils import measure_time
import multiprocessing
import time


def prepare_segment(segment) -> Dict[str, Any]:
    segment_dict = segment._asdict()
    segment_dict.pop("tokens", None)
    return segment_dict


class TranscriptionPipeline:
    def __init__(
        self,
        model_size: str,
        device: str = "cuda",
    ) -> None:
        self.device = device
        print(f"Initializing WhisperModel on device: {self.device}")
        self.model = WhisperModel(model_size, device=self.device, compute_type="int8")

    def run(self, file_path: str, **kwargs) -> str:
        """Run the transcription process."""
        print("Starting transcription process...")
        start_time = time.time()

        segments, info = self.model.transcribe(file_path, **kwargs)

        # Output from faster-whisper
        # for segment in segments:
        #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

        segments_list = list(segments)

        if not segments_list:
            print("No transcription segments found.")
            return orjson.dumps({"text": "", "segments": []}).decode("utf-8")

        transcription_duration = time.time() - start_time
        transcription_duration = max(transcription_duration, 1e-6)

        print(
            f"\n\nFinished! Speed: {info.duration / transcription_duration:.2f} audio seconds/s"
        )

        # Parallel JSON preparation
        def prepare_json(segments: List) -> List[Dict[str, Any]]:
            with multiprocessing.Pool() as pool:
                result = pool.map(prepare_segment, segments)
                return result

        transcription_result, json_total_runtime = measure_time(
            prepare_json, segments_list
        )
        print(
            f"Transcription JSON preparation completed in {json_total_runtime:.2f} seconds"
        )

        json_string, encoding_runtime = measure_time(orjson.dumps, transcription_result)
        print(f"JSON encoding completed in {encoding_runtime:.2f} seconds")

        return json_string.decode("utf-8")
