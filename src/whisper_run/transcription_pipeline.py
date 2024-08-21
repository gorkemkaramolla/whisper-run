# from typing import Dict, List, Any, NamedTuple
# import orjson
# from faster_whisper import WhisperModel
# from whisper_run.utils import measure_time
# import multiprocessing
# from tqdm import tqdm
# import time
# import sys
# import io
# from threading import Thread


# class Segment(NamedTuple):
#     start: float
#     end: float
#     text: str


# def prepare_segment(segment) -> Dict[str, Any]:
#     return {"start": segment.start, "end": segment.end, "text": segment.text}


# class TranscriptionPipeline:
#     def __init__(self, model_size: str, device: str) -> None:
#         self.model = WhisperModel(model_size, device=device, compute_type="int8")

#     def run(self, file_path: str) -> str:
#         def transcribe(file_path: str) -> List:
#             segments, info = self.model.transcribe(file_path, beam_size=5)
#             return segments, info

#         print("Starting transcription process...")
#         segments, info = transcribe(file_path)

#         # Convert the segments generator to a list
#         segments_list = list(segments)

#         if not segments_list:
#             print("No transcription segments found.")
#             return orjson.dumps({"text": "", "segments": []}).decode("utf-8")

#         # Progress bar related variables
#         total_dur = round(info.duration)
#         td_len = str(len(str(total_dur)))
#         global timestamp_prev, timestamp_last
#         timestamp_prev = 0
#         timestamp_last = 0
#         capture = io.StringIO()
#         last_burst = 0.0
#         set_delay = 0.1

#         def pbar_delayed():
#             global timestamp_prev
#             time.sleep(set_delay)
#             pbar.update(timestamp_last - timestamp_prev)
#             timestamp_prev = timestamp_last
#             print(capture.getvalue().splitlines()[-1])
#             sys.stdout.flush()

#         s_time = time.time()
#         print("")
#         bar_f = (
#             "{percentage:3.0f}% | {n_fmt:>"
#             + td_len
#             + "}/{total_fmt} | {elapsed}<<{remaining} | {rate_noinv_fmt}"
#         )

#         with tqdm(
#             file=capture,
#             total=total_dur,
#             unit=" audio seconds",
#             smoothing=0.00001,
#             bar_format=bar_f,
#         ) as pbar:
#             for segment in segments_list:
#                 timestamp_last = round(segment.end)
#                 time_now = time.time()
#                 if time_now - last_burst > set_delay:
#                     last_burst = time_now
#                     Thread(target=pbar_delayed, daemon=False).start()
#             time.sleep(set_delay + 0.3)
#             if timestamp_last < total_dur:
#                 pbar.update(total_dur - timestamp_last)
#                 print(capture.getvalue().splitlines()[-1])
#                 sys.stdout.flush()

#         print(
#             "\n\nFinished! Speed: %s audio seconds/s"
#             % round(info.duration / ((time.time() - s_time)), 2)
#         )

#         # Adding tqdm progress bar for JSON preparation
#         def prepare_json(segments: List) -> List[Dict[str, Any]]:
#             with multiprocessing.Pool() as pool:
#                 result = []
#                 for prepared_segment in tqdm(
#                     pool.imap_unordered(prepare_segment, segments),
#                     total=len(segments),
#                     desc="Preparing JSON",
#                 ):
#                     result.append(prepared_segment)
#                 return result

#         transcription_result, json_total_runtime = measure_time(
#             prepare_json, segments_list
#         )
#         print(
#             f"Transcription JSON preparation completed in {json_total_runtime:.2f} seconds"
#         )

#         # Use orjson for faster JSON encoding
#         json_string, encoding_runtime = measure_time(orjson.dumps, transcription_result)
#         print(f"JSON encoding completed in {encoding_runtime:.2f} seconds")

#         return json_string.decode("utf-8")


# # Example usage
# if __name__ == "__main__":
#     pipeline = TranscriptionPipeline


import torch
from typing import Dict, List, Any, NamedTuple
import orjson
from faster_whisper import WhisperModel
from whisper_run.utils import measure_time
import multiprocessing
import time


class Segment(NamedTuple):
    start: float
    end: float
    text: str


def prepare_segment(segment) -> Dict[str, Any]:
    return {"start": segment.start, "end": segment.end, "text": segment.text}


class TranscriptionPipeline:
    def __init__(
        self,
        model_size: str,
        device: str = "cuda",
    ) -> None:
        self.device = device
        print(f"Initializing WhisperModel on device: {self.device}")
        self.model = WhisperModel(model_size, device=self.device, compute_type="int8")

    def run(self, file_path: str) -> str:
        print("Starting transcription process...")
        start_time = time.time()

        # Ensure that the input file is loaded on the correct device
        segments, info = self.model.transcribe(file_path, beam_size=5)
        segments_list = list(segments)

        if not segments_list:
            print("No transcription segments found.")
            return orjson.dumps({"text": "", "segments": []}).decode("utf-8")

        transcription_duration = time.time() - start_time
        transcription_duration = max(
            transcription_duration, 1e-6
        )  # Avoid division by zero

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

        # Use orjson for faster JSON encoding
        json_string, encoding_runtime = measure_time(orjson.dumps, transcription_result)
        print(f"JSON encoding completed in {encoding_runtime:.2f} seconds")

        return json_string.decode("utf-8")
