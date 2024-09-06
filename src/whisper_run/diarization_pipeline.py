import torch
import librosa
import time
from typing import List, Dict, Callable
from whisper_run.pyannote_onnx import PyannoteONNX


class DiarizationPipeline:
    def __init__(
        self,
        device: str = "cuda",
        show_progress: bool = True,
        progress_callback: Callable[[str, int, int], None] = None,
    ) -> None:
        self.pipeline = PyannoteONNX(show_progress=show_progress)
        self.device = torch.device(device)
        self.progress_callback = progress_callback

        print(
            f"Using device: {self.device} for general processing, CPU for PyannoteONNX"
        )

    def run(self, file_path: str) -> List[Dict[str, float]]:
        start_time = time.time()

        audio, sample_rate = librosa.load(
            file_path, sr=self.pipeline.sample_rate, mono=True
        )
        print(f"Audio shape after loading: {audio.shape}")

        diarization_results = list(self.pipeline.itertracks(audio))
        total_frames = len(audio)
        processed_frames = len(diarization_results)

        if self.progress_callback:
            self.progress_callback("Processing", processed_frames, total_frames)

        elapsed_time = time.time() - start_time
        print(f"Diarization completed in {elapsed_time:.2f} seconds")

        return diarization_results
