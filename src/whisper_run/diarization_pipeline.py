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
        # Ensure that we use CPU for PyannoteONNX
        self.pipeline = PyannoteONNX(show_progress=show_progress)
        self.device = torch.device(device)
        self.progress_callback = progress_callback

        print(
            f"Using device: {self.device} for general processing, CPU for PyannoteONNX"
        )

    def run(self, file_path: str) -> List[Dict[str, float]]:
        start_time = time.time()

        # Load audio file
        audio, sample_rate = librosa.load(
            file_path, sr=self.pipeline.sample_rate, mono=True
        )
        print(f"Audio shape after loading: {audio.shape}")

        # Process the audio with PyannoteONNX on CPU
        diarization_results = []
        total_frames = len(audio)
        processed_frames = 0

        for result in self.pipeline.itertracks(audio):
            diarization_results.append(result)
            processed_frames += 1
            if self.progress_callback:
                self.progress_callback("Processing", processed_frames, total_frames)

        # Convert results to the desired format
        segments = []
        for result in diarization_results:
            segments.append(
                {
                    "start": result["start"],
                    "end": result["stop"],
                    "speaker": f"SPEAKER_{result['speaker']:02d}",
                }
            )

        elapsed_time = time.time() - start_time
        print(f"Diarization completed in {elapsed_time:.2f} seconds")
        return segments
