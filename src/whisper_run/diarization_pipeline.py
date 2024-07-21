import torch
import librosa
from typing import List, Dict
from whisper_run.pyannote_onnx import PyannoteONNX
from tqdm import tqdm
from whisper_run.utils import measure_time


class DiarizationPipeline:
    def __init__(self, device: str = "cpu", show_progress: bool = False) -> None:
        self.pipeline = PyannoteONNX(show_progress=show_progress)
        self.device = torch.device(device)
        if device != "cpu" and not torch.cuda.is_available():
            raise ValueError("CUDA device specified but CUDA is not available.")
        if device != "cpu":
            raise ValueError(
                "PyannoteONNX models are not typically moved to GPU. They run efficiently on CPU."
            )

    def run(self, file_path: str) -> List[Dict[str, float]]:
        print(f"Device: {self.device}")

        def process_audio(file_path: str) -> List[Dict[str, float]]:
            audio, sample_rate = librosa.load(
                file_path, sr=self.pipeline.sample_rate, mono=True
            )

            # Process the audio with PyannoteONNX
            diarization_results = list(self.pipeline.itertracks(audio))
            # Convert results to the desired format
            segments = []
            with tqdm(
                total=len(diarization_results),
                desc="Diarization Progress",
                unit="segment",
            ) as pbar:
                for result in diarization_results:
                    segments.append(
                        {
                            "start": result["start"],
                            "end": result["stop"],
                            "speaker": f"{result['speaker']:02d}",
                        }
                    )
                    pbar.update(1)

            return segments

        segments, elapsed_time = measure_time(process_audio, file_path)
        print(f"Diarization completed in {elapsed_time:.2f} seconds")
        return segments
