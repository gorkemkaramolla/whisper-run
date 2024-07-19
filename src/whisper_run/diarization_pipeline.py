import torch
import time
import numpy as np
import librosa
from typing import List, Dict
from whisper_run.pyannote_onnx import PyannoteONNX

class DiarizationPipeline:
    def __init__(self, device: str = "cpu", show_progress: bool = False) -> None:
        self.pipeline = PyannoteONNX(show_progress=show_progress)
        self.device = torch.device(device)
        if device != "cpu" and not torch.cuda.is_available():
            raise ValueError("CUDA device specified but CUDA is not available.")
        if device != "cpu":
            raise ValueError("PyannoteONNX models are not typically moved to GPU. They run efficiently on CPU.")

    def run(self, file_path: str) -> List[Dict[str, float]]:
        print(f"Device: {self.device}")
        start_time = time.time()

        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=self.pipeline.sample_rate, mono=True)
        print(f"Audio shape after loading: {audio.shape}")

        # Process the audio with PyannoteONNX
        diarization_results = list(self.pipeline.itertracks(audio))

        # Convert results to the desired format
        segments = []
        for result in diarization_results:
            segments.append({
                'start': result['start'],
                'end': result['stop'],
                'speaker': f"SPEAKER_{result['speaker']:02d}"
            })

        elapsed_time = time.time() - start_time
        print(f"Diarization completed in {elapsed_time:.2f} seconds")
        return segments