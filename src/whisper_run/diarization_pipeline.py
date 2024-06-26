import torch
import time
from typing import List, Dict
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
class DiarizationPipeline:
    def __init__(self, model_name: str, auth_token: str, device: str) -> None:
        self.pipeline = PyannotePipeline.from_pretrained(model_name, use_auth_token=auth_token)
        self.device = torch.device(device)
        self.pipeline.to(self.device)

    def run(self, file_path: str) -> List[Dict[str, float]]:
        print(self.device)
        self.pipeline.to(torch.device(self.device))
        start_time = time.time()
        with ProgressHook() as hook:
            diarization = self.pipeline(file_path, hook=hook)
        segments = [
            {'start': round(turn.start, 2), 'end': round(turn.end, 2), 'speaker': speaker}
            for turn, _, speaker in sorted(diarization.itertracks(yield_label=True), key=lambda x: x[0].start)
        ]
        elapsed_time = time.time() - start_time
        print(f"Diarization completed in {elapsed_time:.2f} seconds")
        return segments