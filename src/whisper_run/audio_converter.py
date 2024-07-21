from pydub import AudioSegment
from whisper_run.utils import measure_time
import os

class AudioConverter:
    @staticmethod
    def convert_to_wav(file_path: str) -> str:
        def conversion(file_path: str) -> str:
            original = AudioSegment.from_file(file_path)
            wav_path = f"{os.path.splitext(file_path)[0]}.wav"
            original.export(wav_path, format='wav')
            return wav_path
        
        wav_path, conversion_time = measure_time(conversion, file_path)
        print(f"Converted file saved at: {wav_path}")
        print(f"Conversion completed in {conversion_time:.2f} seconds")
        return wav_path
