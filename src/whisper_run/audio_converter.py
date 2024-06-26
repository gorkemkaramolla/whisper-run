from pydub import AudioSegment
import os
class AudioConverter:
    @staticmethod
    def convert_to_wav(file_path: str) -> str:
        original = AudioSegment.from_file(file_path)
        wav_path = f"{os.path.splitext(file_path)[0]}.wav"
        original.export(wav_path, format='wav')
        return wav_path

