import os
from whisper_run import AudioProcessor


def main():
    model_dir = os.path.join(os.path.dirname(__file__), "large-v3/")
    processor = AudioProcessor(
        file_path="../../audios/short.wav", device="cpu", model_name="large-v3"
    )
    print(processor.process())


if __name__ == "__main__":
    main()
