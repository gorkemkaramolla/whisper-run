import os
from whisper_run import AudioProcessor


def main():
    model_dir = os.path.join(os.path.dirname(__file__), "large-v3/")
    faster_whisper_params = {
        "language": "tr",
        "beam_size": 5,
    }
    processor = AudioProcessor(
        file_path="../../audios/test.wav",
        device="cpu",
        diarization=True,
        # pass a model dir or available model name from faster-whisper
        model_name=model_dir,
        # model_name = "large-v3",
        ##Kwargs here
        **faster_whisper_params,
    )

    # results in JSON format
    result = processor.process()
    print(result)


if __name__ == "__main__":
    main()
