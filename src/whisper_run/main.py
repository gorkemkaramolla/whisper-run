from whisper_run import AudioProcessor

def main():
    processor = AudioProcessor(file_path="../../test.wav",
                               device="cpu",
                               model_name="large-v3")
    print(processor.process())

if __name__ == "__main__":
    main()