from whisper_run import AudioProcessor

def main():
    processor = AudioProcessor(file_path="../../test.wav",
                               device="cpu",
                               model_name="./model_dir")
    print(processor.process())

if __name__ == "__main__":
    main()