import os
import json
import logging
import inquirer
from whisper_run.audio_processor import AudioProcessor
from whisper_run.config import whisper_models
from whisper_run.arg_parser import parse_arguments  # Use relative import

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def main():
    try:
        args = parse_arguments()
    except ValueError as e:
        logging.error(e)
        return

    model_options = {
        f"{model} (fastest)" if model == "distil-whisper/distil-large-v3" else f"{model} (recommended)" if model == "openai/whisper-large-v3" else model: model
        for model in whisper_models
    }

    questions = [
        inquirer.List('model',
                      message="Select a model for audio processing",
                      choices=list(model_options.keys()),
                      ),
    ]
    answers = inquirer.prompt(questions)
    model = model_options[answers['model']]

    file_path = args.file_path
    device = args.device

    processor = AudioProcessor(file_path, device, model)
    results = processor.process()

    results_json = json.dumps(results, indent=4)
    print(results_json)

    if args.save:
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
        results_json_path = os.path.join(results_dir, f"{file_name_without_extension}.json")
        with open(results_json_path, "w") as results_json_file:
            json.dump(results, results_json_file, indent=4)
        print(f"Results saved to {results_json_path}")

if __name__ == "__main__":
    main()
