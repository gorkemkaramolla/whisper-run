import click
import librosa
import matplotlib.pyplot as plt
import numpy as np

from whisper_run.pyannote_onnx import PyannoteONNX


@click.command()
@click.argument("wav_path", type=click.Path(exists=True, file_okay=True))
@click.option("--plot/--no-plot", default=False, help="Plot the vad probabilities")
def main(wav_path: str, plot: bool):
    pyannote = PyannoteONNX()
    for turn in pyannote.itertracks(wav_path):
        print(turn)

    if plot:
        pyannote = PyannoteONNX(show_progress=True)
        wav, sr = librosa.load(wav_path, sr=pyannote.sample_rate)
        outputs = list(pyannote(wav))
        x1 = np.arange(0, len(wav)) / sr
        x2 = [(i * 270 + 721) / sr for i in range(0, len(outputs))]
        plt.plot(x1, wav)
        plt.plot(x2, outputs)
        plt.show()


if __name__ == "__main__":
    main()
