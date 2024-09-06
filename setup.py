from setuptools import find_packages, setup


def read(filename):
    with open(filename, encoding="utf-8") as f:
        return f.read()


def load_requirements(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


setup(
    name="whisper-run",
    version="1.2.63",
    author="Görkem Karamolla",
    author_email="gorkemkaramolla@gmail.com",
    description="Whisper with speaker diarization",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/gorkemkaramolla/whisper-run",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=load_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "whisper_run": ["*.onnx"],  # Include ONNX files in the package
    },
    entry_points={"console_scripts": ["whisper-run=whisper_run.__main__:main"]},
)
