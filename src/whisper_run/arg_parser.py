import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some audio files.')
    parser.add_argument('--file_path', type=str, default='test.wav', help='Path to the audio file')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for processing')
    parser.add_argument('--save', action='store_true', help='Flag to save results to a JSON file')

    args = parser.parse_args()
    return args
