import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some audio files.')
    parser.add_argument('--file_path', type=str, default='test.wav', help='Path to the audio file')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for processing')
    parser.add_argument('--hf_auth_token', type=str, required=False, help='Pass Hugging Face Auth Token or set the HF_AUTH_TOKEN environment variable')
    parser.add_argument('--save', action='store_true', help='Flag to save results to a JSON file')

    args = parser.parse_args()

    hf_auth_token = os.getenv("HF_AUTH_TOKEN", args.hf_auth_token)
    if not hf_auth_token:
        raise ValueError("Hugging Face Auth Token is required. Pass it as an argument or set the HF_AUTH_TOKEN environment variable.")

    return args, hf_auth_token
