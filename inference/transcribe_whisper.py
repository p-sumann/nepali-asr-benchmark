"""
Run Nepali Whisper (Medium or Large-v3-Turbo) on a single audio file.

Usage:
    python transcribe_whisper.py /path/to/clip.{wav,mp3,flac} \
        --model sumanpaudel1997/nepali-asr-whisper-turbo

Audio is resampled to 16 kHz mono. Output is printed to stdout.
"""

import argparse
import librosa
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration


def transcribe(audio_path: str, model_id: str, device: str | None = None) -> str:
    proc = AutoProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    model.to(device)

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
    forced_ids = proc.get_decoder_prompt_ids(language="ne", task="transcribe")

    with torch.no_grad():
        gen = model.generate(
            inputs.input_features.to(device),
            forced_decoder_ids=forced_ids,
            max_new_tokens=448,
        )
    text = proc.batch_decode(gen, skip_special_tokens=True)[0]
    return text.strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="path to audio file")
    p.add_argument(
        "--model",
        default="sumanpaudel1997/nepali-asr-whisper-turbo",
        help="HF model id (default: nepali-asr-whisper-turbo)",
    )
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detected)")
    args = p.parse_args()
    print(transcribe(args.audio, args.model, args.device))


if __name__ == "__main__":
    main()
