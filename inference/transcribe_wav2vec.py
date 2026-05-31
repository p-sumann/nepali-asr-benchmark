"""
Run a Nepali Wav2Vec 2.0-family model (IndicWav2Vec, XLSR-53, or MMS-1B) on a single audio file.

Usage:
    python transcribe_wav2vec.py /path/to/clip.{wav,mp3,flac} \
        --model sumanpaudel1997/nepali-asr-indicwav2vec

Audio is resampled to 16 kHz mono. Output is printed to stdout.
"""

import argparse
import librosa
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC


def transcribe(audio_path: str, model_id: str, device: str | None = None) -> str:
    proc = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    model.to(device)

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = proc(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return proc.batch_decode(pred_ids)[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="path to audio file")
    p.add_argument(
        "--model",
        default="sumanpaudel1997/nepali-asr-indicwav2vec",
        help="HF model id (one of: nepali-asr-indicwav2vec / nepali-asr-xlsr-53 / nepali-asr-mms-1b)",
    )
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detected)")
    args = p.parse_args()
    print(transcribe(args.audio, args.model, args.device))


if __name__ == "__main__":
    main()
