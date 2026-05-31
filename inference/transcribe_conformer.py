"""
Run Conformer-CTC Hindi (fine-tuned on Nepali) on a single audio file.

This checkpoint is a NeMo single-file (.nemo) model and requires
nvidia/nemo-toolkit. Install with:

    pip install nemo_toolkit[asr]

Usage:
    python transcribe_conformer.py /path/to/clip.{wav,mp3,flac}
"""

import argparse
from huggingface_hub import hf_hub_download

import nemo.collections.asr as nemo_asr   # noqa: E402


def transcribe(audio_path: str, repo_id: str = "sumanpaudel1997/nepali-asr-conformer-hi") -> str:
    ckpt = hf_hub_download(repo_id=repo_id, filename="best_model.nemo")
    asr = nemo_asr.models.ASRModel.restore_from(ckpt)
    asr.eval()
    return asr.transcribe([audio_path])[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="path to audio file (16 kHz mono recommended)")
    p.add_argument("--model", default="sumanpaudel1997/nepali-asr-conformer-hi")
    args = p.parse_args()
    print(transcribe(args.audio, args.model))


if __name__ == "__main__":
    main()
