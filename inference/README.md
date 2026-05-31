# Inference Scripts

Minimal, ready-to-run scripts that load any of the six released Nepali ASR
checkpoints from the Hugging Face Hub and transcribe a single audio file.

Audio is automatically resampled to 16 kHz mono. Output is printed to stdout in
Devanagari.

## Setup

```bash
pip install -U transformers librosa torch
# only for Conformer-Hi:
pip install nemo_toolkit[asr]
```

## Whisper family

```bash
python transcribe_whisper.py clip.wav \
    --model sumanpaudel1997/nepali-asr-whisper-turbo
```

The `--model` flag also accepts `sumanpaudel1997/nepali-asr-whisper-medium`.

## Wav2Vec 2.0 family (IndicWav2Vec / XLSR-53 / MMS-1B)

```bash
python transcribe_wav2vec.py clip.wav \
    --model sumanpaudel1997/nepali-asr-indicwav2vec
```

Also supports `sumanpaudel1997/nepali-asr-xlsr-53` and
`sumanpaudel1997/nepali-asr-mms-1b`.

## Conformer-Hi

```bash
python transcribe_conformer.py clip.wav
```

Internally downloads the `.nemo` single-file checkpoint from
`sumanpaudel1997/nepali-asr-conformer-hi` and runs NVIDIA NeMo's
`ASRModel.transcribe()`.

## Notes

- Inputs do not need pre-resampling — `librosa.load(..., sr=16000)` handles it.
- All checkpoints expect single-channel audio. Stereo is averaged automatically.
- Recommended max utterance length is 30 s for Wav2Vec/Whisper paths.
