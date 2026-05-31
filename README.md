<div align="center">

# Nepali ASR Benchmark

### A Controlled Multi-Model Comparison of Multilingual Pre-Trained ASR for Nepali

[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/nepali_asr_benchmark.pdf)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow.svg)](https://huggingface.co/sumanpaudel1997)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow.svg)](https://huggingface.co/datasets/sumanpaudel1997/nepali-asr-benchmark)
[![License](https://img.shields.io/badge/license-Research-blue.svg)](#license)

</div>

---

## Highlights

- First standardized **multi-model, multi-dataset benchmark** for Nepali Automatic Speech Recognition.
- Six pretrained models fine-tuned on **OpenSLR SLR54** (~165 hrs of Nepali read speech) under one identical protocol.
- Three architectural families covered: **CTC self-supervised** (Wav2Vec 2.0), **autoregressive encoder–decoder** (Whisper), **hybrid Conformer-CTC**.
- Evaluated on three test sets — **OpenSLR**, **FLEURS**, **Common Voice** — across **WER**, **CER**, and **Real-Time Factor (RTF)**.
- Top tier of three models within **1 pp WER** despite a **9× parameter gap**: language-family **proximity beats raw scale** for in-domain Nepali.
- **CTC decoders are ~29× faster** than Whisper at equivalent accuracy → decisive for any deployment with a latency budget.

## Released Models

All six fine-tuned checkpoints are released on Hugging Face (currently private; will be opened with the public announcement).

| Model | Architecture Family | Decoder | Params | Best Val WER | HF Hub |
|---|---|---|---|---|---|
| **Whisper-Large-v3-Turbo** | Encoder–Decoder | Autoreg. | 809 M | **14.27 %** | [`sumanpaudel1997/nepali-asr-whisper-turbo`](https://huggingface.co/sumanpaudel1997/nepali-asr-whisper-turbo) |
| **IndicWav2Vec** | CTC SSL (Wav2Vec 2.0) | CTC | 94.4 M | **15.08 %** | [`sumanpaudel1997/nepali-asr-indicwav2vec`](https://huggingface.co/sumanpaudel1997/nepali-asr-indicwav2vec) |
| Whisper-Medium | Encoder–Decoder | Autoreg. | 769 M | 15.12 % | [`sumanpaudel1997/nepali-asr-whisper-medium`](https://huggingface.co/sumanpaudel1997/nepali-asr-whisper-medium) |
| Conformer-Hi | Hybrid Conformer-CTC | CTC | 30.5 M | 24.68 % | [`sumanpaudel1997/nepali-asr-conformer-hi`](https://huggingface.co/sumanpaudel1997/nepali-asr-conformer-hi) |
| XLSR-53 | CTC SSL (Wav2Vec 2.0) | CTC | 317 M | 26.73 % | [`sumanpaudel1997/nepali-asr-xlsr-53`](https://huggingface.co/sumanpaudel1997/nepali-asr-xlsr-53) |
| MMS-1B | CTC SSL (Wav2Vec 2.0) | CTC | 965 M | 26.99 % | [`sumanpaudel1997/nepali-asr-mms-1b`](https://huggingface.co/sumanpaudel1997/nepali-asr-mms-1b) |

## Results

### Benchmark WER (%) — fine-tuned, three test sets

| Model | OpenSLR | FLEURS | Common Voice |
|---|:-:|:-:|:-:|
| Whisper-Turbo | **14.76** | 39.56 | **48.35** |
| IndicWav2Vec | 14.89 | 40.68 | 51.65 |
| Whisper-Medium | 15.57 | **39.06** | 48.98 |
| Conformer-Hi | 26.28 | 41.05 | 58.49 |
| XLSR-53 | 26.85 | 57.78 | 62.78 |
| MMS-1B | 27.28 | 39.83 | 58.65 |

### Inference Real-Time Factor (lower = faster)

| Model | OpenSLR | FLEURS | Common Voice |
|---|:-:|:-:|:-:|
| **Conformer-Hi** | **0.0020** | **0.0019** | **0.0017** |
| IndicWav2Vec | 0.0025 | 0.0030 | 0.0024 |
| XLSR-53 | 0.0080 | 0.0088 | 0.0074 |
| MMS-1B | 0.0214 | 0.0230 | 0.0197 |
| Whisper-Medium | 0.0850 | 0.0826 | 0.0890 |
| Whisper-Turbo | 0.0979 | 0.0460 | 0.0832 |

Measured on a single NVIDIA L4 GPU, batch size 1 (single-utterance real-time inference).

### Deployment recommendations

| Constraint | Recommended model |
|---|---|
| **Real-time / edge** (latency-critical) | **IndicWav2Vec** — ~400× real-time, 94 M params, 14.89 % WER |
| **Cross-domain robustness on clean speech** | **Whisper-Turbo** — most consistent across the three test sets |
| **Out-of-domain generalization** | **MMS-1B** — smallest in-domain → FLEURS WER gap (+12.55 pp) |

## Quick Start

### Whisper family

```python
from transformers import WhisperForConditionalGeneration, AutoProcessor
import librosa, torch

mid = "sumanpaudel1997/nepali-asr-whisper-turbo"
proc = AutoProcessor.from_pretrained(mid)
model = WhisperForConditionalGeneration.from_pretrained(mid).eval()
audio, _ = librosa.load("clip.wav", sr=16000, mono=True)
forced = proc.get_decoder_prompt_ids(language="ne", task="transcribe")
feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
gen = model.generate(feats, forced_decoder_ids=forced, max_new_tokens=200)
print(proc.batch_decode(gen, skip_special_tokens=True)[0])
```

### Wav2Vec 2.0 family (IndicWav2Vec / XLSR-53 / MMS-1B)

```python
from transformers import Wav2Vec2ForCTC, AutoProcessor
import librosa, torch

mid = "sumanpaudel1997/nepali-asr-indicwav2vec"
proc = AutoProcessor.from_pretrained(mid)
model = Wav2Vec2ForCTC.from_pretrained(mid).eval()
audio, _ = librosa.load("clip.wav", sr=16000, mono=True)
inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(inputs.input_values).logits
print(proc.batch_decode(torch.argmax(logits, dim=-1))[0])
```

Ready-to-run scripts: see [`inference/`](inference/).

## Repository Layout

```
.
├── paper/                       # full paper PDF
│   └── nepali_asr_benchmark.pdf
├── results/                     # raw benchmark numbers
│   ├── benchmark_results.csv
│   └── training_loss.csv
├── inference/                   # minimal scripts to reproduce outputs from any clip
│   ├── transcribe_whisper.py
│   ├── transcribe_wav2vec.py
│   └── transcribe_conformer.py
└── README.md
```

Training code is intentionally not released in this repository.

## Benchmark Predictions Dataset

The per-utterance reference, hypothesis, WER, and CER for every (model × test set)
combination — 18 parquet files in total — is released as a Hugging Face dataset
so anyone can re-aggregate metrics, run error analysis, or compare new systems
without re-running inference:

[`sumanpaudel1997/nepali-asr-benchmark`](https://huggingface.co/datasets/sumanpaudel1997/nepali-asr-benchmark)

```python
# Load all 18 (model × test set) prediction parquets
from datasets import load_dataset
ds = load_dataset("sumanpaudel1997/nepali-asr-benchmark", split="predictions")

# Rebuild the WER table from the paper in two lines of pandas
import pandas as pd
df = ds.to_pandas()
print(
    df.groupby(["model", "test_set"])["wer"].mean().mul(100).round(2)
      .unstack("test_set")[["openslr", "fleurs", "common_voice"]]
      .sort_values("openslr")
)
```

## Source Datasets (audio not redistributed; obtain from source)

| Dataset | Hours | Link | Used as |
|---|---|---|---|
| OpenSLR SLR54 (Nepali) | ~165 | <https://www.openslr.org/54/> | training + in-domain test |
| FLEURS (ne_np) | ~10 | <https://huggingface.co/datasets/google/fleurs> | curated OOD test |
| Common Voice (ne-NP) | ~5 | <https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk> | noisy OOD test |

Per-dataset citations:

```bibtex
@inproceedings{kjartansson2018crowdsourced,
  title     = {Crowd-Sourced Speech Corpora for {Javanese}, {Sundanese}, {Sinhala}, {Nepali}, and {Bangladeshi Bengali}},
  author    = {Kjartansson, Oddur and Sarin, Supheakmungkol and Pipatsrisawat, Knot and Jansche, Martin and Ha, Linne},
  booktitle = {Proceedings of the 6th Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU)},
  pages     = {52--55},
  year      = {2018}
}
@inproceedings{conneau2023fleurs,
  title     = {{FLEURS}: {Few}-shot learning evaluation of universal representations of speech},
  author    = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  booktitle = {IEEE SLT 2022},
  pages     = {798--805},
  year      = {2023}
}
@inproceedings{ardila2020common,
  title     = {Common Voice: {A} massively-multilingual speech corpus},
  author    = {Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M. and Weber, Gregor},
  booktitle = {LREC 2020},
  pages     = {4218--4222},
  year      = {2020}
}
```

All evaluation uses NFC normalisation on both references and hypotheses; WER and CER are computed with [`jiwer`](https://github.com/jitsi/jiwer).

## Citation

```bibtex
@article{paudel2026nepaliasr,
  title         = {Comparative Analysis of Multilingual Pre-trained Models for Nepali Automatic Speech Recognition},
  author        = {Paudel, Suman and Sayami, Sarbin},
  year          = {2026},
  month         = may,
  eprint        = {submit/7658766},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

> arXiv:submit/7658766 [cs.AI], 31 May 2026.

## License

Model checkpoints are released for **research use only**; downstream commercial use requires
permission from the authors. Inherited licences from the source checkpoints (Wav2Vec 2.0, Whisper,
MMS, IndicWav2Vec, NeMo Conformer-Hi) apply on top of this restriction.

## Acknowledgements

This work was carried out as a M.Sc. (Data Science) thesis at the **School of Mathematical Sciences,
Institute of Science and Technology, Tribhuvan University**, Kathmandu, Nepal, under the supervision
of **Asst. Prof. Sarbin Sayami** (Central Department of Computer Science and Information Technology).
We thank the open-source communities behind Wav2Vec 2.0, Whisper, MMS, IndicWav2Vec, and NVIDIA NeMo,
whose publicly released checkpoints made this controlled comparison possible.

## Contact

Suman Paudel — `dastonsuman1997@gmail.com`
