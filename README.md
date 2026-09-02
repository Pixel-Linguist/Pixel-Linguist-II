# PIXEL LINGUIST II

Code and checkpoints for *On the Design Fundamentals of Pixel Text Representation
Learning* (EMNLP).

PIXEL LINGUIST II is a vision-only encoder that reads, retrieves and compresses
language directly in pixel space. It is a native-resolution ViT (initialised from
the Qwen2.5-VL vision tower) trained with on-the-fly text rendering, unified
contrastive grounding over natural image-text and rendered text-text pairs, and a
two-stage multilingual curriculum totalling 280M examples seen.

## Models

| Model | Stage |
|---|---|
| [`Pixel-Linguist/Pixel-Linguist-II-Pretrain`](https://huggingface.co/Pixel-Linguist/Pixel-Linguist-II-Pretrain) | End of Stage 1 (foundational pretraining) |
| [`Pixel-Linguist/Pixel-Linguist-II-Midtrain`](https://huggingface.co/Pixel-Linguist/Pixel-Linguist-II-Midtrain) | Stage 1 + Stage 2 — the full curriculum |
| [`Pixel-Linguist/Pixel-Linguist-II-Midtrain-Only`](https://huggingface.co/Pixel-Linguist/Pixel-Linguist-II-Midtrain-Only) | Stage 2 only — the paper's headline English STS and ViDoRe model |

The three models are collected in the private
[PIXEL LINGUIST II collection](https://huggingface.co/collections/Pixel-Linguist/pixel-linguist-ii-6a9873c0612d5bc1d82f8b35).
Each repository carries a usage snippet; the short version is:

```python
import json, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

path = snapshot_download("Pixel-Linguist/Pixel-Linguist-II-Midtrain")
config = Qwen2_5_VLVisionConfig(**json.load(open(f"{path}/config.json")))
model = Qwen2_5_VisionTransformerPretrainedModel(config)
model.load_state_dict(load_file(f"{path}/model.safetensors"), strict=True)
model = model.to("cuda", torch.bfloat16).eval()
```

To embed a string, render it to an image first — that is the point of the model.
`evaluation/pixel_linguist2.py` contains the renderer and the pooling used for
all reported numbers.

## Layout

```
PixelLinguistII/
├── setup_release.sh                       # copies code out of Training//Evaluation/ and exports checkpoints
├── requirements.txt
├── training/
│   ├── train_laion_inbatch_multilingual.py  # Stage 1 + Stage 2 entrypoint
│   ├── train_laion_text_wiki.sh             # Stage 1 launcher
│   ├── train_laion_text.sh                  # Stage 2 launcher
│   ├── train_finetuning.py                  # Stage 3 (AllNLI triplet finetuning)
│   ├── train_finetuning_final.sh            # Stage 3 launcher
│   ├── extract_ViT.py                       # slice the vision tower out of Qwen2.5-VL
│   ├── extract_dtd.py                       # unpack DTD textured backgrounds
│   ├── download_font_lib.py                 # rebuild the 393-font library
│   ├── font_manifest.json                   # exact family/subset/variant of every font
│   ├── filter_dataset.py                    # language + script filtering (Text Corpus 2)
│   ├── wikispan_select.py                   # Wikipedia span preparation (Text Corpus 1)
│   ├── process_data.py                      # AllNLI CSV -> triplet JSON
│   └── deepspeed_config/                    # ZeRO-2 configs
├── evaluation/
│   ├── README.md                            # how to install MIEB, register and run
│   ├── run-mieb-lite.py                     # evaluates the 17 tasks used in the paper
│   ├── pixel_linguist2.py                   # MTEB wrapper + ModelMeta registrations
│   └── paper_results/                       # score JSONs + collect_results.py
└── ckpt_export/                             # HF-ready checkpoints (created by setup_release.sh, not tracked)
```

Run `bash setup_release.sh` once to populate `training/`, `evaluation/` and
`ckpt_export/`. It only reads from `Training/` and `Evaluation/`.

## Training pipeline

All stages use a single self-contained script, `training/train_laion_inbatch_multilingual.py`,
which embeds the font library, the on-the-fly text-to-image renderer, both dataset
classes, the collator and the contrastive trainer. Text is rendered fresh every
epoch, so the model never sees the same visual instantiation of a string twice.

| Stage | Text data | Image data | Init from | LR | Batch | Epochs |
|---|---|---|---|---|---|---|
| 1 — Foundational pretraining | Text Corpus 1: 62M multilingual Wikipedia docs, cropped twice into unsupervised pairs | 26M LAION-2B image-text pairs | Qwen2.5-VL ViT | 5e-5 | 1024 | 2 |
| 2 — Semantic mid-training | Text Corpus 2: 26M curated multilingual semantic pairs | same 26M LAION-2B pairs | Stage 1 | 5e-5 | 1024 | 2 |
| 3 — Finetuning (optional) | ~270K AllNLI triplets | — | Stage 2 | 5e-6 | 768 | 2 |

### Corpora and examples seen

Both stages run the same script and differ only in the text corpus. In the
original repository they correspond to `train_laion_text_wiki.sh` (Stage 1) and
`train_laion_text.sh` (Stage 2); the latter passed no `--unsupervised_data_path`
and therefore used `train()`'s default.

| Corpus | Directory | Examples (as loaded) | Columns |
|---|---|---|---|
| Text Corpus 1 | `data/wikispan-filtered` | 60,767,396 | `sentence1`, `sentence2` |
| Text Corpus 2 | `data/multilingual-filtered2` | 26,009,280 | `sentence1`, `sentence2` |
| Image-text | LAION-2B subset | ~26M | — |

Counts are what `load_from_disk` actually yields, which is what the training
script consumes. They differ slightly from the `num_examples` recorded in
`dataset_info.json` (61,614,907 and 26,083,582) because some of these datasets
carry `cache-*.arrow` shards from a filtering pass. The paper rounds them to 62M
and 26M.

```
Stage 1: (62M wiki  + 26M LAION) x 2 epochs = 176M
Stage 2: (26M pairs + 26M LAION) x 2 epochs = 104M
                                      total = 280M examples seen
```

Loss is a symmetric in-batch InfoNCE with embeddings all-gathered across ranks;
NaViT-style variable-resolution pooling averages the `(H/2)x(W/2)` merged patches
per image. bf16 + DeepSpeed ZeRO-2 + gradient checkpointing throughout.

### Prerequisites

1. **Extract the Qwen2.5-VL vision tower** — the training scripts train the ViT
   slice alone:
   ```bash
   python training/extract_ViT.py \
       --model_path Qwen/Qwen2.5-VL-7B-Instruct \
       --output_path ./Qwen2.5-VL-ViT-Only
   ```
2. **Rebuild the font library** — 393 fonts across 96 families. The fonts are
   *not* redistributed here; the script refetches them from Google Fonts via
   google-webfonts-helper, reproducing the original filenames byte-for-byte:
   ```bash
   python training/download_font_lib.py --out ./font_lib
   ```
   All of them are freely licensed: 378 under SIL OFL 1.1, 2 under Apache-2.0,
   and 13 under either the Ubuntu Font Licence 1.0 or OFL. They are fetched
   rather than vendored to keep the repository small and to leave the font
   licences between you and their upstream authors. `training/font_manifest.json`
   records the exact family, subset and variant of every file.
3. **Extract the DTD texture backgrounds** used for rendering augmentation:
   ```bash
   python training/extract_dtd.py
   ```

### Running

The stage scripts take every path from an environment variable, so nothing is
hardcoded:

```bash
cd training
export BASE_MODEL=/path/to/Qwen2.5-VL-ViT-Only
export DATA_PATH=/path/to/laion-image-text
export UNSUP_DATA_PATH=/path/to/wikipedia-spans      # Text Corpus 1
export DTD_DATA_PATH=/path/to/dtd/extracted_images
export FONTS_DIR=/path/to/font_lib
bash train_laion_text_wiki.sh

export BASE_MODEL=checkpoints/pretrain
export UNSUP_DATA_PATH=/path/to/curated-text-pairs   # Text Corpus 2
bash train_laion_text.sh
```

## Released checkpoints

| Release | Stage | Training steps |
|---|---|---|
| [`Pixel-Linguist-II-Pretrain`](https://huggingface.co/Pixel-Linguist/Pixel-Linguist-II-Pretrain) | End of Stage 1 | 5314 (2 epochs) |
| [`Pixel-Linguist-II-Midtrain`](https://huggingface.co/Pixel-Linguist/Pixel-Linguist-II-Midtrain) | Stage 1 + Stage 2 — the model behind the paper's "pre-training + mid-training" results | 2400 |
| [`Pixel-Linguist-II-Midtrain-Only`](https://huggingface.co/Pixel-Linguist/Pixel-Linguist-II-Midtrain-Only) | Stage 2 only — the paper's "mid-training only" results | 3192 (2 epochs) |

Each repository contains only `model.safetensors`, `config.json` and
`preprocessor_config.json` plus its model card — optimizer shards, RNG states
and trainer state are dropped, so a release is ~1.3GB rather than ~20GB.

The full Stage 1 + Stage 2 run continued to step 3192, but the numbers reported
in the paper come from its step-2400 save, so that is the checkpoint released as
`Pixel-Linguist-II-Midtrain`.

## Reproducing the paper tables

See `evaluation/README.md` for the full evaluation flow. In short:

```bash
cd evaluation
export PIXEL_FONTS_DIR=/path/to/font_lib
python run-mieb-lite.py                 # evaluates the three released checkpoints
python paper_results/collect_results.py # tabulate scores
```

`evaluation/paper_results/` bundles the score JSONs for two checkpoints:

| Directory | Paper row |
|---|---|
| `pixel-linguist-2-pretrain` | Stage 1 only |
| `pixel-linguist-2-midtrain` | PL II (pre-training + mid-training) |

The third released checkpoint,
[`Pixel-Linguist-II-Midtrain-Only`](https://huggingface.co/Pixel-Linguist/Pixel-Linguist-II-Midtrain-Only),
provides the paper's "mid-training only" model; point `BASE_MODEL` in
`train_laion_text.sh` at the raw ViT to reproduce its training. The AllNLI
finetuned variants can be reproduced with `train_finetuning_final.sh`. Their
score JSONs, the raw Qwen2.5-ViT backbone results, and the controlled ablations
of Section 2 are not bundled in this three-model release.

## Citation

```bibtex
@inproceedings{yuan2026pixel,
  title     = {On the Design Fundamentals of Pixel Text Representation Learning},
  author    = {Yuan, Chaohao and Yuan, Ruifeng and Huang, Zhuoxu and Rong, Yu and
               Cheng, Hong and Chan, Hou Pong and Xiao, Chenghao},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```
