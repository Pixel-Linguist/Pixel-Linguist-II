# Evaluation

Reproduces the Visual STS and ViDoRe numbers using the MIEB-lite benchmark. Both
released checkpoints are evaluated as a single vision tower: text queries are
rendered to images and encoded through the same encoder as the document images.

## Install

MIEB tasks (`*VisualSTS`, `Vidore*`) live in the MIEB branch of MTEB, not in
upstream `pip install mteb`:

```bash
git clone https://github.com/embeddings-benchmark/mteb.git
cd mteb && git checkout mieb && pip install -e .
```

## Run

```bash
export PIXEL_FONTS_DIR=/path/to/font_lib     # renderer font library
python run-mieb-lite.py
```

That is all. `run-mieb-lite.py` imports the wrapper directly and attaches
`mteb_model_meta` itself, so there is no need to copy anything into
`mteb/models/` or edit `mteb/models/overview.py`.

Results are written as one JSON per task under `./mieb-results/<model>/<revision>/`.

## Files

- `pixel_linguist2.py` — `Qwen2_5_VL_ViT_Wrapper1` (the encoder, including the
  on-the-fly text renderer and NaViT variable-resolution pooling), the
  `ModelMeta` for each released checkpoint, and `get_model(name)`.
- `run-mieb-lite.py` — evaluates the 17 tasks reported in the paper.

## Options

Weights are pulled from the Hub by default —
[`ychaohao/pixel-linguist-2-pretrain`](https://huggingface.co/ychaohao/pixel-linguist-2-pretrain)
and
[`ychaohao/pixel-linguist-2-midtrain`](https://huggingface.co/ychaohao/pixel-linguist-2-midtrain).
To evaluate a local checkpoint instead:

```bash
export PIXEL_LINGUIST_2_PRETRAIN=/path/to/pretrain-checkpoint
export PIXEL_LINGUIST_2_MIDTRAIN=/path/to/midtrain-checkpoint
```

If the MIEB datasets are already mirrored locally, set `MIEB_DATA_ROOT` to that
directory and the datasets will be loaded from there instead of the Hub.

## Notes

- The 17 tasks are: 5 English Visual STS tasks, plus
  `STS17MultilingualVisualSTS` and `STSBenchmarkMultilingualVisualSTS` (whose
  non-English subsets give the cross-lingual and multilingual tables), plus the
  10 ViDoRe subsets.
- The paper's ViDoRe average uses 6 of the 10 subsets (DocVQA, InfoVQA,
  ShiftProject, SyntheticDocQA-AI, Tabfquad, Tatdqa).
- **Scores are not bit-reproducible.** The renderer samples a font at random from
  the candidate pool for the detected script, so each run rasterises the text
  slightly differently. Expect deviations of roughly 0.1 from the values in
  `paper_results/` — a rerun of `STS13VisualSTS` on `pixel-linguist-2-midtrain`
  gave 71.24 against the recorded 71.30.
- `transformers>=4.57` loads the image processor as `Qwen2VLImageProcessorFast`
  and warns that outputs may differ slightly from the slow processor, which also
  contributes to the drift.
