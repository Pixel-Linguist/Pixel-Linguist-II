#!/bin/bash
# Populates PixelLinguistII/ with the training + evaluation code and exports the
# two released checkpoints in a HuggingFace-ready layout.
#
# Only copies. Training/ and Evaluation/ are never modified.
#
#   bash PixelLinguistII/setup_release.sh
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TRAIN_SRC="$REPO/Training"
EVAL_SRC="$REPO/Evaluation/MTEB-MIEB"

missing=0
copy() {  # copy <src> <dst>
    if [ -e "$1" ]; then
        mkdir -p "$(dirname "$2")"
        cp -r "$1" "$2"
        echo "  ok   $(basename "$2")"
    else
        echo "  MISS $1"
        missing=$((missing + 1))
    fi
}

echo "==> training code"
copy "$TRAIN_SRC/train_laion_inbatch_multilingual.py" "$HERE/training/train_laion_inbatch_multilingual.py"
copy "$TRAIN_SRC/train_finetuning.py"                 "$HERE/training/train_finetuning.py"

echo "==> data preparation"
for f in extract_ViT.py extract_dtd.py filter_dataset.py wikispan_select.py process_data.py; do
    copy "$TRAIN_SRC/$f" "$HERE/training/$f"
done

echo "==> evaluation code"
copy "$EVAL_SRC/run-mieb-lite.py"       "$HERE/evaluation/run-mieb-lite.py"
copy "$EVAL_SRC/mteb/models/a_pixel.py" "$HERE/evaluation/pixel_linguist2.py"

# Only the two released checkpoints' scores are shipped; the other four models
# evaluated in the paper have no public weights.
_RES="$EVAL_SRC/mieb-results/paper_results"
copy "$_RES/collect_results.py"     "$HERE/evaluation/paper_results/collect_results.py"
copy "$_RES/collect_avg_results.py" "$HERE/evaluation/paper_results/collect_avg_results.py"
copy "$_RES/pretrain2"      "$HERE/evaluation/paper_results/pixel-linguist-2-pretrain"
copy "$_RES/midtrain-2400"  "$HERE/evaluation/paper_results/pixel-linguist-2-midtrain"

echo "==> checkpoint export"
# Only the weights + configs are needed for release. The optimizer shards under
# global_step*/, the 64 rng_state_*.pth files and trainer_state.json are dropped,
# taking each checkpoint from ~20GB to ~1.3GB.
export_ckpt() {  # export_ckpt <src_dir> <release_name>
    local src="$1" dst="$HERE/ckpt_export/$2"
    if [ ! -f "$src/model.safetensors" ]; then
        echo "  MISS $src/model.safetensors"
        missing=$((missing + 1))
        return
    fi
    mkdir -p "$dst"
    cp "$src/model.safetensors" "$dst/"
    cp "$src/config.json"       "$dst/"
    # pretrain-wiki-5314 has no preprocessor_config.json; fall back to the base ViT
    if [ -f "$src/preprocessor_config.json" ]; then
        cp "$src/preprocessor_config.json" "$dst/"
    elif [ -f "$TRAIN_SRC/Qwen2.5-VL-ViT-Only/preprocessor_config.json" ]; then
        cp "$TRAIN_SRC/Qwen2.5-VL-ViT-Only/preprocessor_config.json" "$dst/"
        echo "  note preprocessor_config.json taken from Qwen2.5-VL-ViT-Only"
    else
        echo "  WARN no preprocessor_config.json found for $2"
    fi
    echo "  ok   $2 ($(du -sh "$dst" | cut -f1))"
}

export_ckpt "$TRAIN_SRC/checkpoints/pretrain-wiki-5314" "pixel-linguist-2-pretrain"
export_ckpt "$TRAIN_SRC/checkpoints/midtrain-2400"      "pixel-linguist-2-midtrain"

echo
if [ "$missing" -eq 0 ]; then
    echo "Done. No missing sources."
else
    echo "Done, but $missing source(s) were missing (see MISS lines above)."
fi
echo "Review $HERE for hardcoded absolute paths before publishing."
