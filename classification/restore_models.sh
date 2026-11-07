#!/usr/bin/env bash
# Put the pre-trained live/final classifier (LR + LightGBM ensemble, LIVE 96.38) where classification/ expects it,
# so FINAL only predicts (no 6-8 min retrain). Run once from the repo root after cloning:  classification/restore_models.sh
# Needs the same libraries the models were saved with: lightgbm 4.7.0, scikit-learn from requirements.txt (+ brew libomp).
set -euo pipefail
cd "$(dirname "$0")/.."
M=classification/pretrained; O=classification/out; Z=$O/zoo
mkdir -p "$Z"
cp "$M/model_v1_9999-12-31.npz" "$O/"
cp "$M/lgb_v1_9999-12-31.pkl" "$Z/"
gunzip -c "$M/hist_v1_9999-12-31.pkl.gz" > "$Z/hist_v1_9999-12-31.pkl"
gunzip -c "$M/stack_v1_9999-12-31_lgb.pkl.gz" > "$Z/stack_v1_9999-12-31_lgb.pkl"
echo "restored live/final classifier models into $O"
