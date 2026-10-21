#!/bin/bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "Lightning-Template" ]]; then
	printf '请先运行：conda activate Lightning-Template\n' >&2
	exit 1
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

trap 'printf "实验失败，已停止后续运行。\n" >&2' ERR

experiments=(
	conditional_flow_matching_vctk
	conditional_mean_flow_vctk
	meanvoiceflow_vctk
)

for experiment in "${experiments[@]}"; do
	printf '\n开始实验：%s（GPU %s）\n' "$experiment" "$CUDA_VISIBLE_DEVICES"
	python src/train.py \
		"experiment=$experiment" \
		trainer=gpu \
		logger=tensorboard \
		data.num_workers=8 \
		data.pin_memory=true
	printf '实验完成：%s\n' "$experiment"
done

printf '\nB1、B2、B3 全部完成。\n'
