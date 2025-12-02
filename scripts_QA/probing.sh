#!/usr/bin/env bash

# 简单一键跑完整 probing 流程：
# 1) 预计算特征
# 2) 在特征上训练 probe
# 3) 用训练好的 probe 跑 probing 导出 jsonl

set -e  # 任何一步报错就退出

# ===== 路径按你现在的项目来 =====
PROJECT_ROOT="/workspace/probing/Probing"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"

cd "${PROJECT_ROOT}"

echo ">>> [1/3] Precompute Qwen-VL features..."
python "${SCRIPTS_DIR}/precompute_features.py"

# echo ">>> [2/3] Train multi-field probe on precomputed features..."
# python "${SCRIPTS_DIR}/train_probe.py"

# echo ">>> [3/3] Run probing with trained probe..."
# python "${SCRIPTS_DIR}/run_probing.py"

# echo ">>> All done."
