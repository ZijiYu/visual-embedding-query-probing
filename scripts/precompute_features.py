"""
预计算 Qwen-VL 的图像特征，用于后续只在特征上训练 probe（image-only probing）。
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoProcessor

from logs.setup_logger import setup_logger
from get_features import (
    load_samples,
    build_label_mappings,
    build_dataloaders,
    QwenVLFeatureExtractor,
)

# ===================== 配置区 =====================

# 和 train_probe.py 保持一致
# model_path = "/workspace/data/checkpoint-2077"
model_path = "/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct"
file_path = "/workspace/probing/Probing/data/full_data.jsonl"
base_image_dir = "/workspace/"

batch_size = 4
max_pixels = 313600

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 想预计算/训练哪些字段（需要和 get_features.LABEL_CONFIG 里 key 对得上）
TARGET_FIELDS = ["title", "author"]
# 例如：["title", "author", "work_schools", "work_techniques", "artistry_style"]


logger = setup_logger()


# ===================== 工具函数：加载大模型 & processor =====================

def build_model_and_processor(model_path, device):
    logger.info("Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        # 可以按需打开半精度：
        # torch_dtype=torch.float16,
        device_map=None,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 冻结大模型参数（只做特征提取，不训练）
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size
    logger.info(f"Hidden size: {hidden_size}")
    logger.info(f"Model param device: {next(model.parameters()).device}")

    return tokenizer, model, processor, hidden_size


# ===================== 主逻辑：预计算特征（image-only） =====================

def main():
    global logger

    # 1. 加载大模型 & processor
    tokenizer, model, processor, hidden_size = build_model_and_processor(
        model_path=model_path,
        device=device,
    )

    # 纯图像版特征提取器：get_features 里已经改成不需要 text
    feature_extractor = QwenVLFeatureExtractor(model, top_k_layers=1).eval()

    # 2. 加载样本（只含 image_path + 各字段标签）
    logger.info("Loading samples from jsonl...")
    samples = load_samples(
        file_path=file_path,
        base_image_dir=base_image_dir,
    )
    logger.info("Total samples: %d", len(samples))

    # 3. 构建 label 映射（支持多字段）
    field2label2id, field2id2label = build_label_mappings(
        samples,
        target_fields=TARGET_FIELDS,
    )
    active_fields = list(field2label2id.keys())
    logger.info("Active fields: %s", active_fields)
    if not active_fields:
        raise RuntimeError("No active fields for probing, please check data and TARGET_FIELDS.")

    # 4. 构建 DataLoader（这里主要是为了遍历所有样本拿 enc/labels）
    train_loader, val_loader, train_samples, val_samples = build_dataloaders(
        samples=samples,
        field2label2id=field2label2id,
        processor=processor,
        batch_size=batch_size,
        max_pixels=max_pixels,
    )
    logger.info("Train size: %d, Val size: %d", len(train_samples), len(val_samples))

    # 为了简单：我们直接把 train_loader 和 val_loader 串起来，
    # 把它们视作一个整体的数据集来预计算特征。
    all_loaders = [("train", train_loader), ("val", val_loader)]

    all_features = []
    all_labels = {field: [] for field in active_fields}
    all_split_flags = []  # 记录每个样本是 train 还是 val，方便之后拆分

    with torch.no_grad():
        for split_name, loader in all_loaders:
            for enc, batch_labels in tqdm(loader, desc=f"Precompute {split_name}"):
                # enc: processor 的输出 dict（只包含 pixel_values 等，不再有 text）
                enc = {k: v.to(device) for k, v in enc.items()}

                # 1) 抽特征（image-only embedding）
                feats = feature_extractor(enc)  # (B, D)
                # 可选：L2 normalize 一下
                feats = F.normalize(feats, dim=-1)

                all_features.append(feats.cpu())

                # 2) 把每个 field 的 label 一起存下来
                for field in active_fields:
                    if field in batch_labels:
                        all_labels[field].append(batch_labels[field].clone())
                    else:
                        # 理论上不会发生，因为 build_dataloaders 已按 field2label2id 构造了
                        raise RuntimeError(f"Field {field} not found in batch_labels.")

                # 记录 split 信息
                all_split_flags.extend([split_name] * feats.size(0))

    # 拼接所有 batch 的特征和标签
    features = torch.cat(all_features, dim=0)  # (N, D)
    labels = {field: torch.cat(lst, dim=0) for field, lst in all_labels.items()}

    logger.info("Final features shape: %s", str(features.shape))
    for field, y in labels.items():
        logger.info("Labels[%s] shape: %s", field, str(y.shape))

    # 5. 保存到 .pt 文件
    save_path = "/workspace/probing/Probing/src/qwen_precomputed_feature.pt"
    torch.save(
        {
            "features": features,            # (N, D)
            "labels": labels,               # dict[field] -> (N,)
            "field2label2id": field2label2id,
            "field2id2label": field2id2label,
            "hidden_size": hidden_size,
            "active_fields": active_fields,
            "split_flags": all_split_flags,  # 每个样本对应 "train" 或 "val"
        },
        save_path,
    )
    logger.info("Saved precomputed features to: %s", save_path)


if __name__ == "__main__":
    main()
