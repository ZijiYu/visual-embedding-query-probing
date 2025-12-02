# train_probe.py
# 在预计算好的 Qwen-VL 特征上训练多头线性 probe（title / author / ...）
import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from logs.setup_logger import setup_logger

# ===================== 配置区 =====================

# 预计算特征的路径（由 precompute_features.py 生成）
precomputed_path = "/workspace/probing/Probing/src/qwen_precomputed_feature.pt"

batch_size = 64        # 在特征上训练，可以开大点
epochs = 20
lr = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 如果你只想训 title/author，可以填：
# TARGET_FIELDS = ["title", "author"]
TARGET_FIELDS = None   # None 表示用 precompute 里所有 active_fields


logger = setup_logger()


# ===================== 多头线性 probe =====================
class MultiHeadProbe(nn.Module):
    """
    每个 field 一个 Linear classifier，用 ModuleDict 管理：
      heads[field_name]: Linear(D, num_labels_for_field)
    """
    def __init__(self, input_dim, field2num_labels):
        super().__init__()
        self.heads = nn.ModuleDict({
            field: nn.Linear(input_dim, n_labels)
            for field, n_labels in field2num_labels.items()
        })

    def forward(self, features):
        """
        features: (B, D)
        返回: logits_dict: { field: logits(B, C_field) }
        """
        features = F.normalize(features, dim=-1)
        logits_dict = {
            field: head(features)
            for field, head in self.heads.items()
        }
        return logits_dict


# ===================== 特征 Dataset（不再跑大模型） =====================
class FeatureDataset(torch.utils.data.Dataset):
    """
    用预计算好的 features + labels + split_flags 构造 Dataset
    split_name: "train" 或 "val"
    """
    def __init__(self, features, labels, split_flags, split_name, fields):
        super().__init__()
        self.features = features          # (N, D) tensor
        self.labels = labels              # dict[field] -> (N,) tensor
        self.split_flags = split_flags    # list[str], len N
        self.fields = fields              # list[field names]

        self.indices = [
            i for i, s in enumerate(split_flags) if s == split_name
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.features[real_idx]  # (D,)
        y = {f: self.labels[f][real_idx] for f in self.fields}
        return x, y


# ===================== 只在特征上评估（多头） =====================
@torch.no_grad()
def evaluate_multi_head_on_features(
    probe,
    data_loader,
    device,
):
    """
    返回:
      avg_loss: 验证集平均 loss（按 batch 平均）
      field_accs: dict[field] = accuracy
    """
    probe.eval()

    total_loss = 0.0
    total_batches = 0

    # 分字段累计 acc
    field_correct = {field: 0 for field in probe.heads.keys()}
    field_total = {field: 0 for field in probe.heads.keys()}

    for feats, batch_labels in data_loader:
        feats = feats.to(device)  # (B, D)

        # 多头 logits
        logits_dict = probe(feats)

        # 计算总 loss（所有字段 loss 求和）
        batch_loss = None
        for field, logits in logits_dict.items():
            if field not in batch_labels:
                continue
            labels = batch_labels[field].to(device)  # (B,)
            field_loss = F.cross_entropy(
                logits, labels, ignore_index=-100
            )

            if batch_loss is None:
                batch_loss = field_loss
            else:
                batch_loss = batch_loss + field_loss

            # 计算该字段的 acc（只统计 labels != -100 的样本）
            valid_mask = (labels != -100)
            valid_count = valid_mask.sum().item()
            if valid_count > 0:
                preds = logits.argmax(dim=-1)
                correct = (preds[valid_mask] == labels[valid_mask]).sum().item()
                field_correct[field] += correct
                field_total[field] += valid_count

        if batch_loss is None:
            continue
        total_loss += batch_loss.item()
        total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    field_accs = {
        field: (field_correct[field] / field_total[field] if field_total[field] > 0 else 0.0)
        for field in probe.heads.keys()
    }

    return avg_loss, field_accs


# ===================== 只在特征上训练 + 验证（多头） =====================
def train_and_eval_multi_head_on_features(
    epochs,
    train_loader,
    val_loader,
    probe,
    optimizer,
    device,
):

    logger.info("Start training (multi-head probe on precomputed features)...")

    for epoch in range(epochs):
        probe.train()

        total_loss = 0.0
        total_batches = 0

        train_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
        )

        for batch_idx, (feats, batch_labels) in enumerate(train_iter):
            feats = feats.to(device)  # (B, D)

            # 前向
            logits_dict = probe(feats)

            loss = None
            for field, logits in logits_dict.items():
                if field not in batch_labels:
                    continue
                labels = batch_labels[field].to(device)
                field_loss = F.cross_entropy(
                    logits, labels, ignore_index=-100
                )
                if loss is None:
                    loss = field_loss
                else:
                    loss = loss + field_loss

            if loss is None:
                continue

            # 反向传播 & 更新 probe 参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            if (batch_idx + 1) % 10 == 0:
                train_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                )

        train_loss = total_loss / total_batches if total_batches > 0 else 0.0
        val_loss, field_accs = evaluate_multi_head_on_features(
            probe,
            val_loader,
            device,
        )

        # 打印每个字段的 acc
        acc_str = " | ".join(
            [f"{field} Acc: {acc:.4f}" for field, acc in field_accs.items()]
        )
        logger.info(
            f"Epoch {epoch+1}/{epochs} "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"{acc_str}"
        )

    logger.info("Training finished.")
    return probe


def main():
    # ===== 1. 加载预计算特征 =====
    logger.info("Loading precomputed features from: %s", precomputed_path)
    data = torch.load(precomputed_path)

    features = data["features"]             # (N, D)
    labels = data["labels"]                 # dict[field] -> (N,)
    field2label2id = data["field2label2id"]
    field2id2label = data["field2id2label"]
    hidden_size = data["hidden_size"]
    active_fields = data["active_fields"]
    split_flags = data["split_flags"]       # list[str], len N

    logger.info("Loaded features: %s", str(features.shape))
    logger.info("Active fields from precompute: %s", active_fields)

    # 如果你在这里想只训某些字段，可以用 TARGET_FIELDS 过滤一次
    if TARGET_FIELDS is not None:
        fields = [f for f in active_fields if f in TARGET_FIELDS]
    else:
        fields = active_fields

    if not fields:
        raise RuntimeError("No fields to train on after applying TARGET_FIELDS filter.")

    logger.info("Fields used for training: %s", fields)

    # 准备 field->num_labels
    field2num_labels = {field: len(field2label2id[field]) for field in fields}

    # ===== 2. 构建 Dataset & DataLoader（只在 features 上） =====
    train_dataset = FeatureDataset(features, labels, split_flags, split_name="train", fields=fields)
    val_dataset   = FeatureDataset(features, labels, split_flags, split_name="val",   fields=fields)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    logger.info("Train size: %d, Val size: %d", len(train_dataset), len(val_dataset))

    # ===== 3. 构建多头 probe（线性） =====
    probe = MultiHeadProbe(
        input_dim=hidden_size,
        field2num_labels=field2num_labels,
    ).to(device)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=lr,
    )

    # ===== 4. 训练 + 验证（只在 features 上）=====
    probe = train_and_eval_multi_head_on_features(
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        probe=probe,
        optimizer=optimizer,
        device=device,
    )

    # ===== 5. 保存 probe 权重（多头） =====
    save_path = "/workspace/probing/Probing/src/multi_field_probe_on_features.pt"
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "field2label2id": field2label2id,
            "field2id2label": field2id2label,
            "active_fields": fields,
        },
        save_path,
    )
    logger.info("Saved multi-field probe (on features) to %s", save_path)


if __name__ == "__main__":
    main()
