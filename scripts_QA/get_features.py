import os
import json
import random

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor

__all__ = [
    "QwenVLProbeDataset",
    "make_collate_fn",
    "load_samples",
    "build_label_mappings",
    "build_dataloaders",
    "QwenVLFeatureExtractor",
]

# ===================== 配置区 =====================
# 现在不再用文本 prompt 了，只做纯图像 probing
MAX_PIXELS = 313600  # 控制单图像像素上限，防炸
NUM_WORKERS = 4      # DataLoader 里用的 num_workers

LABEL_CONFIG = {
    "title": "work_title",        # 逻辑名 -> json 字段名
    "author": "author_name_cn",   # 逻辑名 -> json 字段名
    "work_schools": "work_schools",
    "work_techniques": "work_techniques",
    "work_composition": "work_composition",
    "artistry_style": "artistry_style",
}


# ===================== 图像设置 =====================
def image_resize(image):
    max_side = 560
    w, h = image.size
    scale = max_side / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.Resampling.BILINEAR)


# ===================== Qwen Data Set（纯图像版） =====================

class QwenVLProbeDataset(Dataset):
    """
    现在 Dataset 只返回：
      image, labels_dict

    labels_dict: { field_name: label_id, ... }
    """
    def __init__(self, samples, field2label2id, max_pixels=MAX_PIXELS):
        self.samples = samples
        self.field2label2id = field2label2id  # dict[field_name] = {label -> id}
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 1. 读图 & 防 DecompressionBomb
        image = Image.open(s["image_path"]).convert("RGB")
        if image.width * image.height > self.max_pixels:
            scale = (self.max_pixels / (image.width * image.height)) ** 0.5
            new_w = int(image.width * scale)
            new_h = int(image.height * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # 2. 动态构造 labels_dict（不再有 system_prompt / question）
        labels_dict = {}
        for field, label2id in self.field2label2id.items():
            if field not in s:
                # 这一条样本没有这个字段就跳过（训练时会用 ignore_index=-100 忽略）
                continue
            raw_label = s[field]
            labels_dict[field] = label2id[raw_label]

        # 只返回 image + labels_dict
        return image, labels_dict


#  ===================== collate_fn：只用图像打包成模型输入 =====================
def make_collate_fn(processor):
    def collate_fn(batch):
        
        images, labels_list = zip(*batch)
        image_token = getattr(processor, "image_token", "<image>")
        dummy_texts = [image_token]*len(images) # qwen2.5 内部假设有text
        enc = processor(
            images=list(images),
            text = dummy_texts,
            padding=True,
            return_tensors="pt",
        )

        # 把每个 field 的 label 打成一个 batch 的 tensor 
        batch_labels = {}

        # 收集这一 batch 里所有出现过的字段名
        all_fields = set()
        for ld in labels_list:
            all_fields.update(ld.keys())

        for field in all_fields:
            field_values = []
            for ld in labels_list:
                if field in ld:
                    field_values.append(ld[field])
                else:
                    # 这一条样本没有这个字段，用 -100 占位（loss 用 ignore_index=-100 忽略）
                    field_values.append(-100)
            batch_labels[field] = torch.tensor(field_values, dtype=torch.long)

        # 返回 enc（只含 pixel_values 等） 和 batch_labels(dict)
        return enc, batch_labels

    return collate_fn


#  ===================== jsonl 导入 + 构造 samples 列表 =====================

def load_samples(file_path, base_image_dir, label_config=LABEL_CONFIG):
    """
    从 jsonl 读入数据，构造一个 list[sample_dict]：
      每个 sample 至少包含：
        "image_path": ...,

      然后根据 label_config 动态添加字段，比如：
        "title":   work_title,
        "author":  author_name_cn,
        "work_schools": ...,
        ...
    """
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            filename = item["filename"]
            image_path = os.path.join(base_image_dir, filename)

            s = {
                "image_path": image_path,
            }

            # labels reading
            for field_name, json_key in label_config.items():
                val = item.get(json_key, None)
                if isinstance(val, str):
                    val = val.strip()
                if val is None or val == "":
                    continue
                s[field_name] = val

            samples.append(s)

    print("sample example:", samples[0])
    print("Total samples:", len(samples))
    if len(samples) == 0:
        raise RuntimeError("No samples loaded, please check file_path / jsonl format.")
    return samples


# ===================== 从 samples 构建 label 映射 =====================
def build_label_mappings(samples, target_fields=None):
    random.seed(42)
    random.shuffle(samples)

    # 自动推断所有候选字段
    if target_fields is None:
        all_keys = set(samples[0].keys())
        # 只排除 image_path，其余都当作潜在 label 字段
        all_keys -= {"image_path"}
        target_fields = sorted(list(all_keys))

    field2label2id = {}
    field2id2label = {}

    for field in target_fields:
        labels = sorted(list({
            s[field] for s in samples
            if field in s and s[field] is not None and s[field] != ""
        }))

        if not labels:
            print(f"[build_label_mappings] Field '{field}' has no valid labels, skip.")
            continue

        label2id = {l: i for i, l in enumerate(labels)}
        id2label = {i: l for l, i in label2id.items()}

        field2label2id[field] = label2id
        field2id2label[field] = id2label

        print(f"Num classes for field '{field}':", len(label2id))

    return field2label2id, field2id2label


# ===================== 构建 DataLoader =====================
def build_dataloaders(
    samples,
    field2label2id,
    processor,
    batch_size,
    max_pixels=MAX_PIXELS,
):
    """
    新接口（纯图像版）：
      samples: load_samples 返回的列表
      field2label2id: build_label_mappings 返回的第一个 dict
    """
    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_dataset = QwenVLProbeDataset(
        train_samples,
        field2label2id=field2label2id,
        max_pixels=max_pixels,
    )
    val_dataset = QwenVLProbeDataset(
        val_samples,
        field2label2id=field2label2id,
        max_pixels=max_pixels,
    )

    collate_fn = make_collate_fn(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, train_samples, val_samples


# ===================== 特征提取（多层 + mean pooling，图像为主） =====================
class QwenVLFeatureExtractor(nn.Module):
    """
    纯图像 probing 版：

    - 调用完整的 Qwen2.5-VL 模型，但只提供 pixel_values
      （AutoProcessor(images=..., return_tensors="pt") 生成的 enc）
    - 取最后 top_k_layers 的 hidden_states 做平均
    - 在 token 维度上做 mean pooling 得到 (B, D)

    说明：
      这里仍然用的是“模型输出的 hidden_states”，只是 input 只有图像，
      这样得到的 embedding 主要来自视觉 encoder，对 probing 更纯粹。
    """
    def __init__(self, qwen_model, top_k_layers=4):
        super().__init__()
        self.qwen = qwen_model
        self.top_k_layers = top_k_layers

    def forward(self, inputs):
        """
        inputs: processor 的输出 dict（包含 pixel_values，可能还带别的 key）
        返回: (B, D) 特征向量
        """
        device = next(self.qwen.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if "input_ids" in inputs and inputs["input_ids"].dtype != torch.long:
            inputs["input_ids"] = inputs["input_ids"].long()
        outputs = self.qwen(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

        # hidden_states: tuple(num_layers+1, B, T, D)，第0个是 embedding 输出
        hidden_states = outputs.hidden_states

        # 取最后 top_k_layers 层（不含 embedding 层）
        hs = hidden_states[1:]  # 去掉 embedding 层
        num_hidden_layers = len(hs)
        k = min(self.top_k_layers, num_hidden_layers)
        last_k = hs[-k:]  # list of (B, T, D)

        # 堆叠后在“层”维度做平均 -> (B, T, D)
        stacked = torch.stack(last_k, dim=0)  # (k, B, T, D)
        h = stacked.mean(dim=0)               # (B, T, D)

        # 没有 text 的情况下，一般不会有 attention_mask；
        # 如果有，就用它；没有就全 1。
        attn = inputs.get(
            "attention_mask",
            torch.ones(h.size()[:2], device=h.device)
        )  # (B, T)
        attn = attn.unsqueeze(-1)             # (B, T, 1)

        attn_sum = attn.sum(dim=1).clamp(min=1.0)  # (B, 1)
        features = (h * attn).sum(dim=1) / attn_sum  # (B, D)

        return features.float()
