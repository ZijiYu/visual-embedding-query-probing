import os
import tqdm
import json
import random

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, AutoProcessor


# ===================== 配置区 =====================
model_path = "/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct"
file_path = "/workspace/probing/Probing/data/shuffled.jsonl"
base_image_dir = "/workspace/"

QUESTION = "这幅画的名称和作者是谁？"

num_works = 4
batch_size = 8
epochs = 3              # 建议多训几轮看看趋势
lr = 1e-3
max_pixels = 80_000_000  # 控制单图像像素上限，防炸

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ===================== 加载模型 & processor =====================
print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    # torch_dtype=torch.float16,
    device_map=None,
).to(device).eval()

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

for p in model.parameters():
    p.requires_grad = False

hidden_size = model.config.hidden_size
print("Hidden size:", hidden_size)
print("Model param device:", next(model.parameters()).device)


# ===================== 构造样本 =====================
samples = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)

        filename = item["filename"]
        work_title = item["work_title"].strip()
        author_name_cn = item["author_name_cn"].strip()

        # label 字符串：作品名 + 作者，作为一个类别
        label_str = f"<作品名>:{work_title}<作者名>:{author_name_cn}"

        image_path = os.path.join(base_image_dir, filename)

        samples.append({
            "image_path": image_path,         # 图片路径
            "question": QUESTION,             # 纯问题，不含 <image> / {t}
            "work_title": work_title,
            "author_name_cn": author_name_cn,
            "label": label_str,               # 用来做 label2id
        })

print("Total samples:", len(samples))
if len(samples) == 0:
    raise RuntimeError("No samples loaded, please check file_path / jsonl format.")

# print("Example sample:", samples[0])


# ===================== 划分训练 / 验证 & label 映射 =====================
random.seed(42)
random.shuffle(samples)

split = int(0.8 * len(samples))
train_samples = samples[:split]
val_samples = samples[split:]

labels = sorted(list({s["label"] for s in samples}))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

print("Num classes:", len(label2id))


# ===================== Dataset & Collate =====================
class QwenVLProbeDataset(Dataset):
    def __init__(self, samples, label2id, max_pixels=80_000_000):
        self.samples = samples
        self.label2id = label2id
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

        # 2. 文本：纯问题
        text = s["question"]

        # 3. 标签 id
        label_id = self.label2id[s["label"]]

        return image, text, label_id


def make_collate_fn(processor):
    def collate_fn(batch):
        images, texts, labels = zip(*batch)

        # 官方的 image_token（如果没有则退回 "<image>"）
        image_token = getattr(processor, "image_token", "<image>")

        # 真正喂给模型的文本：只在这里加一次 image_token
        # 例如 "<image>\n这幅画的名称和作者是谁？"
        texts_for_model = [f"{image_token}\n{t}" for t in texts]

        enc = processor(
            images=list(images),
            text=texts_for_model,
            padding=True,
            return_tensors="pt",
        )

        labels = torch.tensor(labels, dtype=torch.long)
        return enc, labels

    return collate_fn


train_dataset = QwenVLProbeDataset(train_samples, label2id, max_pixels=max_pixels)
val_dataset   = QwenVLProbeDataset(val_samples, label2id, max_pixels=max_pixels)

collate_fn = make_collate_fn(processor)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers= num_works,
    pin_memory=True,
    persistent_workers=True

)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)


# ===================== 特征提取（多层 + mean pooling） =====================
class QwenVLFeatureExtractor(nn.Module):
    """
    从 Qwen2.5-VL Instruct 模型中抽特征：
    1. 取最后 top_k_layers 层 hidden state 做平均
    2. 在 token 维度上做带 attention_mask 的 mean pooling
    """
    def __init__(self, qwen_model, top_k_layers=4):
        super().__init__()
        self.qwen = qwen_model
        self.top_k_layers = top_k_layers

    def forward(self, inputs):
        # inputs: processor 的输出 dict（包含 input_ids, attention_mask, pixel_values 等）
        device = next(self.qwen.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.qwen(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

        # hidden_states: tuple of (num_layers+1, B, T, D)，第0个通常是 embedding
        hidden_states = outputs.hidden_states   # len = num_layers + 1

        # 取最后 top_k_layers 层（不含 embedding 层）
        hs = hidden_states[1:]  # 去掉 embedding 层
        num_hidden_layers = len(hs)

        k = min(self.top_k_layers, num_hidden_layers)
        print("top_k:",k )
        last_k = hs[-k:]  # list of (B, T, D)

        # 堆叠后在“层”维度做平均 -> (B, T, D)
        stacked = torch.stack(last_k, dim=0)  # (k, B, T, D)
        h = stacked.mean(dim=0)              # (B, T, D)

        # 带 attention_mask 的 mean pooling：更稳定
        attn = inputs.get("attention_mask", torch.ones(h.size()[:2], device=h.device))  # (B, T)
        attn = attn.unsqueeze(-1)  # (B, T, 1)

        # 防止全 0
        attn_sum = attn.sum(dim=1).clamp(min=1.0)  # (B, 1)

        features = (h * attn).sum(dim=1) / attn_sum  # (B, D)
        return features.float()

    
# ===================== 线性 probe =====================
class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, features, labels=None):
        # 可选：先 L2 normalize 一下，稳定一点
        features = F.normalize(features, dim=-1)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss


feature_extractor = QwenVLFeatureExtractor(model, top_k_layers=4)
probe = LinearProbe(hidden_size, num_labels=len(label2id)).to(device)

optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)


# ===================== 评估函数 =====================
@torch.no_grad()
def evaluate(probe, feature_extractor, data_loader, device):
    probe.eval()
    total_correct = 0
    total_count = 0
    total_loss = 0.0

    for enc, labels in data_loader:
        labels = labels.to(device)

        features = feature_extractor(enc)
        logits, loss = probe(features, labels)

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    avg_acc = total_correct / total_count if total_count > 0 else 0.0
    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    return avg_acc, avg_loss


# ===================== 训练主循环 =====================
train_losses = []
val_losses = []
val_accuracies = []

print("Start training...")
for epoch in range(epochs):
    probe.train()
    total_loss = 0.0
    total_count = 0

    train_iter = tqdm.tqdm(
        train_loader,
        desc = f"Epoch {epoch +1 }/{epochs}",
        leave = True
    )

    for batch_idx, (enc, labels) in enumerate(train_iter):
        labels = labels.to(device)

        # 冻结大模型，只在 probe 上反向
        with torch.no_grad():
            features = feature_extractor(enc)

        logits, loss = probe(features, labels)

        """
        debug
        """
        if torch.isnan(features).any():
            print("NaN in features!")
        if torch.isnan(logits).any():
            print("NaN in logits!")
        if torch.isnan(loss):
            print("NaN in loss!")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_count += bs

        if (batch_idx + 1) % 10 == 0:
            print(
                f"[Epoch {epoch+1}/{epochs}] "
                f"Batch {batch_idx+1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = total_loss / total_count if total_count > 0 else 0.0
    val_acc, val_loss = evaluate(probe, feature_extractor, val_loader, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )
    torch.save(probe.state_dict(),"probe.pt")

print("Training finished.")


# ===================== 工具：解析 label 字符串 =====================
def parse_label_str(label_str):
    """
    将 "<作品名>:月曼清游图<作者名>:冷枚"
    解析成 (work_title, author_name_cn)
    """
    work_title = ""
    author_name = ""
    try:
        if "<作品名>:" in label_str and "<作者名>:" in label_str:
            tmp = label_str.split("<作品名>:", 1)[1]
            work_title, author_name = tmp.split("<作者名>:", 1)
            work_title = work_title.strip()
            author_name = author_name.strip()
    except Exception:
        pass
    return work_title, author_name


# ===================== Probing 预测 + 存 jsonl =====================
@torch.no_grad()
def run_probing_and_dump(
    samples,
    feature_extractor,
    probe,
    processor,
    id2label,
    output_path,
    max_pixels=80_000_000,
    print_every=20,
):
    """
    对给定 samples 跑一遍 probing，打印部分结果，并将所有结果写入 jsonl。
    """
    probe.eval()
    feature_extractor.eval()

    results = []

    image_token = getattr(processor, "image_token", "<image>")

    for idx, s in enumerate(samples):
        # 1. 读图 + 防炸
        image = Image.open(s["image_path"]).convert("RGB")
        if image.width * image.height > max_pixels:
            scale = (max_pixels / (image.width * image.height)) ** 0.5
            new_w = int(image.width * scale)
            new_h = int(image.height * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # 2. 构造输入文本
        text = s["question"]
        full_text = f"{image_token}\n{text}"

        # 3. processor 编码
        enc = processor(
            images=[image],
            text=[full_text],
            padding=True,
            return_tensors="pt",
        )

        # 4. 特征 + 预测
        features = feature_extractor(enc)
        logits, _ = probe(features)
        pred_id = logits.argmax(dim=-1).item()
        pred_label_str = id2label[pred_id]

        # 5. 真值
        gt_label_str = s["label"]
        correct = (pred_label_str == gt_label_str)

        gt_work_title = s["work_title"]
        gt_author_name = s["author_name_cn"]

        pred_work_title, pred_author_name = parse_label_str(pred_label_str)

        result = {
            "image_path": s["image_path"],
            "question": text,
            "gt_label": gt_label_str,
            "gt_work_title": gt_work_title,
            "gt_author_name_cn": gt_author_name,
            "pred_label": pred_label_str,
            "pred_work_title": pred_work_title,
            "pred_author_name_cn": pred_author_name,
            "correct": correct,
        }

        results.append(result)

        # 打印前几个样本 & 每 print_every 个样本打印一次
        # if idx < 5 or (idx + 1) % print_every == 0:
        #     print(json.dumps(result, ensure_ascii=False))

    # 6. 写入 jsonl 文件
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} probing predictions to: {output_path}")


# ====== 实际调用：对整个数据集跑一遍，并存到 jsonl ======
output_jsonl_path = "probing_predictions_all.jsonl"
run_probing_and_dump(
    samples=samples,                # 也可以只传 val_samples
    feature_extractor=feature_extractor,
    probe=probe,
    processor=processor,
    id2label=id2label,
    output_path=output_jsonl_path,
    max_pixels=max_pixels,
    print_every=20,
)
