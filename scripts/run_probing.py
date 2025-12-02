# run_probing.py
# 使用 Qwen-VL + 训练好的 probe，对每个字段做 probing 并导出 jsonl
import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import json
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoProcessor

from logs.setup_logger import setup_logger
from get_features import (
    load_samples,
    QwenVLFeatureExtractor,
    image_resize,
)

# ===================== 配置区 =====================

model_path = "/workspace/data/checkpoint-2077"
file_path = "/workspace/probing/Probing/data/full_data.jsonl"
base_image_dir = "/workspace/"

probe_path = "/workspace/probing/Probing/src/multi_field_probe_on_features.pt"
output_jsonl_path = "/workspace/probing/Probing/src/probing_predictions_all_multi_fields.jsonl"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = setup_logger()


# ===================== 多头线性 probe（结构需和训练时一致） =====================
class MultiHeadProbe(nn.Module):
    def __init__(self, input_dim, field2num_labels):
        super().__init__()
        self.heads = nn.ModuleDict({
            field: nn.Linear(input_dim, n_labels)
            for field, n_labels in field2num_labels.items()
        })

    def forward(self, features):
        features = F.normalize(features, dim=-1)
        logits_dict = {
            field: head(features)
            for field, head in self.heads.items()
        }
        return logits_dict


# ===================== 加载大模型 & processor（用于 probing） =====================
def build_model_and_processor(model_path, device):

    logger.info("Loading tokenizer & model (for probing)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        # 可以考虑用半精度：torch_dtype=torch.float16,
        device_map=None,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 冻结大模型参数
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size
    logger.info(f"Hidden size: {hidden_size}")
    logger.info(f"Model param device: {next(model.parameters()).device}")

    return tokenizer, model, processor, hidden_size


# ===================== Probing 预测 + 存 jsonl（多字段） =====================
@torch.no_grad()
def run_probing_and_dump_multi_fields(
    samples,
    feature_extractor,
    probe,
    processor,
    field2id2label,
    output_path,
    system_prompt=(
        "你是一个国画领域的AI模型，擅长回答关于国画的问题。"
        "请赏析这幅画，基于其内容、结构、技法来分析这幅画的相关信息，随后，回答以下问题。"
    ),
):
    """
    对给定 samples 跑一遍 probing：
      - 对每个字段分别构造一个问句，单独抽特征 + 通过对应 head 预测
      - 将 gt / pred / 是否正确 写入 jsonl
      - 同时计算并返回各字段准确率和样本级 overall 准确率
    """
    feature_extractor.eval()
    probe.eval()

    results = []

    image_token = getattr(processor, "image_token", "<image>")

    # 针对不同 field 的 user 问句模板（你可以按需改写）
    default_field_prompts = {
        "title": "请据图片判断这幅画的作品名。只回答作品名。",
        "author": "请根据图片判断这幅画的作者。只回答作者的名字。",
        # "work_techniques": "请判断这幅画所使用的主要绘画技法。只回答技法名称。",
        # "work_schools": "请判断这幅画属于哪种画派。只回答画派名称。",
        # "artistry_style": "请判断这幅画的艺术风格。只回答风格名称。",
    }

    active_fields = list(field2id2label.keys())

    # 准确率统计
    field_correct = {field: 0 for field in active_fields}
    field_total = {field: 0 for field in active_fields}
    overall_correct_all = 0   # 所有有 GT 的字段都预测正确的样本数
    overall_total_all = 0     # 至少有一个字段有 GT 的样本数

    for idx, s in enumerate(tqdm(samples, desc="Probing")):
        # 1. 读图 + resize
        image = Image.open(s["image_path"]).convert("RGB")
        image = image_resize(image)

        result = {
            "image_path": s["image_path"],
        }

        all_correct = True
        any_has_gt = False

        for field in active_fields:
            if field not in probe.heads:
                continue

            user_prompt = default_field_prompts.get(
                field, f"请根据图片回答该作品的 {field}。只回答{field}。"
            )

            field_text = (
                f"<|system|>\n{system_prompt}\n"
                f"<|user|>\n{image_token}\n{user_prompt}"
            )

            # 保留原问题文本，方便之后分析
            result[f"{field}_question"] = field_text

            enc_field = processor(
                images=[image],
                text=[field_text],
                padding=True,
                return_tensors="pt",
            )

            # 特征 + 预测
            enc_field = {k: v.to(device) for k, v in enc_field.items()}
            features_field = feature_extractor(enc_field)  # (1, D)
            logits_dict = probe(features_field)
            logits = logits_dict[field]  # (1, num_classes_for_field)

            pred_id = logits.argmax(dim=-1).item()
            pred_label = field2id2label[field][pred_id]

            # 真值：来自 load_samples 时保存的 s[field]
            gt_label = s.get(field, None)
            correct = (pred_label == gt_label) if gt_label is not None else None

            result[f"gt_{field}"] = gt_label
            result[f"pred_{field}"] = pred_label
            result[f"correct_{field}"] = correct

            # 累计字段级准确率
            if gt_label is not None:
                field_total[field] += 1
                if correct:
                    field_correct[field] += 1
                any_has_gt = True
                if not correct:
                    all_correct = False

        # 如果所有有 gt 的字段都对了，则标记 correct_all
        result["correct_all_fields"] = (all_correct and any_has_gt)

        # 样本级 overall 统计
        if any_has_gt:
            overall_total_all += 1
            if all_correct:
                overall_correct_all += 1

        results.append(result)
        # logger.info(json.dumps(result, ensure_ascii=False))

    # 写入 jsonl 文件
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Saved probing predictions to: {output_path}")

    # 计算准确率
    field_accs = {}
    for field in active_fields:
        if field_total[field] > 0:
            field_accs[field] = field_correct[field] / field_total[field]
        else:
            field_accs[field] = 0.0

    overall_acc = (
        overall_correct_all / overall_total_all
        if overall_total_all > 0 else 0.0
    )

    # 打印准确率结果
    for field, acc in field_accs.items():
        logger.info(f"Field [{field}] accuracy: {acc:.4f} "
                    f"({field_correct[field]}/{field_total[field]})")

    logger.info(
        f"Overall sample-level accuracy (all available fields correct): "
        f"{overall_acc:.4f} ({overall_correct_all}/{overall_total_all})"
    )

    # 返回给调用者（如果你想在 main 里再用）
    return field_accs, overall_acc


def main():
    # ===== 1. 加载训练好的 probe =====
    logger.info("Loading trained probe from: %s", probe_path)
    ckpt = torch.load(probe_path, map_location="cpu")

    field2label2id = ckpt["field2label2id"]
    field2id2label = ckpt["field2id2label"]
    active_fields = ckpt["active_fields"]

    # input_dim 需要和当时训练时的 hidden_size 一致
    # 这里默认用 Qwen 模型的 hidden_size：
    tokenizer, model, processor, hidden_size = build_model_and_processor(
        model_path=model_path,
        device=device,
    )

    field2num_labels = {f: len(field2label2id[f]) for f in active_fields}
    probe = MultiHeadProbe(
        input_dim=hidden_size,
        field2num_labels=field2num_labels,
    ).to(device)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()

    # ===== 2. 构建特征提取器 =====
    feature_extractor = QwenVLFeatureExtractor(model, top_k_layers=1)

    # ===== 3. 加载样本（用于拿 image_path / gt label 字符串）=====
    logger.info("Loading samples from jsonl for probing...")
    samples = load_samples(
        file_path=file_path,
        base_image_dir=base_image_dir,
    )
    logger.info("Total samples: %d", len(samples))

    # ===== 4. 运行 probing 并导出 jsonl，同时得到准确率 =====
    field_accs, overall_acc = run_probing_and_dump_multi_fields(
        samples=samples,
        feature_extractor=feature_extractor,
        probe=probe,
        processor=processor,
        field2id2label=field2id2label,
        output_path=output_jsonl_path
    )

    logger.info("Final field accuracies: %s", field_accs)
    logger.info("Final overall accuracy (all fields correct): %.4f", overall_acc)
    logger.info("Done.")


if __name__ == "__main__":
    main()
