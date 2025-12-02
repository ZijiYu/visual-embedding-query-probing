import json
import random

input_path = "/workspace/probing/Probing/data/test_case.jsonl"
output_path = "shuffled.jsonl"

# 1. 读入所有行
lines = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            lines.append(line)

# 2. 打乱顺序
random.shuffle(lines)

# 3. 写出为新的 jsonl
with open(output_path, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print(f"Shuffled JSONL saved to {output_path}")