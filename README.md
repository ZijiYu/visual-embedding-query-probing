# 国画多属性识别 Probing 项目

## 项目概述

本项目基于 Qwen2.5-VL-7B-Instruct 大模型，实现对国画作品多个属性的自动识别，包括作品名称、作者、画派、技法、构图和艺术风格等。项目采用两阶段训练策略：首先预计算视觉-语言特征，然后训练轻量级的线性分类器进行多任务学习。

## 核心特性

- 🎨 **多属性识别**：支持同时识别国画作品的6个核心属性
- 🧠 **大模型基础**：基于 Qwen2.5-VL-7B-Instruct 视觉语言模型
- ⚡ **高效训练**：两阶段架构，预计算特征，避免重复推理
- 🔄 **多任务学习**：集成式多头架构，支持属性间表示共享
- 📊 **完整评估**：提供字段级和样本级的准确率评估

## 项目结构

```
Probing/
├── data/                           # 数据集目录
│   ├── full_data.jsonl            # 完整标注数据集
│   ├── shuffled.jsonl             # 随机打乱版本
│   └── test_case.jsonl            # 测试用例
├── scripts/                        # 核心脚本
│   ├── precompute_features.py     # 预计算视觉-语言特征
│   ├── train_probe.py             # 训练多头线性probe
│   ├── run_probing.py             # 推理和评估
│   └── get_features.py            # 特征提取和数据处理工具
├── src/                           # 生成文件目录
│   ├── precomputed_features.pt    # 预计算特征缓存
│   ├── multi_field_probe_on_features.pt  # 训练好的模型权重
│   └── probing_predictions_*.jsonl # 预测结果
├── configs/                       # 配置文件
│   └── probing_vl.py             # 原始VL probing配置
├── logs/                          # 日志目录
│   ├── setup_logger.py           # 日志配置
│   └── train_probe.log           # 训练日志
└── README.md                      # 本文档
```

## 技术架构

### 1. 数据格式

项目使用 JSONL 格式的标注数据，每行包含一张国画作品的完整信息：

```json
{
  "filename": "share/data/jpg/6a60fb26-4070-11ed-9adc-c934f75048ef.jpg",
  "work_title": "月曼清游图",
  "work_dynasty": "清代",
  "work_media_type": "绢本",
  "author_name_cn": "冷枚",
  "work_schools": "清代宫廷画派",
  "work_composition": "对称构图, 留白构图, S形构图",
  "artistry_style": "工笔重彩",
  "work_techniques": "铁线描, 没骨法, 渲染"
}
```

**支持的属性字段：**
- `title` (work_title): 作品名称
- `author` (author_name_cn): 作者姓名
- `work_schools`: 画派流派
- `work_techniques`: 绘画技法
- `work_composition`: 构图方式
- `artistry_style`: 艺术风格

### 2. 模型架构

#### 特征提取器 (QwenVLFeatureExtractor)
```python
class QwenVLFeatureExtractor(nn.Module):
    def __init__(self, qwen_model, top_k_layers=4):
        # 基于Qwen2.5-VL提取多层特征
        # 使用最后top_k_layers层的平均池化
```

#### 多头线性分类器 (MultiHeadProbe)
```python
class MultiHeadProbe(nn.Module):
    def __init__(self, input_dim, field2num_labels):
        self.heads = nn.ModuleDict({
            field: nn.Linear(input_dim, n_labels)
            for field, n_labels in field2num_labels.items()
        })
```

### 3. 两阶段训练策略

**阶段一：特征预计算**
- 使用冻结的 Qwen2.5-VL 模型提取视觉-语言特征
- 支持图像和文本的多模态融合
- 特征进行 L2 归一化处理

**阶段二：线性分类器训练**
- 在预计算特征上训练轻量级线性分类器
- 多任务联合优化，损失函数为各字段损失之和
- 支持灵活的字段组合配置

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision transformers pillow tqdm

# 确保数据路径正确
export DATA_PATH="/workspace/probing/Probing/data/full_data.jsonl"
export MODEL_PATH="/workspace/data/checkpoint-2077"  # 微调后的Qwen2.5-VL
```

### 2. 数据准备

确保数据集格式正确，包含图像文件和JSONL标注：

```python
# 示例：加载数据
from scripts.get_features import load_samples

samples = load_samples(
    file_path="/workspace/probing/Probing/data/full_data.jsonl",
    base_image_dir="/workspace/"
)
```

### 3. 特征预计算

```bash
cd /workspace/probing/Probing/scripts
python precompute_features.py
```

**输出：**
- 生成 `precomputed_features.pt` 文件
- 包含特征向量、标签映射和元数据

### 4. 训练Probe模型

```bash
python train_probe.py
```

**主要配置：**
```python
TARGET_FIELDS = ["title", "author"]  # 可配置需要训练的字段
batch_size = 64
epochs = 10
lr = 1e-3
```

### 5. 推理和评估

```bash
python run_probing.py
```

**输出：**
- 生成详细的预测结果 JSONL
- 字段级和样本级准确率报告
- 可视化预测样本

## 配置说明

### 特征提取配置

```python
# get_features.py
MAX_PIXELS = 313600          # 图像像素上限
TOP_K_LAYERS = 4             # 使用最后4层特征
BATCH_SIZE = 4               # 特征提取批次大小
```

### 训练配置

```python
# train_probe.py
TARGET_FIELDS = None         # None 表示使用所有字段，或指定 ["title", "author"]
EPOCHS = 10                 # 训练轮数
LEARNING_RATE = 1e-3        # 学习率
```

### 推理配置

```python
# run_probing.py
DEFAULT_FIELD_PROMPTS = {
    "title": "请据图片判断这幅画的作品名。只回答作品名。",
    "author": "请根据图片判断这幅画的作者。只回答作者的名字。",
    # 其他字段提示词...
}
```

## 性能表现

### 模型效果
基于当前数据集的训练结果：
- **Title识别准确率**: 取决于作品名复杂度
- **Author识别准确率**: 对于知名画家表现较好
- **Overall准确率**: 所有属性同时正确识别的比例

### 优化建议
1. **数据增强**: 扩充训练数据，特别是稀有类别
2. **提示词优化**: 针对不同属性设计更精准的提示词
3. **特征融合**: 尝试不同的特征聚合策略
4. **模型集成**: 结合多个模型的预测结果

## 常见问题

### Q1: 为什么选择两阶段训练？
**A**: 两阶段训练避免了重复的大模型推理，显著提高训练效率，同时支持灵活的实验配置。

### Q2: 如何添加新的属性字段？
**A**:
1. 在数据中添加对应字段
2. 更新 `LABEL_CONFIG` 映射
3. 重新运行完整的训练流程

### Q3: 训练显存不足怎么办？
**A**:
1. 减少 `batch_size`
2. 降低 `max_pixels` 限制
3. 使用梯度累积

### Q4: 如何处理长尾分布问题？
**A**:
1. 使用加权损失函数
2. 数据过采样
3. 类别平衡采样

## 文件说明

### 核心脚本详解

#### `precompute_features.py`
- **功能**: 预计算Qwen-VL的视觉-语言特征
- **输入**: 图像+文本数据，大模型路径
- **输出**: 预计算特征文件 (.pt)
- **关键特性**: 支持多字段标签，L2归一化

#### `train_probe.py`
- **功能**: 在预计算特征上训练多头线性分类器
- **输入**: 预计算特征，标签映射
- **输出**: 训练好的模型权重
- **关键特性**: 多任务联合训练，灵活字段配置

#### `run_probing.py`
- **功能**: 推理预测和结果评估
- **输入**: 训练好的模型，测试数据
- **输出**: 预测结果JSONL，准确率报告
- **关键特性**: 详细的错误分析，可视化输出

#### `get_features.py`
- **功能**: 数据处理和特征提取工具类
- **组件**: Dataset类、特征提取器、工具函数
- **特性**: 支持多模态输入，内存优化

## 扩展开发

### 添加新的属性类型
1. 更新数据标注格式
2. 修改 `LABEL_CONFIG` 配置
3. 调整提示词模板
4. 重新训练模型

### 集成其他视觉模型
1. 实现 `BaseFeatureExtractor` 接口
2. 调整输入数据格式
3. 修改模型训练脚本

### 优化推理性能
1. 模型量化
2. 批量推理
3. 特征缓存策略

## 引用

本项目基于以下技术构建：
- Qwen2.5-VL: Alibaba 的视觉语言大模型
- PyTorch: 深度学习框架
- Transformers: Hugging Face 模型库


**注意**: 本项目主要用于学术研究和国画数字化保护，请合理使用模型和数据。