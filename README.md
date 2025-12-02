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

## 引用

本项目基于以下技术构建：
- Qwen2.5-VL: Alibaba 的视觉语言大模型
- PyTorch: 深度学习框架
- Transformers: Hugging Face 模型库

