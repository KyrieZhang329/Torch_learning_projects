# 文本生成

基于PyTorch的LSTM文本生成模型，可以根据种子文本生成新文本。

## 项目结构

```
Text_Processing/
├── config.py              # 参数配置
├── utils.py               # 工具函数
├── Net.py                 # 神经网络模型
├── train.py               # 训练脚本
├── generate.py            # 生成脚本
├── main.py                # 主程序
├── data.txt               # 训练数据（用户提供）
├── vocab.json             # 词汇表（自动生成）
└── text_generator_model.pth  # 训练好的模型
```

## 使用方法

### 准备训练数据

把要用的文本放在`data.txt`中（英文小说、剧本、代码等都可以）。

### 安装依赖

官网下载torch

### 训练模型


运行 train.py


会加载data.txt，训练模型并保存到`text_generator_model.pth`。

### 生成文本

运行 generate.py


## 配置参数

config.py中可以修改：
- `EMBEDDING_DIM`: 词嵌入维度，默认128
- `HIDDEN_DIM`: LSTM隐藏层维度，默认256
- `NUM_LAYERS`: LSTM层数，默认2
- `BATCH_SIZE`: 批次大小，默认32
- `SEQUENCE_LENGTH`: 序列长度，默认50
- `NUM_EPOCHS`: 训练轮数，默认50
- `LEARNING_RATE`: 学习率，默认0.001
- `TEMPERATURE`: 温度参数，默认0.8（控制生成的随机性）

## 模型架构

- 嵌入层（将字符转为向量）
- 2层LSTM（学习文本特征）
- 全连接层（输出预测）

## 工作原理

1. 编码：把文本转换为字符ID
2. 序列化：创建50个字符的序列
3. 训练：模型学习序列的下一个字符
4. 生成：输入种子文本，逐个预测字符

## 说明

- CPU上小文本训练约几分钟
- 模型只能学习文本统计特性，无法理解深层含义
- 可以调整TEMPERATURE参数改变生成的多样性：
  - 0.5: 更保守
  - 1.0: 平衡
  - 1.5: 更多样化

## Net.py

定义两个模型：
- `TextGeneratorRNN`: 基于LSTM
- `TextGeneratorGRU`: 基于GRU（更轻量）
