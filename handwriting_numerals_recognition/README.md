# 手写数字识别

基于PyTorch的CNN模型，用于识别MNIST数据集中的手写数字。

## 项目结构

```
handwriting_numerals_recognition/
├── config.py      # 参数配置
├── utils.py       # 工具函数
├── Net.py         # 神经网络模型
├── reco_model.py  # 训练脚本
├── test.py        # 测试脚本
└── numerals/      # 测试图片文件夹
```

## 使用方法

### 安装依赖

官网下载torch

### 训练模型


运行 reco_model.py


会自动下载MNIST数据，训练模型并保存到`final_model.pth`。

### 测试模型

运行 test.py

自动识别`numerals/`文件夹中的图片。

## 配置参数

config.py中可以修改：
- `BATCH_SIZE`: 批次大小，默认64
- `NUM_EPOCHS`: 训练轮数，默认10
- `LEARNING_RATE`: 学习率，默认1e-3

## 模型架构

简单的CNN，包含：
- 两层卷积层（各32和64个滤波器）
- 最大池化层
- 两层全连接层

输入：28x28的灰度图像
输出：10个数字的概率分布

## 说明

- 在CPU上训练约5-10分钟
- 可以修改config.py来调整超参数
- 使用Adam优化器
