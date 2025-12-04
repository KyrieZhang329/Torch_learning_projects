# 物体识别

基于PyTorch的CNN模型，用于识别CIFAR-10数据集中的10种物体。

## 项目结构

```
objective_recognition/
├── config.py    # 参数配置
├── utils.py     # 工具函数
├── Net.py       # 神经网络模型
├── dis_model.py # 训练脚本
├── test.py      # 测试脚本
└── test_data/   # 测试图片文件夹
```

## 使用方法

### 安装依赖

官网下载torch

### 训练模型


运行 dis_model.py

会自动下载CIFAR-10数据，训练模型并保存到`final_model.pth`。

### 测试模型

运行 test.py

自动识别`test_data/`文件夹中的图片。

## 配置参数

config.py中可以修改：
- `BATCH_SIZE`: 批次大小，默认64
- `NUM_EPOCHS`: 训练轮数，默认10
- `LEARNING_RATE`: 学习率，默认1e-2

## 模型架构

三层卷积网络，包含：
- 三层卷积层（32、32、64个滤波器）
- 最大池化层
- 两层全连接层

输入：32x32的RGB彩色图像
输出：10个类别的概率分布

## CIFAR-10类别

飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车

## 说明

- CPU上训练约30-60分钟
- 可以修改config.py来调整超参数
- 使用SGD优化器
