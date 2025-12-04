
# 数据相关配置
DATA_PATH = './data'         # CIFAR-10数据集下载路径
BATCH_SIZE = 64              # 批次大小
NUM_WORKERS = 0              # 数据加载线程数（Windows建议设为0）

# 模型相关配置
NUM_CLASSES = 10             # CIFAR-10有10个类别
INPUT_CHANNELS = 3           # 输入通道数（RGB为3）
IMAGE_SIZE = 32              # CIFAR-10图片尺寸为32×32

# 训练相关配置
NUM_EPOCHS = 10              # 训练轮数
LEARNING_RATE = 1e-2         # 学习率
OPTIMIZER = 'SGD'            # 优化器类型
WEIGHT_DECAY = 0             # 权重衰减

# 日志和保存
LOGS_DIR = './logs'          # TensorBoard日志保存目录
MODEL_SAVE_PATH = './final_model.pth'  # 模型保存路径

# 设备配置
DEVICE = 'cuda'              # 使用GPU，如果没有GPU改为'cpu'
