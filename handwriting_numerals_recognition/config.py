# ==================== 配置文件 ====================
# 这个文件存储所有模型训练和测试的超参数配置
# 便于统一管理，避免在代码中硬编码数值

# 数据相关配置
DATA_PATH = './data'         # MNIST数据集下载路径
BATCH_SIZE = 64              # 批次大小
NUM_WORKERS = 0              # 数据加载线程数（Windows建议设为0）

# 模型相关配置
NUM_CLASSES = 10             # 数字类别数（0-9共10个）
INPUT_CHANNELS = 1           # 输入通道数（灰度图为1）

# 训练相关配置
NUM_EPOCHS = 10              # 训练轮数
LEARNING_RATE = 1e-3         # 学习率
WEIGHT_DECAY = 0             # 权重衰减（L2正则化）
OPTIMIZER = 'Adam'           # 优化器类型

# 日志和保存
LOGS_DIR = './logs'          # TensorBoard日志保存目录
MODEL_SAVE_PATH = './final_model.pth'  # 模型保存路径

# 设备配置
DEVICE = 'cuda'              # 使用GPU，如果没有GPU改为'cpu'
