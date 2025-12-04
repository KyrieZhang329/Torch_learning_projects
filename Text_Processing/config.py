DATA_PATH = 'data.txt'  # 训练数据文件路径
ENCODING = 'utf-8'      # 文件编码方式


VOCAB_SIZE = None       # 词汇表大小（将在加载数据时自动计算）
EMBEDDING_DIM = 128     # 词嵌入维度
HIDDEN_DIM = 256        # 隐藏层维度
NUM_LAYERS = 2          # RNN层数
DROPOUT = 0.3           # Dropout比例，防止过拟合


BATCH_SIZE = 32         # 批次大小
SEQUENCE_LENGTH = 50    # 每个序列的长度
NUM_EPOCHS = 50         # 训练轮数
LEARNING_RATE = 0.001   # 学习率
WEIGHT_DECAY = 1e-5     # 权重衰减（L2正则化）


SEED_TEXT = "The"       # 种子文本（生成的起始文本）
GENERATE_LENGTH = 200   # 生成文本的长度
TEMPERATURE = 0.8       # 温度参数（控制随机性，值越低越确定性）
TOP_K = 10              # Top-K采样参数


DEVICE = 'cuda'         # 使用GPU，如果没有GPU改为'cpu'
MODEL_SAVE_PATH = 'text_generator_model.pth'  # 模型保存路径
