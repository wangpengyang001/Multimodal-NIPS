# 系统配置文件

# 数据采集配置
DATA_ACQUISITION = {
    'session_timeout': 60,  # 流超时时间（秒）
    'max_packets_per_flow': 100,  # 每个流的最大包数
    'interface': None  # 网络接口，None表示自动选择
}

# 特征工程配置
FEATURE_ENGINEERING = {
    'payload_max_length': 512  # Payload最大长度
}

# 模型配置
MODEL = {
    'model_path': 'models/model.pth',  # 模型保存路径
    'device': 'auto'  # 设备选择：auto, cuda, cpu
}

# NIPS配置
NIPS = {
    'block_duration': 3600,  # IP封禁时间（秒）
    'check_interval': 60  # IP检查间隔（秒）
}

# 训练配置
TRAINING = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_workers': 4
}
