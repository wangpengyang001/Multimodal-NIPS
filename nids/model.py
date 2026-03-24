import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        
        # 分支A：处理统计特征的MLP
        self.statistical_branch = nn.Sequential(
            nn.Linear(17, 128),  # 17个统计特征
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # 分支B：处理Payload特征的1D-CNN
        self.payload_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(32 + 128, 64),  # 32 (统计特征) + 128 (Payload特征)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # 二分类：正常/异常
        )
        
        # 检查设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        logger.info(f'Model initialized on {self.device}')
    
    def forward(self, features):
        # 处理统计特征
        statistical_features = features['statistical']
        statistical_features = torch.tensor(statistical_features, dtype=torch.float32).to(self.device)
        stat_out = self.statistical_branch(statistical_features)
        
        # 处理Payload特征
        payload_features = features['payload']
        payload_features = torch.tensor(payload_features, dtype=torch.float32).to(self.device)
        payload_out = self.payload_branch(payload_features)
        
        # 融合特征
        combined = torch.cat([stat_out, payload_out], dim=1)
        output = self.fusion(combined)
        
        return output
    
    def predict(self, features):
        self.eval()
        with torch.no_grad():
            output = self.forward(features)
            _, prediction = torch.max(output, 1)
            return prediction.item()
    
    def load_model(self, model_path):
        try:
            if torch.cuda.is_available():
                self.load_state_dict(torch.load(model_path))
            else:
                self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            logger.info(f'Model loaded from {model_path}')
        except Exception as e:
            logger.warning(f'Failed to load model: {e}. Using random weights.')
    
    def save_model(self, model_path):
        try:
            torch.save(self.state_dict(), model_path)
            logger.info(f'Model saved to {model_path}')
        except Exception as e:
            logger.error(f'Failed to save model: {e}')
