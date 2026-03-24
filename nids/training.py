import logging
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json

logger = logging.getLogger(__name__)

class TrafficDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = []
        self._load_dataset(dataset_path)
    
    def _load_dataset(self, dataset_path):
        try:
            # 这里使用伪代码加载ISCX-VPN数据集
            # 实际实现时需要根据具体数据集格式进行调整
            logger.info(f'Loading dataset from {dataset_path}')
            
            # 假设数据集是JSON格式的文件列表
            for filename in os.listdir(dataset_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(dataset_path, filename)
                    with open(filepath, 'r') as f:
                        flow_data = json.load(f)
                        self.data.append(flow_data)
            
            logger.info(f'Loaded {len(self.data)} samples')
        except Exception as e:
            logger.error(f'Error loading dataset: {e}')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        flow = self.data[idx]
        # 提取特征（这里简化处理，实际需要使用FeatureEngineering）
        statistical_features = np.array(flow.get('statistical_features', [0]*17), dtype=np.float32)
        payload_features = np.array(flow.get('payload_features', [0]*512), dtype=np.float32).reshape(1, -1)
        label = flow.get('label', 0)
        
        return {
            'statistical': statistical_features,
            'payload': payload_features,
            'label': label
        }

class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.device = model.device
    
    def train(self, dataset_path, epochs, batch_size, learning_rate):
        # 创建数据集和数据加载器
        dataset = TrafficDataset(dataset_path)
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练循环
        best_val_acc = 0.0
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                # 准备数据
                statistical = batch['statistical'].to(self.device)
                payload = batch['payload'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                outputs = self.model({'statistical': statistical, 'payload': payload})
                loss = criterion(outputs, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 计算统计信息
                train_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = 100.0 * train_correct / train_total
            
            # 验证阶段
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            logger.info(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.info(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    def evaluate(self, data_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                statistical = batch['statistical'].to(self.device)
                payload = batch['payload'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model({'statistical': statistical, 'payload': payload})
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def save_model(self, model_path):
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_model(model_path)
