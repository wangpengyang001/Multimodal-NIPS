import logging
import threading
import time
from .data_acquisition import DataAcquisition
from .feature_engineering import FeatureEngineering
from .model import MultimodalModel
from .training import ModelTrainer
from .nips import NIPSEngine

logger = logging.getLogger(__name__)

class NIDSSystem:
    def __init__(self, interface=None, model_path='models/model.pth'):
        self.interface = interface
        self.model_path = model_path
        self.data_acquisition = DataAcquisition(interface)
        self.feature_engineering = FeatureEngineering()
        self.model = MultimodalModel()
        self.model.load_model(model_path)
        self.nips_engine = NIPSEngine()
        self.running = False
        
    def start_detection(self):
        self.running = True
        logger.info('Starting NIDS detection engine')
        
        # 启动数据采集线程
        acquisition_thread = threading.Thread(target=self.data_acquisition.start)
        acquisition_thread.daemon = True
        acquisition_thread.start()
        
        # 启动检测线程
        detection_thread = threading.Thread(target=self._detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info('Stopping NIDS...')
            self.stop()
    
    def _detection_loop(self):
        while self.running:
            try:
                # 从数据采集器获取网络流
                flows = self.data_acquisition.get_flows()
                
                for flow in flows:
                    # 提取特征
                    features = self.feature_engineering.extract_features(flow)
                    if features:
                        # 模型预测
                        prediction = self.model.predict(features)
                        
                        # 防御执行
                        if prediction == 1:  # 异常流量
                            src_ip = flow['src_ip']
                            logger.warning(f'Malicious traffic detected from {src_ip}')
                            self.nips_engine.block_ip(src_ip)
                            
            except Exception as e:
                logger.error(f'Error in detection loop: {e}')
                time.sleep(1)
    
    def train(self, dataset_path, epochs, batch_size, learning_rate):
        logger.info(f'Starting model training with dataset: {dataset_path}')
        trainer = ModelTrainer(self.model)
        trainer.train(dataset_path, epochs, batch_size, learning_rate)
        trainer.save_model(self.model_path)
        logger.info(f'Model saved to {self.model_path}')
    
    def stop(self):
        self.running = False
        self.data_acquisition.stop()
        self.nips_engine.stop()
        logger.info('NIDS stopped')
