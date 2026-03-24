import logging
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self):
        self.payload_max_length = 512  # Payload最大长度
    
    def extract_features(self, flow):
        try:
            # 提取统计特征
            statistical_features = self._extract_statistical_features(flow)
            
            # 提取Payload特征
            payload_features = self._extract_payload_features(flow)
            
            if statistical_features is None or payload_features is None:
                return None
            
            return {
                'statistical': statistical_features,
                'payload': payload_features
            }
        except Exception as e:
            logger.error(f'Error extracting features: {e}')
            return None
    
    def _extract_statistical_features(self, flow):
        packets = flow.get('packets', [])
        if not packets:
            return None
        
        # 提取包长列表
        packet_lengths = [p['length'] for p in packets]
        
        # 提取时间间隔列表
        timestamps = [p['timestamp'] for p in packets]
        time_intervals = []
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]
            time_intervals.append(interval)
        
        # 计算统计特征
        features = []
        
        # 1. 流持续时间
        features.append(flow.get('duration', 0))
        
        # 2. 数据包数量
        features.append(len(packets))
        
        # 3. 平均包长
        features.append(np.mean(packet_lengths) if packet_lengths else 0)
        
        # 4. 包长标准差
        features.append(np.std(packet_lengths) if len(packet_lengths) > 1 else 0)
        
        # 5. 包长最小值
        features.append(np.min(packet_lengths) if packet_lengths else 0)
        
        # 6. 包长最大值
        features.append(np.max(packet_lengths) if packet_lengths else 0)
        
        # 7. 包长中位数
        features.append(np.median(packet_lengths) if packet_lengths else 0)
        
        # 8. 包长偏度
        features.append(stats.skew(packet_lengths) if len(packet_lengths) > 2 else 0)
        
        # 9. 包长峰度
        features.append(stats.kurtosis(packet_lengths) if len(packet_lengths) > 3 else 0)
        
        # 10. 平均时间间隔
        features.append(np.mean(time_intervals) if time_intervals else 0)
        
        # 11. 时间间隔标准差
        features.append(np.std(time_intervals) if len(time_intervals) > 1 else 0)
        
        # 12. 时间间隔最小值
        features.append(np.min(time_intervals) if time_intervals else 0)
        
        # 13. 时间间隔最大值
        features.append(np.max(time_intervals) if time_intervals else 0)
        
        # 14. 总字节数
        features.append(sum(packet_lengths))
        
        # 15. 协议类型（TCP=6, UDP=17）
        features.append(flow.get('protocol', 0))
        
        # 16. 源端口
        features.append(flow.get('src_port', 0))
        
        # 17. 目的端口
        features.append(flow.get('dst_port', 0))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_payload_features(self, flow):
        packets = flow.get('packets', [])
        if not packets:
            return None
        
        # 收集前N个字节的Payload
        payload_data = b''
        for packet in packets:
            payload = packet.get('payload', b'')
            payload_data += payload
            if len(payload_data) >= self.payload_max_length:
                break
        
        # 截断或零填充到固定长度
        if len(payload_data) > self.payload_max_length:
            payload_data = payload_data[:self.payload_max_length]
        else:
            payload_data = payload_data.ljust(self.payload_max_length, b'\x00')
        
        # 转换为numpy数组并归一化
        payload_array = np.frombuffer(payload_data, dtype=np.uint8)
        payload_array = payload_array.astype(np.float32) / 255.0
        
        # 重塑为适合1D-CNN的形状
        return payload_array.reshape(1, -1)
