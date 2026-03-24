import logging
import threading
import time
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict

logger = logging.getLogger(__name__)

class DataAcquisition:
    def __init__(self, interface=None):
        self.interface = interface
        self.running = False
        self.flows = defaultdict(dict)
        self.flow_queue = []
        self.lock = threading.Lock()
        self.session_timeout = 60  # 流超时时间（秒）
        self.max_packets_per_flow = 100  # 每个流的最大包数
        
    def start(self):
        self.running = True
        logger.info('Starting data acquisition...')
        
        # 启动抓包线程
        sniff_thread = threading.Thread(target=self._sniff_packets)
        sniff_thread.daemon = True
        sniff_thread.start()
        
        # 启动流管理线程
        manage_thread = threading.Thread(target=self._manage_flows)
        manage_thread.daemon = True
        manage_thread.start()
    
    def _sniff_packets(self):
        try:
            sniff(
                iface=self.interface,
                prn=self._process_packet,
                filter="ip",
                store=False,
                stop_filter=lambda x: not self.running
            )
        except Exception as e:
            logger.error(f'Error in packet sniffing: {e}')
    
    def _process_packet(self, packet):
        if IP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            protocol = packet[IP].proto
            
            if TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
            elif UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
            else:
                return
            
            # 创建流的唯一标识符（五元组）
            flow_id = (src_ip, dst_ip, src_port, dst_port, protocol)
            
            with self.lock:
                current_time = time.time()
                
                # 初始化流
                if flow_id not in self.flows:
                    self.flows[flow_id] = {
                        'src_ip': src_ip,
                        'dst_ip': dst_ip,
                        'src_port': src_port,
                        'dst_port': dst_port,
                        'protocol': protocol,
                        'packets': [],
                        'start_time': current_time,
                        'last_activity': current_time,
                        'packet_count': 0
                    }
                
                flow = self.flows[flow_id]
                
                # 更新流信息
                flow['last_activity'] = current_time
                flow['packet_count'] += 1
                
                # 提取数据包信息
                packet_info = {
                    'timestamp': current_time,
                    'length': len(packet),
                    'payload': bytes(packet.payload) if hasattr(packet, 'payload') else b''
                }
                
                # 添加数据包到流
                if flow['packet_count'] <= self.max_packets_per_flow:
                    flow['packets'].append(packet_info)
                
                # 检查流是否达到最大包数
                if flow['packet_count'] == self.max_packets_per_flow:
                    self._finalize_flow(flow_id)
    
    def _manage_flows(self):
        while self.running:
            time.sleep(1)
            current_time = time.time()
            
            with self.lock:
                # 检查超时流
                expired_flows = []
                for flow_id, flow in self.flows.items():
                    if current_time - flow['last_activity'] > self.session_timeout:
                        expired_flows.append(flow_id)
                
                # 处理超时流
                for flow_id in expired_flows:
                    self._finalize_flow(flow_id)
    
    def _finalize_flow(self, flow_id):
        if flow_id in self.flows:
            flow = self.flows.pop(flow_id)
            # 计算流持续时间
            flow['duration'] = flow['last_activity'] - flow['start_time']
            # 添加到流队列
            self.flow_queue.append(flow)
    
    def get_flows(self):
        with self.lock:
            flows = self.flow_queue.copy()
            self.flow_queue.clear()
            return flows
    
    def stop(self):
        self.running = False
        logger.info('Data acquisition stopped')
