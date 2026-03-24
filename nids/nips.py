import logging
import threading
import time
import subprocess
import platform

logger = logging.getLogger(__name__)

class NIPSEngine:
    def __init__(self):
        self.blocked_ips = {}
        self.lock = threading.Lock()
        self.block_duration = 3600  # IP封禁时间（秒）
        self.running = False
        
        # 启动IP管理线程
        self._start_ip_manager()
    
    def _start_ip_manager(self):
        self.running = True
        manager_thread = threading.Thread(target=self._manage_blocked_ips)
        manager_thread.daemon = True
        manager_thread.start()
    
    def block_ip(self, ip):
        with self.lock:
            if ip not in self.blocked_ips:
                # 执行封禁操作
                if self._execute_block(ip):
                    self.blocked_ips[ip] = time.time() + self.block_duration
                    logger.warning(f'Blocked IP: {ip} for {self.block_duration} seconds')
                else:
                    logger.error(f'Failed to block IP: {ip}')
            else:
                # 延长封禁时间
                self.blocked_ips[ip] = time.time() + self.block_duration
                logger.info(f'Extended block for IP: {ip}')
    
    def _execute_block(self, ip):
        try:
            system = platform.system()
            if system == 'Linux':
                # 使用iptables
                cmd = f'sudo iptables -A INPUT -s {ip} -j DROP'
                subprocess.run(cmd, shell=True, check=True)
                logger.info(f'Executed: {cmd}')
                return True
            elif system == 'Windows':
                # 使用Windows防火墙
                cmd = f'netsh advfirewall firewall add rule name="Block {ip}" dir=in action=block remoteip={ip}'
                subprocess.run(cmd, shell=True, check=True)
                logger.info(f'Executed: {cmd}')
                return True
            else:
                logger.warning(f'Blocking IP not supported on {system}')
                return False
        except Exception as e:
            logger.error(f'Error blocking IP {ip}: {e}')
            return False
    
    def _unblock_ip(self, ip):
        try:
            system = platform.system()
            if system == 'Linux':
                # 使用iptables
                cmd = f'sudo iptables -D INPUT -s {ip} -j DROP'
                subprocess.run(cmd, shell=True, check=True)
                logger.info(f'Executed: {cmd}')
                return True
            elif system == 'Windows':
                # 使用Windows防火墙
                cmd = f'netsh advfirewall firewall delete rule name="Block {ip}"'
                subprocess.run(cmd, shell=True, check=True)
                logger.info(f'Executed: {cmd}')
                return True
            else:
                logger.warning(f'Unblocking IP not supported on {system}')
                return False
        except Exception as e:
            logger.error(f'Error unblocking IP {ip}: {e}')
            return False
    
    def _manage_blocked_ips(self):
        while self.running:
            time.sleep(60)  # 每分钟检查一次
            current_time = time.time()
            
            with self.lock:
                expired_ips = []
                for ip, expiry_time in self.blocked_ips.items():
                    if current_time > expiry_time:
                        expired_ips.append(ip)
                
                for ip in expired_ips:
                    if self._unblock_ip(ip):
                        del self.blocked_ips[ip]
                        logger.info(f'Unblocked expired IP: {ip}')
    
    def get_blocked_ips(self):
        with self.lock:
            return list(self.blocked_ips.keys())
    
    def stop(self):
        self.running = False
        # 清理所有被封禁的IP
        with self.lock:
            for ip in list(self.blocked_ips.keys()):
                self._unblock_ip(ip)
            self.blocked_ips.clear()
        logger.info('NIPS engine stopped')
