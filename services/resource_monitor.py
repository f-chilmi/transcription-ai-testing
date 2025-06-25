import os

os.environ['USE_NNPACK'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import time
import psutil
import threading
import torch
torch.backends.nnpack.enabled = False



class ResourceMonitor:
    """Monitors CPU, memory, and other system resources during processing"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'timestamps': []
        }
    
    def start_monitoring(self):
        self.monitoring = True
        self.stats = {'cpu_usage': [], 'memory_usage': [], 'timestamps': []}
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def _monitor(self):
        while self.monitoring:
            self.stats['cpu_usage'].append(psutil.cpu_percent(percpu=True))
            self.stats['memory_usage'].append(psutil.virtual_memory().percent)
            self.stats['timestamps'].append(time.time())
            time.sleep(0.5)  # Monitor every 0.5 seconds
    
    def get_summary(self):
        if not self.stats['cpu_usage']:
            return {}
        
        avg_cpu_per_core = [sum(core_usage)/len(core_usage) for core_usage in zip(*self.stats['cpu_usage'])]
        max_cpu_per_core = [max(core_usage) for core_usage in zip(*self.stats['cpu_usage'])]
        
        return {
            'avg_cpu_per_core': avg_cpu_per_core,
            'max_cpu_per_core': max_cpu_per_core,
            'avg_memory': sum(self.stats['memory_usage']) / len(self.stats['memory_usage']),
            'max_memory': max(self.stats['memory_usage']),
            'total_cpu_cores': len(avg_cpu_per_core),
            'monitoring_duration': self.stats['timestamps'][-1] - self.stats['timestamps'][0] if self.stats['timestamps'] else 0
        }
