---
title: GPU Resource Optimization in Kubernetes - From Waste to Efficiency in ML Workloads
tags: kubernetes gpu nvidia machine-learning resource-optimization cost-optimization mlops
article_header:
  type: overlay
  theme: dark
  background_color: '#7c3aed'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(124, 58, 237, .4), rgba(34, 197, 94, .4))'
---

Our AWS bill was getting ridiculous. $50K/month in GPU costs with clusters sitting mostly idle - classic. After getting some raised eyebrows from finance (and a few pointed questions about "why we're paying for GPUs to watch Netflix"), I had to figure out how to actually use what we were paying for.

<!--more-->

## How We Were Wasting Money

The numbers were embarrassing. Our monitoring showed:

- 40+ GPU nodes across clusters, mostly idle
- Average utilization hovering around 30-40%
- ML engineers complaining about long queues while GPUs sat unused
- Some Jupyter notebooks hogging entire A100s just to run pandas operations

The problem was obvious in hindsight: Kubernetes' all-or-nothing GPU allocation. Request 1 GPU, get a whole $30k A100 to yourself, even if you're just testing a tiny model. It's like renting a mansion to store a single box.

## Understanding GPU Workload Patterns

### Workload Analysis

We instrumented our clusters to understand GPU usage patterns:

```python
import nvidia_ml_py as nv
import time
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class GPUMetrics:
    timestamp: float
    gpu_id: int
    utilization_percent: float
    memory_used_mb: int
    memory_total_mb: int
    temperature_c: int
    power_draw_w: float
    pod_name: str
    namespace: str

class GPUMonitor:
    def __init__(self):
        nv.nvmlInit()
        self.device_count = nv.nvmlDeviceGetCount()
        
    def collect_metrics(self) -> List[GPUMetrics]:
        metrics = []
        
        for i in range(self.device_count):
            handle = nv.nvmlDeviceGetHandleByIndex(i)
            
            # GPU utilization
            utilization = nv.nvmlDeviceGetUtilizationRates(handle)
            
            # Memory info
            memory_info = nv.nvmlDeviceGetMemoryInfo(handle)
            
            # Temperature
            temp = nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)
            
            # Power draw
            power = nv.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
            
            # Get pod information from Kubernetes API
            pod_info = self.get_pod_using_gpu(i)
            
            metrics.append(GPUMetrics(
                timestamp=time.time(),
                gpu_id=i,
                utilization_percent=utilization.gpu,
                memory_used_mb=memory_info.used // 1024**2,
                memory_total_mb=memory_info.total // 1024**2,
                temperature_c=temp,
                power_draw_w=power,
                pod_name=pod_info.get('name', 'unknown'),
                namespace=pod_info.get('namespace', 'unknown')
            ))
            
        return metrics
```

### Key Findings

Our analysis revealed distinct workload patterns:

1. **Training Jobs**: High GPU utilization (80-95%) but bursty memory usage
2. **Inference Serving**: Consistent low utilization (10-30%) with fast response requirements  
3. **Jupyter Notebooks**: Extremely low utilization (5-15%) with long idle periods
4. **Batch Processing**: Variable utilization patterns depending on data pipeline stage

## Multi-Level GPU Sharing Strategy

### 1. MPS (Multi-Process Service) for Inference

NVIDIA MPS allows multiple processes to share a single GPU context:

```yaml
# MPS DaemonSet for inference nodes
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-mps
  namespace: gpu-sharing
spec:
  selector:
    matchLabels:
      name: nvidia-mps
  template:
    metadata:
      labels:
        name: nvidia-mps
    spec:
      nodeSelector:
        gpu-sharing: "inference"
      hostPID: true
      containers:
      - name: nvidia-mps
        image: nvidia/cuda:11.8-runtime-ubuntu20.04
        command:
        - /bin/bash
        - -c
        - |
          # Start MPS daemon
          export CUDA_VISIBLE_DEVICES=0
          nvidia-cuda-mps-control -d
          
          # Configure memory limits per client
          echo "set_default_active_thread_percentage 50" | nvidia-cuda-mps-control
          echo "set_default_memory_limit 25%" | nvidia-cuda-mps-control
          
          # Keep daemon running
          tail -f /dev/null
        securityContext:
          privileged: true
        volumeMounts:
        - name: nvidia-mps-socket
          mountPath: /tmp/nvidia-mps
        - name: proc
          mountPath: /host/proc
      volumes:
      - name: nvidia-mps-socket
        hostPath:
          path: /tmp/nvidia-mps
      - name: proc
        hostPath:
          path: /proc
```

### 2. Virtual GPU Slicing

We implemented GPU memory and compute slicing using NVIDIA's vGPU technology:

```python
class GPUScheduler:
    def __init__(self):
        self.gpu_inventory = self.discover_gpu_nodes()
        self.workload_classifier = WorkloadClassifier()
        
    def schedule_pod(self, pod_spec: Dict) -> str:
        """Intelligent GPU scheduling based on workload type"""
        
        # Classify workload type
        workload_type = self.workload_classifier.classify(pod_spec)
        
        # Get resource requirements
        gpu_memory_req = self.extract_gpu_memory_requirement(pod_spec)
        gpu_compute_req = self.extract_gpu_compute_requirement(pod_spec)
        
        if workload_type == "inference":
            return self.schedule_inference_workload(gpu_memory_req, gpu_compute_req)
        elif workload_type == "training":
            return self.schedule_training_workload(pod_spec)
        elif workload_type == "notebook":
            return self.schedule_notebook_workload(gpu_memory_req)
        else:
            return self.schedule_batch_workload(pod_spec)
            
    def schedule_inference_workload(self, memory_req: int, compute_req: int) -> str:
        """Schedule inference on shared GPU with MPS"""
        
        for node_id, node_info in self.gpu_inventory.items():
            for gpu_id, gpu_info in node_info['gpus'].items():
                
                # Check if MPS is enabled
                if not gpu_info.get('mps_enabled'):
                    continue
                    
                # Check available memory slice
                available_memory = gpu_info['total_memory'] - gpu_info['allocated_memory']
                if available_memory >= memory_req:
                    
                    # Check compute availability 
                    available_compute = 100 - gpu_info['allocated_compute_percent']
                    if available_compute >= compute_req:
                        
                        # Reserve resources
                        self.reserve_gpu_slice(node_id, gpu_id, memory_req, compute_req)
                        return f"{node_id}/gpu-{gpu_id}"
                        
        return None  # No suitable node found
```

### 3. Time-Sliced GPU Sharing

For development workloads, we implemented time-based GPU sharing:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gpu-time-slicing-config
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4  # 4 processes can share 1 GPU
          memoryGB: 8  # Each process gets 8GB memory limit
```

## Advanced Scheduling Logic

### Workload Classification

```python
class WorkloadClassifier:
    def __init__(self):
        self.training_patterns = [
            r'.*pytorch.*train.*',
            r'.*tensorflow.*fit.*',
            r'.*horovod.*',
            r'.*distributed.*training.*'
        ]
        
        self.inference_patterns = [
            r'.*serving.*',
            r'.*inference.*',
            r'.*predict.*',
            r'.*api.*server.*'
        ]
        
        self.notebook_patterns = [
            r'.*jupyter.*',
            r'.*notebook.*',
            r'.*lab.*'
        ]
        
    def classify(self, pod_spec: Dict) -> str:
        """Classify workload type from pod specification"""
        
        # Check container images and commands
        containers = pod_spec.get('spec', {}).get('containers', [])
        
        for container in containers:
            image = container.get('image', '').lower()
            command = ' '.join(container.get('command', [])).lower()
            args = ' '.join(container.get('args', [])).lower()
            
            full_spec = f"{image} {command} {args}"
            
            if any(re.match(pattern, full_spec) for pattern in self.training_patterns):
                return 'training'
            elif any(re.match(pattern, full_spec) for pattern in self.inference_patterns):
                return 'inference'
            elif any(re.match(pattern, full_spec) for pattern in self.notebook_patterns):
                return 'notebook'
                
        # Check resource requirements as fallback
        resources = container.get('resources', {})
        requests = resources.get('requests', {})
        
        gpu_request = requests.get('nvidia.com/gpu', 0)
        memory_request = requests.get('memory', '0Gi')
        
        if gpu_request > 1:
            return 'training'  # Multi-GPU typically means training
        elif 'Gi' in memory_request and int(memory_request.split('Gi')[0]) > 32:
            return 'training'  # High memory usually training
        else:
            return 'batch'  # Default to batch processing
```

### Predictive Scheduling

We built a predictive model to anticipate resource needs:

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

class PredictiveGPUScheduler:
    def __init__(self):
        self.utilization_model = RandomForestRegressor(n_estimators=100)
        self.duration_model = RandomForestRegressor(n_estimators=100)
        self.historical_data = self.load_historical_data()
        self.train_models()
        
    def predict_gpu_utilization(self, pod_spec: Dict) -> float:
        """Predict expected GPU utilization for a workload"""
        
        features = self.extract_features(pod_spec)
        predicted_utilization = self.utilization_model.predict([features])[0]
        
        return max(0.1, min(1.0, predicted_utilization))  # Clamp between 10-100%
        
    def predict_job_duration(self, pod_spec: Dict) -> int:
        """Predict how long the job will run (in minutes)"""
        
        features = self.extract_features(pod_spec)
        predicted_duration = self.duration_model.predict([features])[0]
        
        return max(5, int(predicted_duration))  # At least 5 minutes
        
    def extract_features(self, pod_spec: Dict) -> List[float]:
        """Extract features for ML prediction"""
        
        features = []
        
        # Container image features
        image = pod_spec.get('spec', {}).get('containers', [{}])[0].get('image', '')
        features.append(1 if 'pytorch' in image else 0)
        features.append(1 if 'tensorflow' in image else 0)
        features.append(1 if 'jupyter' in image else 0)
        
        # Resource request features
        resources = pod_spec.get('spec', {}).get('containers', [{}])[0].get('resources', {})
        requests = resources.get('requests', {})
        
        # GPU count
        gpu_count = float(requests.get('nvidia.com/gpu', 0))
        features.append(gpu_count)
        
        # Memory request (in GB)
        memory_str = requests.get('memory', '0Gi')
        memory_gb = float(memory_str.replace('Gi', '')) if 'Gi' in memory_str else 0
        features.append(memory_gb)
        
        # CPU request
        cpu_str = requests.get('cpu', '0')
        cpu_cores = float(cpu_str.replace('m', '')) / 1000 if 'm' in cpu_str else float(cpu_str)
        features.append(cpu_cores)
        
        # Time features
        now = datetime.now()
        features.append(now.hour)  # Hour of day
        features.append(now.weekday())  # Day of week
        
        # User/namespace features (simplified)
        namespace = pod_spec.get('metadata', {}).get('namespace', 'default')
        features.append(hash(namespace) % 100)  # Namespace hash as feature
        
        return features
        
    def should_preempt(self, running_pod: Dict, incoming_pod: Dict) -> bool:
        """Decide whether to preempt a running job for incoming job"""
        
        # Get priorities
        running_priority = self.get_pod_priority(running_pod)
        incoming_priority = self.get_pod_priority(incoming_pod)
        
        # Don't preempt higher priority jobs
        if running_priority >= incoming_priority:
            return False
            
        # Predict remaining runtime for running job
        running_remaining = self.predict_remaining_runtime(running_pod)
        
        # Predict total runtime for incoming job
        incoming_duration = self.predict_job_duration(incoming_pod)
        
        # Preempt if incoming job is much shorter
        if incoming_duration < running_remaining * 0.3:
            return True
            
        return False
```

## Dynamic Resource Allocation

### GPU Memory Management

```python
class GPUMemoryManager:
    def __init__(self):
        self.memory_pools = {}
        
    def allocate_memory(self, node_id: str, gpu_id: int, 
                       memory_mb: int, pod_name: str) -> bool:
        """Allocate GPU memory slice to pod"""
        
        gpu_key = f"{node_id}/gpu-{gpu_id}"
        
        if gpu_key not in self.memory_pools:
            # Initialize memory pool for this GPU
            self.memory_pools[gpu_key] = {
                'total_memory_mb': self.get_gpu_total_memory(node_id, gpu_id),
                'allocations': {},
                'fragmentation_score': 0.0
            }
            
        pool = self.memory_pools[gpu_key]
        allocated_memory = sum(pool['allocations'].values())
        available_memory = pool['total_memory_mb'] - allocated_memory
        
        if available_memory >= memory_mb:
            # Allocate memory
            pool['allocations'][pod_name] = memory_mb
            
            # Update fragmentation score
            pool['fragmentation_score'] = self.calculate_fragmentation(pool)
            
            # Set CUDA memory limit for the pod
            self.set_cuda_memory_limit(node_id, gpu_id, pod_name, memory_mb)
            
            return True
            
        return False
        
    def set_cuda_memory_limit(self, node_id: str, gpu_id: int, 
                             pod_name: str, memory_mb: int):
        """Set CUDA memory limit for specific pod"""
        
        # Create memory limit enforcement script
        script = f"""
        import os
        import torch
        
        # Set memory fraction for this process
        memory_fraction = {memory_mb} / {self.get_gpu_total_memory(node_id, gpu_id)}
        torch.cuda.set_per_process_memory_fraction(memory_fraction, {gpu_id})
        
        # Also set environment variable for other libraries
        os.environ['CUDA_MEMORY_LIMIT'] = str({memory_mb}M)'
        """
        
        # Inject script into pod's init container
        self.inject_memory_limit_script(pod_name, script)
```

### Automatic Scaling

```python
class GPUAutoScaler:
    def __init__(self):
        self.scaling_metrics = {
            'queue_length': 0,
            'avg_wait_time': 0,
            'gpu_utilization': 0,
            'cost_per_hour': 0
        }
        
    def should_scale_up(self) -> bool:
        """Determine if we should add more GPU nodes"""
        
        # Check queue length
        if self.scaling_metrics['queue_length'] > 10:
            return True
            
        # Check average wait time
        if self.scaling_metrics['avg_wait_time'] > 300:  # 5 minutes
            return True
            
        # Check if all GPUs are highly utilized
        if self.scaling_metrics['gpu_utilization'] > 0.8:
            return True
            
        return False
        
    def should_scale_down(self) -> bool:
        """Determine if we should remove GPU nodes"""
        
        # Don't scale down if there's any queue
        if self.scaling_metrics['queue_length'] > 0:
            return False
            
        # Scale down if utilization is low for sustained period
        if self.scaling_metrics['gpu_utilization'] < 0.3:
            return True
            
        return False
        
    def select_instance_type(self, workload_profile: Dict) -> str:
        """Select optimal instance type for current workload"""
        
        instance_options = {
            'p3.2xlarge': {'gpus': 1, 'gpu_memory': 16, 'cost_per_hour': 3.06},
            'p3.8xlarge': {'gpus': 4, 'gpu_memory': 64, 'cost_per_hour': 12.24},
            'p4d.24xlarge': {'gpus': 8, 'gpu_memory': 320, 'cost_per_hour': 32.77},
            'g4dn.xlarge': {'gpus': 1, 'gpu_memory': 16, 'cost_per_hour': 0.526}
        }
        
        # Calculate efficiency score for each instance type
        best_instance = None
        best_score = 0
        
        for instance_type, specs in instance_options.items():
            # Calculate cost per GPU-hour
            cost_per_gpu_hour = specs['cost_per_hour'] / specs['gpus']
            
            # Calculate memory per dollar
            memory_per_dollar = specs['gpu_memory'] / specs['cost_per_hour']
            
            # Workload-specific scoring
            if workload_profile['type'] == 'inference':
                # For inference, prioritize cost efficiency
                score = memory_per_dollar / cost_per_gpu_hour
            elif workload_profile['type'] == 'training':
                # For training, prioritize raw performance
                score = specs['gpus'] * specs['gpu_memory'] / cost_per_gpu_hour
            else:
                # For general workloads, balance cost and performance
                score = specs['gpu_memory'] / cost_per_gpu_hour
                
            if score > best_score:
                best_score = score
                best_instance = instance_type
                
        return best_instance
```

## The Results (Finally)

After 3 months of implementation and debugging:

- Cut GPU costs from $50K to $30K monthly (finance was happy)
- Average utilization jumped to 75-80%
- Queue wait times dropped from 2-3 hours to 15-20 minutes
- ML engineers stopped complaining (mostly)

**What worked best by workload:**
- **Training**: Multi-GPU jobs got better placement, avoiding resource contention
- **Inference**: Went from 1 service per GPU to 4-6 services per GPU  
- **Notebooks**: Time-slicing meant data scientists could actually get resources for quick experiments
- **Batch jobs**: Better bin-packing meant higher overall throughput

## Advanced Monitoring

### Custom Prometheus Metrics

```python
from prometheus_client import Gauge, Histogram, Counter

# GPU utilization by workload type
gpu_utilization_by_workload = Gauge(
    'gpu_utilization_by_workload_type',
    'GPU utilization percentage by workload type',
    ['node', 'gpu_id', 'workload_type']
)

# GPU memory fragmentation
gpu_memory_fragmentation = Gauge(
    'gpu_memory_fragmentation_score',
    'GPU memory fragmentation score (0-1)',
    ['node', 'gpu_id']
)

# Job queue metrics
gpu_queue_length = Gauge(
    'gpu_job_queue_length',
    'Number of jobs waiting for GPU resources',
    ['workload_type', 'priority']
)

# Cost efficiency metrics
cost_per_gpu_hour = Gauge(
    'cost_per_gpu_hour_by_workload',
    'Cost per GPU hour by workload type',
    ['workload_type', 'instance_type']
)

class GPUMetricsExporter:
    def __init__(self):
        self.gpu_monitor = GPUMonitor()
        
    def update_metrics(self):
        """Update all GPU-related metrics"""
        
        metrics = self.gpu_monitor.collect_metrics()
        
        for metric in metrics:
            # Update utilization gauge
            workload_type = self.classify_workload_from_pod(metric.pod_name)
            gpu_utilization_by_workload.labels(
                node=metric.node_id,
                gpu_id=metric.gpu_id,
                workload_type=workload_type
            ).set(metric.utilization_percent)
            
            # Update memory fragmentation
            fragmentation = self.calculate_memory_fragmentation(
                metric.node_id, metric.gpu_id
            )
            gpu_memory_fragmentation.labels(
                node=metric.node_id,
                gpu_id=metric.gpu_id
            ).set(fragmentation)
```

## Hard-Learned Lessons

**Different workloads need different strategies**: Trying to apply one scheduling approach to everything was a mistake. Training jobs need predictable access, inference needs low latency, notebooks need flexibility.

**Data beats intuition**: I thought I knew our workload patterns. I was wrong. Actually measuring GPU usage revealed surprising insights about when and how teams used resources.

**Make costs visible**: Once we added GPU cost attribution to our internal dashboards, teams suddenly cared about efficiency. Amazing how budget visibility changes behavior.

**Change gradually**: I initially wanted to flip everything to GPU sharing at once. Bad idea. Rolling out incrementally by team and workload type prevented disasters.

**Keep developers happy**: The best optimization in the world is useless if it frustrates your users. Every change had to improve both cost AND user experience.

## Future Enhancements

- **Cross-cluster scheduling**: Load balancing across multiple Kubernetes clusters
- **Spot instance optimization**: Intelligent placement of fault-tolerant workloads on spot instances  
- **Multi-tenant GPU isolation**: Hardware-level isolation for security-sensitive workloads
- **AI-driven scheduling**: Deep reinforcement learning for optimal resource allocation

Looking back, this project was equal parts technical challenge and organizational change management. The tech was complicated, but getting teams to change their resource usage habits was harder.

The biggest win wasn't just the cost savings - it was seeing ML engineers actually able to iterate faster because they could get GPU access when they needed it. That's the real value of infrastructure optimization: removing friction so smart people can focus on solving real problems instead of fighting with resource constraints.