---
title: Distributed ML Training at Scale - Building a Multi-GPU Kubernetes Platform
tags: kubernetes distributed-training pytorch horovod machine-learning gpu-optimization mlops
article_header:
  type: overlay
  theme: dark
  background_color: '#1e40af'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(30, 64, 175, .4), rgba(248, 113, 113, .4))'
---

Our ML training jobs were getting ridiculous. 2-week training runs on single GPUs, models that couldn't fit in memory, and constant hardware failures wiping out days of progress. I knew we needed distributed training, but getting it right in production was way harder than the tutorials made it look.

<!--more-->

## The Problem Was Getting Real

Our training pipeline was broken in obvious ways:
- ResNet training: 2 weeks on a single V100 (seriously?)
- BERT models literally wouldn't fit in GPU memory
- GPUs sitting at 20% utilization because data loading was the bottleneck
- Hardware failures meant starting from scratch and losing days of work
- Paying for powerful instances that spent most time idle

We needed to go from single-GPU experiments to serious multi-node training without breaking the bank or driving our ML engineers insane.

## Understanding Distributed Training Patterns

### Data Parallelism vs Model Parallelism

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import horovod.torch as hvd

class DistributedTrainingManager:
    def __init__(self, strategy: str = "data_parallel"):
        self.strategy = strategy
        self.world_size = None
        self.rank = None
        self.local_rank = None
        
    def initialize_distributed(self, backend: str = "nccl"):
        """Initialize distributed training environment"""
        
        if self.strategy == "horovod":
            # Horovod initialization
            hvd.init()
            self.world_size = hvd.size()
            self.rank = hvd.rank()
            self.local_rank = hvd.local_rank()
            
            # Set CUDA device
            torch.cuda.set_device(self.local_rank)
            
        else:
            # PyTorch native distributed
            dist.init_process_group(
                backend=backend,
                init_method='env://',  # Use environment variables
                world_size=int(os.environ['WORLD_SIZE']),
                rank=int(os.environ['RANK'])
            )
            
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ['LOCAL_RANK'])
            
            torch.cuda.set_device(self.local_rank)
            
    def setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for distributed training"""
        
        # Move model to GPU
        model = model.cuda(self.local_rank)
        
        if self.strategy == "horovod":
            # Horovod doesn't need DDP wrapper
            return model
        else:
            # Wrap with DistributedDataParallel
            model = DDP(model, device_ids=[self.local_rank])
            return model
            
    def setup_optimizer(self, model: nn.Module, base_lr: float, optimizer_class=torch.optim.SGD):
        """Setup optimizer for distributed training"""
        
        # Scale learning rate by world size
        scaled_lr = base_lr * self.world_size
        
        optimizer = optimizer_class(model.parameters(), lr=scaled_lr)
        
        if self.strategy == "horovod":
            # Wrap optimizer with Horovod
            optimizer = hvd.DistributedOptimizer(
                optimizer, 
                named_parameters=model.named_parameters(),
                compression=hvd.Compression.fp16,  # Optional compression
                op=hvd.Average
            )
            
            # Broadcast initial parameters
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
            
        return optimizer
        
    def setup_dataloader(self, dataset, batch_size: int, **kwargs) -> DataLoader:
        """Setup distributed data loader"""
        
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=kwargs.get('shuffle', True)
        )
        
        # Calculate per-GPU batch size
        per_gpu_batch_size = batch_size // self.world_size
        
        dataloader = DataLoader(
            dataset,
            batch_size=per_gpu_batch_size,
            sampler=sampler,
            num_workers=kwargs.get('num_workers', 4),
            pin_memory=kwargs.get('pin_memory', True)
        )
        
        return dataloader, sampler
```

### Advanced Communication Patterns

```python
class AdvancedDistributedTraining:
    def __init__(self):
        self.gradient_compression = True
        self.gradient_accumulation_steps = 1
        self.mixed_precision = True
        
    def setup_gradient_compression(self, model: nn.Module):
        """Setup gradient compression for bandwidth efficiency"""
        
        if self.strategy == "horovod":
            # Use built-in Horovod compression
            return hvd.DistributedOptimizer(
                optimizer, 
                named_parameters=model.named_parameters(),
                compression=hvd.Compression.fp16,
                op=hvd.Average
            )
        else:
            # Custom gradient compression for PyTorch DDP
            class CompressedDDP(DDP):
                def reduce_gradients(self):
                    # Compress gradients before allreduce
                    compressed_grads = self.compress_gradients()
                    # Perform allreduce on compressed gradients
                    super().reduce_gradients()
                    
            return CompressedDDP(model, device_ids=[self.local_rank])
            
    def gradient_accumulation_training_step(self, model, optimizer, dataloader, criterion):
        """Training step with gradient accumulation"""
        
        model.train()
        accumulation_loss = 0
        
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # Scale loss by accumulation steps
                loss = loss / self.gradient_accumulation_steps
                
            # Backward pass
            if self.mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            accumulation_loss += loss.item()
            
            # Update parameters every N steps
            if (i + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                optimizer.zero_grad()
                
                # Synchronize metrics across ranks
                if self.rank == 0:
                    avg_loss = accumulation_loss / self.gradient_accumulation_steps
                    print(f"Step {i//self.gradient_accumulation_steps}: Loss = {avg_loss:.4f}")
                    
                accumulation_loss = 0
```

## Kubernetes Operator for Distributed Training

### PyTorchJob Custom Resource Definition

```yaml
# pytorch-operator-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: pytorchjobs.kubeflow.org
spec:
  group: kubeflow.org
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              pytorchReplicaSpecs:
                type: object
                properties:
                  Master:
                    type: object
                    properties:
                      replicas:
                        type: integer
                      restartPolicy:
                        type: string
                      template:
                        type: object
                  Worker:
                    type: object
                    properties:
                      replicas:
                        type: integer
                      restartPolicy:
                        type: string
                      template:
                        type: object
  scope: Namespaced
  names:
    plural: pytorchjobs
    singular: pytorchjob
    kind: PyTorchJob
```

### Distributed Training Job Configuration

```yaml
# distributed-training-job.yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: resnet50-distributed-training
  namespace: ml-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        metadata:
          labels:
            app: pytorch-training
            role: master
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
            command:
            - python
            - -m
            - torch.distributed.launch
            - --nproc_per_node=4
            - --nnodes=3
            - --node_rank=0
            - --master_addr=resnet50-distributed-training-master-0
            - --master_port=23456
            - train_distributed.py
            - --data=/data
            - --epochs=90
            - --batch-size=256
            - --lr=0.1
            resources:
              requests:
                memory: "32Gi"
                nvidia.com/gpu: 4
              limits:
                memory: "64Gi"
                nvidia.com/gpu: 4
            volumeMounts:
            - name: training-data
              mountPath: /data
            - name: model-output
              mountPath: /output
            - name: shared-memory
              mountPath: /dev/shm
            env:
            - name: NCCL_DEBUG
              value: "INFO"
            - name: NCCL_SOCKET_IFNAME
              value: "eth0"
            - name: NCCL_IB_DISABLE
              value: "1"
          volumes:
          - name: training-data
            persistentVolumeClaim:
              claimName: imagenet-dataset
          - name: model-output
            persistentVolumeClaim:
              claimName: model-artifacts
          - name: shared-memory
            emptyDir:
              medium: Memory
              sizeLimit: 8Gi
          nodeSelector:
            accelerator: nvidia-tesla-v100
            
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        metadata:
          labels:
            app: pytorch-training
            role: worker
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
            command:
            - python
            - -m
            - torch.distributed.launch
            - --nproc_per_node=4
            - --nnodes=3
            - --node_rank=${POD_INDEX}
            - --master_addr=resnet50-distributed-training-master-0
            - --master_port=23456
            - train_distributed.py
            - --data=/data
            - --epochs=90
            - --batch-size=256
            - --lr=0.1
            resources:
              requests:
                memory: "32Gi"
                nvidia.com/gpu: 4
              limits:
                memory: "64Gi"
                nvidia.com/gpu: 4
            volumeMounts:
            - name: training-data
              mountPath: /data
            - name: model-output
              mountPath: /output
            - name: shared-memory
              mountPath: /dev/shm
            env:
            - name: NCCL_DEBUG
              value: "INFO"
            - name: NCCL_SOCKET_IFNAME
              value: "eth0"
            - name: NCCL_IB_DISABLE
              value: "1"
            - name: POD_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.labels['pytorch.kubeflow.org/replica-index']
          volumes:
          - name: training-data
            persistentVolumeClaim:
              claimName: imagenet-dataset
          - name: model-output
            persistentVolumeClaim:
              claimName: model-artifacts
          - name: shared-memory
            emptyDir:
              medium: Memory
              sizeLimit: 8Gi
          nodeSelector:
            accelerator: nvidia-tesla-v100
```

## Intelligent Job Scheduling

### Custom Scheduler for ML Workloads

```python
from kubernetes import client, config, watch
import threading
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class GPUNode:
    name: str
    total_gpus: int
    available_gpus: int
    gpu_memory_total: int
    gpu_memory_available: int
    network_bandwidth: int
    interconnect_type: str  # nvlink, infiniband, ethernet

class MLJobScheduler:
    def __init__(self):
        config.load_incluster_config()
        self.v1 = client.CoreV1Api()
        self.custom_api = client.CustomObjectsApi()
        
        self.gpu_nodes = {}
        self.pending_jobs = []
        self.running_jobs = {}
        
        # Start monitoring threads
        self.node_monitor_thread = threading.Thread(target=self.monitor_nodes)
        self.job_scheduler_thread = threading.Thread(target=self.schedule_jobs)
        
        self.node_monitor_thread.start()
        self.job_scheduler_thread.start()
        
    def monitor_nodes(self):
        """Monitor GPU node availability and resources"""
        
        while True:
            try:
                # Get all nodes with GPUs
                nodes = self.v1.list_node(label_selector="accelerator")
                
                for node in nodes.items:
                    node_name = node.metadata.name
                    
                    # Extract GPU information
                    gpu_info = self.extract_gpu_info(node)
                    
                    # Update node inventory
                    self.gpu_nodes[node_name] = GPUNode(
                        name=node_name,
                        total_gpus=gpu_info['total_gpus'],
                        available_gpus=gpu_info['available_gpus'],
                        gpu_memory_total=gpu_info['memory_total'],
                        gpu_memory_available=gpu_info['memory_available'],
                        network_bandwidth=gpu_info['network_bandwidth'],
                        interconnect_type=gpu_info['interconnect_type']
                    )
                    
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                print(f"Error monitoring nodes: {e}")
                time.sleep(60)
                
    def schedule_jobs(self):
        """Intelligent job scheduling algorithm"""
        
        while True:
            try:
                if self.pending_jobs:
                    job = self.pending_jobs[0]
                    
                    # Find optimal node placement
                    placement = self.find_optimal_placement(job)
                    
                    if placement:
                        # Schedule job
                        self.schedule_job_on_nodes(job, placement)
                        self.pending_jobs.remove(job)
                        self.running_jobs[job['name']] = placement
                        
                time.sleep(10)  # Schedule every 10 seconds
                
            except Exception as e:
                print(f"Error scheduling jobs: {e}")
                time.sleep(30)
                
    def find_optimal_placement(self, job: Dict) -> Optional[Dict]:
        """Find optimal node placement for distributed job"""
        
        required_gpus = job['spec']['total_gpus']
        gpu_memory_req = job['spec'].get('gpu_memory_mb', 16000)  # Default 16GB
        communication_intensive = job['spec'].get('communication_intensive', True)
        
        # Strategy 1: Single node if possible (best communication)
        for node_name, node in self.gpu_nodes.items():
            if (node.available_gpus >= required_gpus and
                node.gpu_memory_available >= gpu_memory_req * required_gpus):
                
                return {
                    'strategy': 'single_node',
                    'nodes': [{'name': node_name, 'gpus': required_gpus}],
                    'total_nodes': 1,
                    'communication_cost': 0  # Intra-node communication is fastest
                }
                
        # Strategy 2: Multi-node with optimal communication
        if communication_intensive:
            # Prefer nodes with high-speed interconnect
            high_speed_nodes = [
                node for node in self.gpu_nodes.values()
                if node.interconnect_type in ['nvlink', 'infiniband']
            ]
            
            placement = self.pack_gpus_optimally(required_gpus, high_speed_nodes, gpu_memory_req)
            if placement:
                return placement
                
        # Strategy 3: Best fit across available nodes
        return self.pack_gpus_optimally(required_gpus, list(self.gpu_nodes.values()), gpu_memory_req)
        
    def pack_gpus_optimally(self, required_gpus: int, available_nodes: List[GPUNode], 
                          memory_per_gpu: int) -> Optional[Dict]:
        """Pack GPUs across nodes optimally"""
        
        # Sort nodes by available GPUs (descending)
        available_nodes.sort(key=lambda x: x.available_gpus, reverse=True)
        
        placement_nodes = []
        remaining_gpus = required_gpus
        
        for node in available_nodes:
            if remaining_gpus <= 0:
                break
                
            # Check memory constraint
            max_gpus_by_memory = node.gpu_memory_available // memory_per_gpu
            usable_gpus = min(node.available_gpus, max_gpus_by_memory, remaining_gpus)
            
            if usable_gpus > 0:
                placement_nodes.append({
                    'name': node.name,
                    'gpus': usable_gpus
                })
                remaining_gpus -= usable_gpus
                
        if remaining_gpus > 0:
            return None  # Cannot satisfy resource requirements
            
        # Calculate communication cost
        comm_cost = self.calculate_communication_cost(placement_nodes)
        
        return {
            'strategy': 'multi_node',
            'nodes': placement_nodes,
            'total_nodes': len(placement_nodes),
            'communication_cost': comm_cost
        }
        
    def calculate_communication_cost(self, placement_nodes: List[Dict]) -> float:
        """Calculate estimated communication cost for node placement"""
        
        if len(placement_nodes) == 1:
            return 0.0
            
        # Simple model: cost increases with number of nodes and decreases with bandwidth
        total_cost = 0.0
        
        for i, node1 in enumerate(placement_nodes):
            for j, node2 in enumerate(placement_nodes[i+1:], i+1):
                node1_info = self.gpu_nodes[node1['name']]
                node2_info = self.gpu_nodes[node2['name']]
                
                # Inter-node communication cost
                bandwidth = min(node1_info.network_bandwidth, node2_info.network_bandwidth)
                cost = (node1['gpus'] * node2['gpus']) / bandwidth
                total_cost += cost
                
        return total_cost
```

## Fault Tolerance and Checkpointing

### Automatic Checkpoint Management

```python
import torch
import os
import time
from typing import Dict, Any
import boto3

class DistributedCheckpointManager:
    def __init__(self, checkpoint_dir: str, s3_bucket: str = None):
        self.checkpoint_dir = checkpoint_dir
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3') if s3_bucket else None
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, step: int, metrics: Dict[str, float],
                       rank: int = 0) -> str:
        """Save distributed training checkpoint"""
        
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        # Save locally first
        checkpoint_filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
        local_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        
        if rank == 0:  # Only master saves checkpoints
            torch.save(checkpoint_data, local_path)
            print(f"Checkpoint saved: {local_path}")
            
            # Upload to S3 for persistence
            if self.s3_client:
                s3_key = f"checkpoints/{checkpoint_filename}"
                self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                print(f"Checkpoint uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
                
        # Synchronize across all ranks
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            
        return local_path
        
    def load_checkpoint(self, checkpoint_path: str = None) -> Dict[str, Any]:
        """Load latest or specified checkpoint"""
        
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_path = self.find_latest_checkpoint()
            
        if not checkpoint_path:
            print("No checkpoint found")
            return None
            
        # Download from S3 if not available locally
        if not os.path.exists(checkpoint_path) and self.s3_client:
            self.download_from_s3(checkpoint_path)
            
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"Checkpoint loaded from: {checkpoint_path}")
            return checkpoint
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
            
    def find_latest_checkpoint(self) -> str:
        """Find the latest checkpoint file"""
        
        checkpoint_files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('checkpoint_') and f.endswith('.pt')
        ]
        
        if not checkpoint_files:
            return None
            
        # Sort by modification time
        checkpoint_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)),
            reverse=True
        )
        
        return os.path.join(self.checkpoint_dir, checkpoint_files[0])
        
    def auto_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, step: int, metrics: Dict[str, float],
                       checkpoint_interval: int = 1000, rank: int = 0):
        """Automatically save checkpoints at specified intervals"""
        
        if step % checkpoint_interval == 0:
            self.save_checkpoint(model, optimizer, epoch, step, metrics, rank)
            
            # Clean up old checkpoints (keep only last 3)
            self.cleanup_old_checkpoints(keep_last=3)
            
    def cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoint files to save disk space"""
        
        checkpoint_files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('checkpoint_') and f.endswith('.pt')
        ]
        
        if len(checkpoint_files) > keep_last:
            # Sort by modification time
            checkpoint_files.sort(
                key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x))
            )
            
            # Remove oldest files
            for filename in checkpoint_files[:-keep_last]:
                file_path = os.path.join(self.checkpoint_dir, filename)
                os.remove(file_path)
                print(f"Removed old checkpoint: {filename}")
```

### Failure Recovery and Preemption Handling

```python
class FaultTolerantTrainer:
    def __init__(self, model, optimizer, dataloader, checkpoint_manager):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.checkpoint_manager = checkpoint_manager
        
        # Recovery state
        self.start_epoch = 0
        self.start_step = 0
        self.best_metric = 0.0
        
        # Preemption handling
        self.preemption_signal_received = False
        signal.signal(signal.SIGTERM, self.handle_preemption)
        signal.signal(signal.SIGINT, self.handle_preemption)
        
    def handle_preemption(self, signum, frame):
        """Handle preemption gracefully"""
        print(f"Received signal {signum}, preparing for graceful shutdown...")
        self.preemption_signal_received = True
        
    def recover_from_checkpoint(self):
        """Recover training state from checkpoint"""
        
        checkpoint = self.checkpoint_manager.load_checkpoint()
        
        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.start_step = checkpoint['step']
            self.best_metric = checkpoint['metrics'].get('best_accuracy', 0.0)
            
            print(f"Recovered from epoch {self.start_epoch}, step {self.start_step}")
            return True
            
        return False
        
    def train_with_fault_tolerance(self, num_epochs: int):
        """Main training loop with fault tolerance"""
        
        # Attempt recovery
        self.recover_from_checkpoint()
        
        for epoch in range(self.start_epoch, num_epochs):
            if self.preemption_signal_received:
                print("Preemption detected, saving checkpoint and exiting...")
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, self.start_step, 
                    {'best_accuracy': self.best_metric}
                )
                break
                
            self.train_epoch(epoch)
            
            # Validate and checkpoint
            accuracy = self.validate_epoch(epoch)
            
            if accuracy > self.best_metric:
                self.best_metric = accuracy
                # Save best model checkpoint
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, self.start_step,
                    {'best_accuracy': accuracy, 'is_best': True}
                )
                
    def train_epoch(self, epoch: int):
        """Train single epoch with checkpointing"""
        
        self.model.train()
        running_loss = 0.0
        
        for step, (inputs, targets) in enumerate(self.dataloader):
            global_step = epoch * len(self.dataloader) + step
            
            # Check for preemption
            if self.preemption_signal_received:
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, global_step,
                    {'running_loss': running_loss / (step + 1)}
                )
                return
                
            # Training step
            inputs, targets = inputs.cuda(), targets.cuda()
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Auto-checkpoint
            self.checkpoint_manager.auto_checkpoint(
                self.model, self.optimizer, epoch, global_step,
                {'running_loss': running_loss / (step + 1)}
            )
            
            # Progress reporting
            if step % 100 == 0 and torch.distributed.get_rank() == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
```

## Performance Optimization and Monitoring

### Custom Metrics and Monitoring

```python
from prometheus_client import Gauge, Counter, Histogram
import psutil
import nvidia_ml_py as nvml
import torch

class DistributedTrainingMonitor:
    def __init__(self):
        # Initialize NVIDIA ML
        nvml.nvmlInit()
        
        # Prometheus metrics
        self.training_loss = Gauge('training_loss', 'Training loss', ['job_name', 'rank'])
        self.gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization', ['node', 'gpu_id'])
        self.gpu_memory_used = Gauge('gpu_memory_used_mb', 'GPU memory used', ['node', 'gpu_id'])
        self.communication_time = Histogram('communication_time_seconds', 'Time spent in communication', ['operation'])
        self.throughput = Gauge('training_throughput_samples_per_sec', 'Training throughput', ['job_name'])
        
        # Distributed training specific metrics
        self.gradient_sync_time = Histogram('gradient_sync_time_seconds', 'Gradient synchronization time')
        self.data_loading_time = Histogram('data_loading_time_seconds', 'Data loading time per batch')
        self.computation_time = Histogram('computation_time_seconds', 'Forward/backward computation time')
        
    def monitor_gpu_usage(self):
        """Monitor GPU utilization and memory"""
        
        device_count = nvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            
            # GPU utilization
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            self.gpu_utilization.labels(node=socket.gethostname(), gpu_id=i).set(utilization.gpu)
            
            # GPU memory
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_mb = memory.used // (1024 ** 2)
            self.gpu_memory_used.labels(node=socket.gethostname(), gpu_id=i).set(memory_used_mb)
            
    def time_communication(self, operation: str):
        """Context manager to time communication operations"""
        
        class CommTimer:
            def __init__(self, monitor, op):
                self.monitor = monitor
                self.operation = op
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.monitor.communication_time.labels(operation=self.operation).observe(duration)
                
        return CommTimer(self, operation)
        
    def measure_training_step(self, model, optimizer, inputs, targets, criterion):
        """Measure performance of a single training step"""
        
        step_start_time = time.time()
        
        # Data loading time (already loaded, but can be measured separately)
        data_load_time = 0  # This would be measured in dataloader
        
        # Computation time
        comp_start = time.time()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        comp_time = time.time() - comp_start
        self.computation_time.observe(comp_time)
        
        # Gradient synchronization time
        sync_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        sync_time = time.time() - sync_start
        self.gradient_sync_time.observe(sync_time)
        
        # Total step time
        total_step_time = time.time() - step_start_time
        
        # Calculate throughput (samples per second)
        batch_size = inputs.size(0) * torch.distributed.get_world_size()
        throughput = batch_size / total_step_time
        self.throughput.labels(job_name=os.environ.get('JOB_NAME', 'unknown')).set(throughput)
        
        return {
            'loss': loss.item(),
            'computation_time': comp_time,
            'sync_time': sync_time,
            'total_time': total_step_time,
            'throughput': throughput
        }
```

## What We Actually Achieved

After 6 months of building this platform:
- Training times went from weeks to hours (8-GPU jobs finishing overnight)
- GPU utilization jumped from 30% to 85%+ cluster-wide
- Developers stopped asking me "why is my job stuck in queue?"
- Hardware failures became annoying rather than catastrophic
- Our experimentation rate doubled because people could actually run more models

## Advanced Techniques

### Dynamic Learning Rate Scaling

```python
class DistributedLRScheduler:
    def __init__(self, optimizer, world_size, base_lr):
        self.optimizer = optimizer
        self.world_size = world_size
        self.base_lr = base_lr
        self.current_lr = base_lr
        
    def scale_lr_for_batch_size(self, global_batch_size):
        """Scale learning rate based on total batch size"""
        
        # Linear scaling rule
        scaled_lr = self.base_lr * (global_batch_size / 256)  # Assuming 256 as reference
        
        # Apply square root scaling for large batch sizes
        if global_batch_size > 8192:
            scaled_lr = self.base_lr * math.sqrt(global_batch_size / 256)
            
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = scaled_lr
            
        self.current_lr = scaled_lr
        
    def warmup_schedule(self, epoch, warmup_epochs=5):
        """Learning rate warmup for large batch training"""
        
        if epoch < warmup_epochs:
            # Linear warmup
            warmup_lr = self.base_lr * (epoch + 1) / warmup_epochs
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
                
            self.current_lr = warmup_lr
```

### Communication Optimization

```python
class CommunicationOptimizer:
    def __init__(self, model):
        self.model = model
        self.compression_enabled = True
        
    def enable_gradient_compression(self):
        """Enable gradient compression for bandwidth efficiency"""
        
        class CompressedAllReduce:
            def __init__(self, compression_ratio=0.1):
                self.compression_ratio = compression_ratio
                
            def compress_gradients(self, gradients):
                """Top-k sparsification"""
                compressed_grads = {}
                
                for name, grad in gradients.items():
                    if grad is not None:
                        # Flatten gradient
                        flat_grad = grad.flatten()
                        
                        # Select top-k values
                        k = int(len(flat_grad) * self.compression_ratio)
                        _, indices = torch.topk(flat_grad.abs(), k)
                        
                        # Create sparse representation
                        compressed_grads[name] = {
                            'values': flat_grad[indices],
                            'indices': indices,
                            'shape': grad.shape
                        }
                        
                return compressed_grads
                
            def decompress_gradients(self, compressed_grads):
                """Reconstruct gradients from compressed representation"""
                gradients = {}
                
                for name, compressed in compressed_grads.items():
                    # Create full gradient tensor
                    grad_size = torch.prod(torch.tensor(compressed['shape']))
                    full_grad = torch.zeros(grad_size, device=compressed['values'].device)
                    
                    # Fill in non-zero values
                    full_grad[compressed['indices']] = compressed['values']
                    
                    # Reshape to original shape
                    gradients[name] = full_grad.reshape(compressed['shape'])
                    
                return gradients
                
        return CompressedAllReduce()
```

## What I Wish I'd Known Earlier

**Communication kills performance at scale**: Once you hit 8+ GPUs, network becomes the bottleneck. I spent weeks optimizing model code before realizing the real issue was gradient synchronization.

**Plan for failures from day one**: At scale, something breaks every day. Adding checkpointing and recovery as an afterthought was painful. Should've built it in from the start.

**FP16 is magic (when it works)**: Nearly doubled our training speed with minimal accuracy loss. But gradient scaling can be tricky to get right.

**Dumb round-robin scheduling is terrible**: Our first scheduler just assigned jobs randomly to available nodes. Huge mistake. Communication topology matters enormously.

**Monitor everything or debug nothing**: When a 100-GPU job fails, you need detailed metrics to understand why. Added comprehensive monitoring after too many mysterious failures.

## Future Enhancements

- **Federated Learning**: Extend platform for cross-organization collaborative training
- **Model Parallelism**: Support for models too large for single-GPU memory
- **Elastic Training**: Dynamic scaling of resources during training
- **Multi-Cloud**: Seamless training across multiple cloud providers

Building this distributed training platform taught me that the real challenge isn't making multiple GPUs work together - it's making them work together without driving your users crazy.

The best infrastructure is invisible infrastructure. Our ML engineers now scale from 1 to 100 GPUs without thinking about it, which means they can focus on the hard problems (building better models) instead of the annoying problems (why won't PyTorch find my GPUs?).