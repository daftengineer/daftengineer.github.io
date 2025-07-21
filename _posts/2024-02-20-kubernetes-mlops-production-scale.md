---
title: Scaling MLOps on Kubernetes - Lessons from Managing 8 EKS Clusters Across 4 Regions
tags: kubernetes mlops eks terraform production scaling devops
article_header:
  type: overlay
  theme: dark
  background_color: '#1e3a8a'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(30, 58, 138, .4), rgba(16, 185, 129, .4))'
---

Managing 8 production EKS clusters with live customer tenants across 4 geographical regions taught me that Kubernetes is not just about container orchestration - it's about building resilient, observable, and secure platforms at scale. Here's what I learned from architecting MLOps infrastructure that serves thousands of ML model predictions daily.

<!--more-->

## The Multi-Region Challenge

When we acquired Pathlock's cloud infrastructure, we inherited a complex multi-tenant SaaS platform with strict SOC2 compliance requirements. The challenge was maintaining high availability across:

- **4 geographical regions** (US-East, US-West, EU-West, AP-Southeast)
- **8 EKS clusters** with different configurations
- **Live customer workloads** with zero-downtime requirements
- **ML model serving** with sub-100ms latency SLAs

## Infrastructure as Code: The Foundation

### Terraform + Terragrunt Architecture
```hcl
# terragrunt.hcl structure for multi-region deployments
dependency "vpc" {
  config_path = "../vpc"
}

dependency "security_groups" {
  config_path = "../security-groups"
}

inputs = {
  cluster_name = "ml-prod-${local.region}"
  node_groups = {
    ml_workers = {
      instance_types = ["m5.2xlarge", "m5.4xlarge"]
      gpu_enabled = true
      min_size = 2
      max_size = 50
    }
  }
}
```

### Key Design Decisions

1. **Cluster-per-Region Strategy**: Independent clusters for blast radius containment
2. **GitOps with ArgoCD**: Declarative application deployments across all clusters  
3. **Network Policies**: Kubernetes-native microsegmentation for compliance
4. **Resource Quotas**: Preventing resource exhaustion in multi-tenant environments

## MLOps Pipeline Architecture

### Model Serving with NVIDIA Triton
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference-server
spec:
  template:
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-repository
          mountPath: /models
```

### Auto-scaling Strategy
We implemented multi-layered autoscaling:

1. **Horizontal Pod Autoscaler (HPA)**: Based on custom metrics (inference queue depth)
2. **Vertical Pod Autoscaler (VPA)**: Right-sizing for cost optimization  
3. **Cluster Autoscaler**: Adding nodes during traffic spikes
4. **Predictive Scaling**: ML-driven capacity planning for known traffic patterns

## Observability at Scale

### The Monitoring Stack
- **Prometheus**: Metrics collection with custom ML model performance metrics
- **Grafana**: Dashboards for both infrastructure and ML model performance
- **Loki**: Log aggregation across all clusters and regions
- **OPA (Open Policy Agent)**: Policy enforcement and compliance monitoring

### Custom ML Metrics
```python
# Custom Prometheus metrics for ML models
model_inference_duration = Histogram(
    'ml_model_inference_duration_seconds',
    'Time spent on model inference',
    ['model_name', 'version', 'region']
)

model_accuracy_score = Gauge(
    'ml_model_accuracy_score',
    'Model accuracy on validation set',
    ['model_name', 'version']
)
```

## Security and Compliance

### SOC2 Requirements Implementation
1. **Pod Security Policies**: Preventing privileged containers
2. **Network Policies**: Zero-trust networking between namespaces
3. **RBAC**: Principle of least privilege for all service accounts
4. **Image Scanning**: Automated vulnerability scanning in CI/CD
5. **Audit Logging**: Comprehensive audit trails for all cluster activities

### Secrets Management
```yaml
# External Secrets Operator configuration
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
      auth:
        secretRef:
          accessKeyID:
            name: awssm-secret
            key: access-key
```

## Performance Optimization

### GPU Resource Management
Managing GPU resources efficiently across ML workloads required:

1. **GPU Sharing**: Multiple small models per GPU using NVIDIA MPS
2. **Dynamic Allocation**: GPU nodes that scale to zero when not needed
3. **Workload Prioritization**: Critical models get guaranteed GPU access
4. **Cost Optimization**: Spot instances for training workloads

### Network Performance
- **CNI Optimization**: Calico with BGP routing for cross-AZ traffic
- **Load Balancer Strategy**: Regional NLBs with sticky sessions for stateful ML workloads
- **CDN Integration**: CloudFront for model artifacts and static assets

## Disaster Recovery and High Availability

### Multi-Region Failover Strategy
```yaml
# External DNS for automatic failover
apiVersion: externaldns.alpha.kubernetes.io/v1alpha1
kind: DNSEndpoint
metadata:
  name: ml-api
spec:
  endpoints:
  - dnsName: ml-api.company.com
    recordType: A
    targets:
    - us-west-nlb-123.elb.amazonaws.com
    setIdentifier: us-west
    providerSpecific:
    - name: aws-region
      value: us-west-2
    - name: aws-health-check-id
      value: health-check-123
```

### Backup Strategy
- **etcd snapshots**: Automated daily backups to S3
- **Persistent Volume snapshots**: Application data protection
- **Application-level backups**: ML model checkpoints and training data

## Cost Optimization

Managing costs across 8 clusters required sophisticated approaches:

1. **Right-sizing**: VPA recommendations implemented automatically
2. **Spot Instances**: 70% of training workloads on spot with graceful handling
3. **Cluster Consolidation**: Off-peak workload scheduling
4. **Resource Scheduling**: Time-based scaling for predictable workloads


## Lessons Learned

1. **Observability is non-negotiable**: You can't manage what you can't measure
2. **Security by default**: Implementing security controls early is easier than retrofitting
3. **Automation reduces toil**: Manual processes don't scale across multiple clusters
4. **Cost visibility drives optimization**: Detailed cost attribution changes behavior
5. **Disaster recovery testing**: Regular chaos engineering prevents surprises

## Looking Forward

The future of MLOps on Kubernetes involves:
- **Service Mesh**: Istio for advanced traffic management and security
- **WebAssembly**: Edge inference with WASM modules
- **Serverless ML**: Knative for event-driven model serving
- **AI-driven Operations**: ML models managing ML infrastructure

Managing ML infrastructure at this scale taught me that successful MLOps isn't just about the technology - it's about building systems that empower teams to focus on what matters: delivering value through machine learning.