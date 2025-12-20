# Kubernetes Deployment Files

This directory contains Kubernetes YAML files generated using `kubectl` commands.

## Files

- `deployment.yaml` - Basic deployment configuration
- `service.yaml` - ClusterIP service configuration  
- `deployment-with-probes.yaml` - Deployment with health probes configured
- `hpa.yaml` - Horizontal Pod Autoscaler configuration

## Generation Instructions

These files were generated using the following kubectl commands:

### 1. Create Deployment
```bash
kubectl create deployment fer-api \
    --image=yourusername/fer-api:latest \
    --port=8000 \
    --dry-run=client \
    -o yaml > k8s/deployment.yaml
```

### 2. Create Service
```bash
kubectl create service clusterip fer-api \
    --tcp=8000:8000 \
    --dry-run=client \
    -o yaml > k8s/service.yaml
```

### 3. Configure Health Probes
```bash
# Liveness probe
kubectl set probe deployment/fer-api \
    --liveness \
    --get-url=http://:8000/healthz \
    --initial-delay-seconds=40 \
    --period-seconds=10 \
    --timeout-seconds=5 \
    --failure-threshold=3

# Readiness probe
kubectl set probe deployment/fer-api \
    --readiness \
    --get-url=http://:8000/healthz \
    --initial-delay-seconds=10 \
    --period-seconds=5 \
    --timeout-seconds=3 \
    --failure-threshold=3

# Export updated deployment
kubectl get deployment fer-api -o yaml > k8s/deployment-with-probes.yaml
```

### 4. Create HPA
```bash
kubectl autoscale deployment fer-api \
    --min=1 \
    --max=5 \
    --cpu-percent=50

# Export HPA
kubectl get hpa fer-api -o yaml > k8s/hpa.yaml
```







