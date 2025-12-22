# FER-2013 Facial Expression Recognition

A cloud-native deployment of a Facial Expression Recognition model using FastAPI, Docker, and Kubernetes. This project recognizes 7 different emotions from facial images: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## 🎯 Features

- **Machine Learning Model**: Residual CNN trained on FER-2013 dataset
- **REST API**: FastAPI-based API for facial expression predictions
- **Docker Containerization**: Fully containerized application ready for deployment
- **Kubernetes Deployment**: Complete K8s deployment with health checks and autoscaling
- **Health Checks**: Liveness and readiness probes for reliability
- **Horizontal Pod Autoscaler (HPA)**: Automatic scaling based on CPU usage
- **Web Interface**: Streamlit GUI for local testing and development

## 📋 Prerequisites

- Python 3.10+
- Docker Desktop (for containerization)
- Kubernetes cluster (Minikube, Docker Desktop Kubernetes, or cloud cluster)
- kubectl configured
- Docker Hub account (for pushing images)

## 🚀 Quick Start

### Run FastAPI Server Locally

For API access:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
python app.py
```

The API will be available at `http://localhost:8000`

**Test the API:**

```bash
# Health check
curl http://localhost:8000/healthz

# Predict emotion (Windows PowerShell)
curl.exe -X POST http://localhost:8000/predict -F "file=@path/to/image.jpg"

# Predict emotion (Linux/Mac)
curl -X POST http://localhost:8000/predict -F "file=@path/to/image.jpg"
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t yourusername/fer-api:latest .
```

### Run Docker Container

```bash
docker run -p 8000:8000 yourusername/fer-api:latest
```

### Push to Docker Hub

```bash
docker login
docker push yourusername/fer-api:latest
```

**Note**: Update the image name in `k8s/deployment.yaml` with your Docker Hub username.

## ☸️ Kubernetes Deployment

### 1. Prerequisites

Ensure Kubernetes is running:

```bash
# Check cluster connection
kubectl cluster-info

# Check nodes
kubectl get nodes
```

**For Docker Desktop:**

- Go to Settings → Kubernetes → Enable Kubernetes

**For Minikube:**

```bash
minikube start
```

### 2. Deploy to Kubernetes

```bash
# Apply deployment and service
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify deployment
kubectl get deployments
kubectl get services
kubectl get pods
```

### 3. Configure Health Checks

```bash
# Set resource limits (required for HPA)
kubectl set resources deployment fer-api --requests=cpu=200m,memory=512Mi --limits=cpu=1000m,memory=1Gi

# Configure liveness probe
kubectl patch deployment fer-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"fer-api","livenessProbe":{"httpGet":{"path":"/healthz","port":8000},"initialDelaySeconds":40,"periodSeconds":10,"timeoutSeconds":5,"failureThreshold":3}}]}}}}'

# Configure readiness probe
kubectl patch deployment fer-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"fer-api","readinessProbe":{"httpGet":{"path":"/healthz","port":8000},"initialDelaySeconds":10,"periodSeconds":5,"timeoutSeconds":3,"failureThreshold":3}}]}}}}'
```

### 4. Set Up Horizontal Pod Autoscaler (HPA)

```bash
# Enable metrics-server (for Minikube)
minikube addons enable metrics-server

# Create HPA
kubectl autoscale deployment fer-api --min=1 --max=5 --cpu-percent=50

# Check HPA status
kubectl get hpa fer-api
```

### 5. Access the Service

```bash
# Port forward to access locally
kubectl port-forward service/fer-api 8000:8000

# Then access at http://localhost:8000
```

## 📁 Project Structure

```
FER-2013/
├── app.py                  # FastAPI application
├── streamlit_app.py        # Streamlit web interface (local only)
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image configuration
├── .dockerignore          # Docker build exclusions
│
├── best_model.h5          # Trained model (H5 format)
├── saved_model/           # Trained model (SavedModel format)
│
├── k8s/                   # Kubernetes configurations
│   ├── deployment.yaml    # Deployment configuration
│   └── service.yaml       # Service configuration
│
├── train/                 # Training dataset
│   └── [emotion folders]
├── test/                  # Test dataset
│   └── [emotion folders]
└── README.md              # This file
```

## 📡 API Endpoints

### GET `/`

Returns API information and available endpoints.

**Response:**

```json
{
  "message": "FER-2013 Facial Expression Recognition API",
  "version": "1.0.0",
  "endpoints": {
    "/predict": "POST endpoint to predict facial expression from image",
    "/healthz": "GET endpoint for health checks"
  }
}
```

### GET `/healthz`

Health check endpoint for Kubernetes probes.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Service is ready"
}
```

### POST `/predict`

Predict facial expression from an uploaded image.

**Request:** Multipart form-data with `file` field containing an image

**Response:**

```json
{
  "predicted_expression": "happy",
  "confidence": 0.9876,
  "all_predictions": {
    "angry": 0.0012,
    "disgust": 0.0001,
    "fear": 0.0005,
    "happy": 0.9876,
    "neutral": 0.0098,
    "sad": 0.0006,
    "surprise": 0.0002
  }
}
```

## 🎨 Streamlit Interface

The Streamlit app provides:

- **Image Upload**: Upload images from your computer
- **Camera Capture**: Take photos directly from your webcam
- **Real-time Predictions**: See predictions with confidence scores
- **Visualization**: Bar charts showing all emotion probabilities
- **Beautiful UI**: Modern, intuitive interface

**Note**: Streamlit is for local development only and is not containerized or deployed to Kubernetes.
