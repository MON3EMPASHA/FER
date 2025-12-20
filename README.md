# 😊 FER-2013 Facial Expression Recognition

A cloud-native deployment of a Facial Expression Recognition model using FastAPI, Streamlit, Docker, and Kubernetes.

## 🎯 Features

- **REST API**: FastAPI-based API for facial expression predictions
- **Web Interface**: Beautiful Streamlit GUI for interactive predictions
- **Docker**: Containerized application ready for deployment
- **Kubernetes**: Full Kubernetes deployment with health checks and autoscaling
- **7 Emotions**: Recognizes Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise

## 📋 Prerequisites

- Python 3.10+
- Docker Desktop (for local containerization)
- Kubernetes cluster (Minikube, Docker Desktop, or cloud cluster)
- kubectl configured

## 🚀 Quick Start

### Option 1: Run Streamlit App (Easiest)

The simplest way to try the application:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501` and upload an image or use your camera!

### Option 2: Run FastAPI Server

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

## ☸️ Kubernetes Deployment

### 1. Update Docker Image in Deployment

Edit `k8s/deployment.yaml` and replace `mon3empasha/fer-api:latest` with your Docker Hub username:

```yaml
image: yourusername/fer-api:latest
```

### 2. Deploy to Kubernetes

```bash
# Apply deployment and service
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Verify deployment
kubectl get deployments
kubectl get services
kubectl get pods
```

### 3. Configure Health Checks

```bash
# Linux/Mac
kubectl set probe deployment/fer-api \
    --liveness \
    --get-url=http://:8000/healthz \
    --initial-delay-seconds=40 \
    --period-seconds=10

kubectl set probe deployment/fer-api \
    --readiness \
    --get-url=http://:8000/healthz \
    --initial-delay-seconds=10 \
    --period-seconds=5

# Windows PowerShell - see setup_probes.ps1 for script
```

### 4. Access the Service

```bash
# Port forward to access locally
kubectl port-forward service/fer-api 8000:8000

# Then access at http://localhost:8000
```

## 📁 Project Structure

```
FER-2013/
├── app.py                  # FastAPI application
├── streamlit_app.py        # Streamlit web interface
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image configuration
├── .dockerignore          # Docker build exclusions
├── .gitignore             # Git exclusions
│
├── best_model.h5          # Trained model (H5 format)
├── saved_model/           # Trained model (SavedModel format)
│
├── k8s/                   # Kubernetes configurations
│   ├── deployment.yaml    # Deployment configuration
│   ├── service.yaml       # Service configuration
│   └── hpa.yaml           # Horizontal Pod Autoscaler
│
└── README.md              # This file
```

## 📡 API Endpoints

### GET `/`
Returns API information and available endpoints.

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

## 🔧 Configuration

### Environment Variables

- `MODEL_PATH`: Path to H5 model file (default: `best_model.h5`)
- `SAVED_MODEL_PATH`: Path to SavedModel directory (default: `saved_model`)
- `PORT`: Server port (default: `8000` for FastAPI, `8501` for Streamlit)

## 📊 Horizontal Pod Autoscaler (HPA)

The HPA automatically scales pods based on CPU usage:
- **Min replicas**: 1
- **Max replicas**: 5
- **Target CPU**: 50%

```bash
# Check HPA status
kubectl get hpa fer-api

# For HPA to work, ensure metrics-server is installed
minikube addons enable metrics-server
```

## 🐛 Troubleshooting

### Model Not Loading
- Ensure `best_model.h5` or `saved_model/` exists
- Check container logs: `kubectl logs <pod-name>`

### Health Checks Failing
- Increase initial delay if model takes longer to load
- Check if port 8000 is accessible

### Streamlit Issues
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.10+)

## 📝 License

This project is for educational purposes.

## 👤 Author

Cloud Computing Project - FER-2013 Deployment
