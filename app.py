"""
Facial Expression Recognition API
FastAPI application to serve the FER-2013 model
"""

import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import cv2 for face detection
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Face detection will be skipped.")

# Initialize FastAPI app
app = FastAPI(
    title="FER-2013 Facial Expression Recognition API",
    description="API for predicting facial expressions from images",
    version="1.0.0"
)

# Global variables for model and class names
model = None
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Model configuration
# Try SavedModel format first (more compatible), fallback to H5
SAVED_MODEL_PATH = os.getenv('SAVED_MODEL_PATH', 'saved_model')
MODEL_PATH = os.getenv('MODEL_PATH', 'best_model.h5')
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48

def load_model_instance():
    """Load the trained model"""
    global model
    if model is None:
        import tensorflow as tf
        
        # Try H5 format first (most compatible)
        try:
            logger.info(f"Loading model from H5 format: {MODEL_PATH}...")
            model = load_model(MODEL_PATH, compile=False)
            # Compile the model after loading
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            logger.info("Model loaded successfully from H5 format!")
            return model
        except Exception as e:
            logger.warning(f"H5 load failed: {e}, trying SavedModel format...")
        
        # Fallback to SavedModel format
        if os.path.exists(SAVED_MODEL_PATH) and os.path.isdir(SAVED_MODEL_PATH):
            try:
                logger.info(f"Loading model from SavedModel format: {SAVED_MODEL_PATH}")
                # Try loading as Keras SavedModel first
                try:
                    loaded = tf.keras.models.load_model(SAVED_MODEL_PATH)
                    # Check if it's actually a Keras model
                    if hasattr(loaded, 'predict'):
                        model = loaded
                        logger.info("Model loaded successfully from SavedModel format (Keras)!")
                        return model
                    else:
                        raise ValueError("Loaded object is not a Keras model")
                except Exception as keras_error:
                    # If Keras load fails, it's TensorFlow SavedModel format
                    logger.warning(f"Keras SavedModel load failed, trying TensorFlow SavedModel: {keras_error}")
                    # Load as TensorFlow SavedModel and wrap it
                    saved_model = tf.saved_model.load(SAVED_MODEL_PATH)
                    # Create a wrapper to make it work like a Keras model
                    class SavedModelWrapper:
                        def __init__(self, saved_model_obj):
                            self.saved_model = saved_model_obj
                            # Get the serving function
                            if hasattr(saved_model_obj, 'signatures'):
                                self.predict_fn = saved_model_obj.signatures.get('serving_default', 
                                    list(saved_model_obj.signatures.values())[0] if saved_model_obj.signatures else None)
                            else:
                                self.predict_fn = saved_model_obj
                        
                        def predict(self, x, verbose=0):
                            # Convert input to tensor if needed
                            if isinstance(x, np.ndarray):
                                x = tf.constant(x)
                            # Call the serving function
                            if self.predict_fn:
                                result = self.predict_fn(x)
                                # Extract output (may be a dict with 'output_0' or similar)
                                if isinstance(result, dict):
                                    output = list(result.values())[0]
                                else:
                                    output = result
                                return output.numpy() if hasattr(output, 'numpy') else output
                            else:
                                raise ValueError("No serving function found in SavedModel")
                    
                    model = SavedModelWrapper(saved_model)
                    logger.info("Model loaded successfully from SavedModel format (TensorFlow)!")
                    return model
            except Exception as e:
                logger.error(f"SavedModel load also failed: {e}")
                raise
        else:
            logger.error(f"Neither H5 model ({MODEL_PATH}) nor SavedModel ({SAVED_MODEL_PATH}) found!")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return model

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model_instance()

def detect_and_crop_face(image):
    """
    Detect and crop face from image using OpenCV
    Returns cropped face image or original image if face not detected
    """
    if not CV2_AVAILABLE:
        return image
    
    try:
        # Convert PIL image to numpy array (RGB)
        img_array = np.array(image.convert('RGB'))
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Load the face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Get the largest face (most likely the main subject)
            face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = face
            
            # Add padding around the face (20% on each side)
            padding_x = int(w * 0.2)
            padding_y = int(h * 0.2)
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = min(img_array.shape[1] - x, w + 2 * padding_x)
            h = min(img_array.shape[0] - y, h + 2 * padding_y)
            
            # Crop the face region
            cropped = img_array[y:y+h, x:x+w]
            
            # Convert back to PIL Image
            return Image.fromarray(cropped)
        else:
            # No face detected, return original image
            logger.warning("No face detected, using full image")
            return image
            
    except Exception as e:
        logger.warning(f"Face detection failed: {e}, using full image")
        return image

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for model prediction
    - Detect and crop face if possible
    - Convert to grayscale
    - Resize to 48x48
    - Normalize pixel values to [0, 1]
    """
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Detect and crop face first
        image = detect_and_crop_face(image)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 48x48
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Reshape to match model input shape: (1, 48, 48, 1)
        img_array = img_array.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FER-2013 Facial Expression Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST endpoint to predict facial expression from image",
            "/healthz": "GET endpoint for health checks"
        }
    }

@app.get("/healthz")
async def health_check():
    """
    Health check endpoint for Kubernetes liveness and readiness probes
    """
    try:
        # Check if model is loaded
        if model is None:
            load_model_instance()
        
        # Perform a dummy prediction to ensure model is working
        dummy_input = np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "model_loaded": model is not None,
                "message": "Service is ready"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@app.post("/predict")
async def predict_expression(file: UploadFile = File(...)):
    """
    Predict facial expression from uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON response with predicted class and confidence scores
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Load model if not already loaded
        model_instance = load_model_instance()
        
        # Make prediction
        predictions = model_instance.predict(processed_image, verbose=0)
        
        # Get predicted class index and probability
        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[0][predicted_index])
        
        # Get all class probabilities
        class_probabilities = {
            CLASS_NAMES[i]: float(predictions[0][i])
            for i in range(len(CLASS_NAMES))
        }
        
        return JSONResponse(
            status_code=200,
            content={
                "predicted_expression": predicted_class,
                "confidence": round(confidence, 4),
                "all_predictions": class_probabilities
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



