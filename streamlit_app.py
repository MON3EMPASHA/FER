"""
Facial Expression Recognition - Streamlit GUI
Interactive web interface for FER-2013 model inference
"""

import os
import numpy as np
import streamlit as st
from PIL import Image
import io
import tensorflow as tf
# Use tf.keras for TensorFlow 2.16.1 - it's now properly available
from tensorflow.keras.models import load_model as tf_load_model
import logging
import matplotlib.pyplot as plt
import pandas as pd

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

# Page configuration
st.set_page_config(
    page_title="FER-2013 Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables for model and class names
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_EMOJIS = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

# Model configuration
SAVED_MODEL_PATH = os.getenv('SAVED_MODEL_PATH', 'saved_model')
MODEL_PATH = os.getenv('MODEL_PATH', 'best_model.h5')
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48

@st.cache_resource
def load_model_instance():
    """Load the trained model (cached for performance)"""
    # Try H5 format first (most compatible)
    try:
        logger.info(f"Loading model from H5 format: {MODEL_PATH}...")
        model = tf_load_model(MODEL_PATH, compile=False)
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
                loaded = tf_load_model(SAVED_MODEL_PATH)
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

def detect_and_crop_face(image, return_bbox=False):
    """
    Detect and crop face from image using OpenCV
    Returns tuple: (cropped face image, face_detected boolean, [bbox coordinates if return_bbox=True])
    bbox format: (x, y, w, h) - bounding box coordinates
    """
    if not CV2_AVAILABLE:
        if return_bbox:
            return image, False, None
        return image, False
    
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
            
            # Store original bbox before padding
            original_bbox = (x, y, w, h)
            
            # Add padding around the face (20% on each side)
            padding_x = int(w * 0.2)
            padding_y = int(h * 0.2)
            
            x_padded = max(0, x - padding_x)
            y_padded = max(0, y - padding_y)
            w_padded = min(img_array.shape[1] - x_padded, w + 2 * padding_x)
            h_padded = min(img_array.shape[0] - y_padded, h + 2 * padding_y)
            
            # Crop the face region (with padding)
            cropped = img_array[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
            
            # Convert back to PIL Image
            cropped_image = Image.fromarray(cropped)
            
            if return_bbox:
                # Return bbox with padding
                return cropped_image, True, (x_padded, y_padded, w_padded, h_padded)
            return cropped_image, True
        else:
            # No face detected, return original image
            logger.warning("No face detected, using full image")
            if return_bbox:
                return image, False, None
            return image, False
            
    except Exception as e:
        logger.warning(f"Face detection failed: {e}, using full image")
        if return_bbox:
            return image, False, None
        return image, False

def draw_face_bbox(image, bbox):
    """
    Draw bounding box around detected face on image
    Returns image with bounding box drawn
    """
    if bbox is None:
        return image
    
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        x, y, w, h = bbox
        
        # Draw rectangle (green, thickness 3)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Add label "Face Detected"
        label = "Face Detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(img_bgr, (x, y - text_height - 10), (x + text_width + 10, y), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(img_bgr, label, (x + 5, y - 5), font, font_scale, (0, 0, 0), thickness)
        
        # Convert back to RGB and PIL Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
        
    except Exception as e:
        logger.error(f"Error drawing bounding box: {e}")
        return image

def preprocess_image(image, return_face_info=False, return_bbox=False):
    """
    Preprocess image for model prediction (matches FastAPI exactly)
    - Detect and crop face if possible
    - Convert to grayscale
    - Resize to 48x48
    - Normalize pixel values to [0, 1]
    
    Args:
        image: PIL Image
        return_face_info: If True, also return face detection status
        return_bbox: If True, also return bounding box coordinates
    
    Returns:
        Preprocessed image array, or tuple with additional info if requested
    """
    try:
        # Detect and crop face first
        if return_bbox:
            image, face_detected, bbox = detect_and_crop_face(image, return_bbox=True)
        else:
            image, face_detected = detect_and_crop_face(image)
            bbox = None
        
        # Convert to grayscale if needed (matches FastAPI exactly)
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 48x48 (matches FastAPI exactly)
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Convert to numpy array and normalize (matches FastAPI exactly)
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Reshape to match model input shape: (1, 48, 48, 1)
        img_array = img_array.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        
        if return_bbox:
            return img_array, face_detected, bbox
        elif return_face_info:
            return img_array, face_detected
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise ValueError(f"Error processing image: {str(e)}")

def predict_emotion(image, model, return_bbox=False):
    """Make prediction on preprocessed image"""
    if return_bbox:
        processed_image, face_detected, bbox = preprocess_image(image, return_face_info=True, return_bbox=True)
    else:
        processed_image, face_detected = preprocess_image(image, return_face_info=True)
        bbox = None
    
    predictions = model.predict(processed_image, verbose=0)
    
    # Get predicted class index and probability
    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])
    
    # Get all class probabilities
    class_probabilities = {
        CLASS_NAMES[i]: float(predictions[0][i])
        for i in range(len(CLASS_NAMES))
    }
    
    if return_bbox:
        return predicted_class, confidence, class_probabilities, face_detected, bbox
    return predicted_class, confidence, class_probabilities, face_detected

def plot_predictions(class_probabilities):
    """Create a bar chart of emotion predictions"""
    emotions = list(class_probabilities.keys())
    probabilities = list(class_probabilities.values())
    
    # Create color map based on probability
    colors = plt.cm.RdYlGn(probabilities)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(emotions, probabilities, color=colors)
    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_title('Emotion Prediction Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add value labels on bars
    for i, (emotion, prob) in enumerate(zip(emotions, probabilities)):
        ax.text(prob + 0.01, i, f'{prob:.2%}', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Main UI
def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.2);
        margin: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">😊 Facial Expression Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image or use your camera to detect facial expressions</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This application uses a deep learning model trained on the FER-2013 dataset 
        to recognize **7 different facial expressions**:
        
        - 😠 **Angry**
        - 🤢 **Disgust**
        - 😨 **Fear**
        - 😊 **Happy**
        - 😐 **Neutral**
        - 😢 **Sad**
        - 😲 **Surprise**
        """)
        
        st.markdown("---")
        st.header("⚙️ Model Status")
        
        # Check if model files exist
        model_exists = os.path.exists(MODEL_PATH) or (os.path.exists(SAVED_MODEL_PATH) and os.path.isdir(SAVED_MODEL_PATH))
        
        if model_exists:
            st.success("✅ Model file found")
            if os.path.exists(MODEL_PATH):
                st.caption(f"📁 {MODEL_PATH}")
            else:
                st.caption(f"📁 {SAVED_MODEL_PATH}")
        else:
            st.error("❌ Model file not found!")
            st.warning(f"Please ensure either {MODEL_PATH} or {SAVED_MODEL_PATH} exists")
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            model = load_model_instance()
        st.sidebar.success("✅ Model ready!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Main content area - Tabs for input methods
    tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Camera"])
    
    image = None
    image_source = None
    
    with tab1:
        st.markdown("### Upload an Image File")
        st.caption("Supported formats: JPG, JPEG, PNG, BMP")
        
        # File uploader with better styling
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a facial image for emotion recognition",
            label_visibility="visible"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image_source = "uploaded"
                st.success("✅ Image loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                st.stop()
    
    with tab2:
        st.markdown("### Capture from Camera")
        st.caption("Position your face in front of the camera and click the button to capture")
        
        # Camera input
        camera_image = st.camera_input(
            "Take a picture",
            help="Click the button to capture your face",
            label_visibility="visible"
        )
        
        if camera_image is not None:
            try:
                image = Image.open(camera_image)
                image_source = "camera"
                st.success("✅ Photo captured successfully!")
            except Exception as e:
                st.error(f"❌ Error processing camera image: {str(e)}")
                st.stop()
    
    # Prediction section - only show if image is available
    if image is not None:
        st.markdown("---")
        
        # Make prediction first (before displaying in columns)
        predicted_class = None
        confidence = None
        class_probabilities = None
        face_detected = False
        bbox = None
        image_with_bbox = image
        
        try:
            with st.spinner("🔍 Analyzing facial expression..."):
                predicted_class, confidence, class_probabilities, face_detected, bbox = predict_emotion(image, model, return_bbox=True)
            
            # Draw bounding box on image if face was detected
            if face_detected and bbox is not None and CV2_AVAILABLE:
                image_with_bbox = draw_face_bbox(image, bbox)
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.exception(e)
        
        # Create two columns for image and prediction
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Image Preview")
            
            # Display image with or without bounding box
            if face_detected and bbox is not None and CV2_AVAILABLE:
                st.image(image_with_bbox, use_container_width=True, caption="Face detected (green box)")
            else:
                st.image(image, use_container_width=True, caption="Your image")
            
            # Compact image info
            with st.expander("ℹ️ Image Details"):
                st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
                st.write(f"**Mode:** {image.mode}")
                st.write(f"**Source:** {'File Upload' if image_source == 'uploaded' else 'Camera'}")
                if CV2_AVAILABLE and bbox is not None:
                    st.write(f"**Face Bounding Box:** x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]}")
        
        with col2:
            st.markdown("### 🎯 Prediction Results")
            
            if predicted_class is not None:
                # Face detection status
                if CV2_AVAILABLE:
                    if face_detected:
                        st.success("✅ Face detected and cropped (see green box on image)")
                    else:
                        st.warning("⚠️ Face not detected - using full image (accuracy may be lower)")
                else:
                    st.info("ℹ️ Face detection not available (OpenCV not installed)")
                
                # Main prediction display with better styling
                emoji = EMOTION_EMOJIS.get(predicted_class, '😐')
                
                # Large prediction card
                st.markdown(f"""
                <div class="prediction-card">
                    <h1 style="font-size: 4rem; margin: 0;">{emoji}</h1>
                    <h2 style="margin: 0.5rem 0;">{predicted_class.upper()}</h2>
                    <p style="font-size: 1.2rem; margin: 0;">Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence progress bar
                st.progress(confidence)
                
                # All predictions in a compact table
                st.markdown("#### 📊 All Emotion Scores")
                df = pd.DataFrame({
                    'Emotion': list(class_probabilities.keys()),
                    'Confidence': [f"{v:.2%}" for v in class_probabilities.values()],
                    'Score': list(class_probabilities.values())
                })
                df = df.sort_values('Score', ascending=False)
                
                # Add emoji column
                df[''] = [EMOTION_EMOJIS.get(emotion, '😐') for emotion in df['Emotion']]
                df = df[['', 'Emotion', 'Confidence']]
                
                # Style the dataframe
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "": st.column_config.TextColumn("", width="small"),
                        "Emotion": st.column_config.TextColumn("Emotion", width="medium"),
                        "Confidence": st.column_config.TextColumn("Confidence", width="small")
                    }
                )
            else:
                st.warning("⚠️ Unable to make prediction. Please try again.")
        
        # Visualization section
        if 'class_probabilities' in locals() and class_probabilities is not None:
            st.markdown("---")
            st.markdown("### 📈 Detailed Visualization")
            
            try:
                fig = plot_predictions(class_probabilities)
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
    else:
        # Welcome message when no image
        st.markdown("---")
        st.info("👆 **Get started:** Upload an image file or use your camera to capture a photo above!")
        
        # Show example emotions
        st.markdown("### 🎭 Recognized Emotions")
        cols = st.columns(7)
        for i, (emotion, emoji) in enumerate(EMOTION_EMOJIS.items()):
            with cols[i]:
                st.markdown(f"### {emoji}")
                st.caption(emotion.capitalize())
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; padding: 1rem;'>"
        "FER-2013 Facial Expression Recognition Model | Built with Streamlit & TensorFlow"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

