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

def preprocess_image(image):
    """
    Preprocess image for model prediction
    - Convert to grayscale
    - Resize to 48x48
    - Normalize pixel values to [0, 1]
    """
    try:
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
        raise ValueError(f"Error processing image: {str(e)}")

def predict_emotion(image, model):
    """Make prediction on preprocessed image"""
    processed_image = preprocess_image(image)
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
    
    return predicted_class, confidence, class_probabilities

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
    # Title and description
    st.title("😊 Facial Expression Recognition")
    st.markdown("**Upload an image to detect facial expressions using the FER-2013 model**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This application uses a deep learning model trained on the FER-2013 dataset 
        to recognize 7 different facial expressions:
        
        - 😠 Angry
        - 🤢 Disgust
        - 😨 Fear
        - 😊 Happy
        - 😐 Neutral
        - 😢 Sad
        - 😲 Surprise
        """)
        
        st.markdown("---")
        st.header("⚙️ Model Info")
        
        # Check if model files exist
        model_exists = os.path.exists(MODEL_PATH) or (os.path.exists(SAVED_MODEL_PATH) and os.path.isdir(SAVED_MODEL_PATH))
        
        if model_exists:
            st.success("✅ Model file found")
            if os.path.exists(MODEL_PATH):
                st.info(f"📁 Using: {MODEL_PATH}")
            else:
                st.info(f"📁 Using: {SAVED_MODEL_PATH}")
        else:
            st.error("❌ Model file not found!")
            st.warning(f"Please ensure either {MODEL_PATH} or {SAVED_MODEL_PATH} exists")
    
    # Load model
    try:
        model = load_model_instance()
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Main content area
    # Add tabs for different input methods
    tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Camera"])
    
    image = None
    image_source = None
    
    with tab1:
        st.subheader("Upload an Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a facial image for emotion recognition",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            try:
                image = Image.open(uploaded_file)
                image_source = "uploaded"
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                st.stop()
    
    with tab2:
        st.subheader("Capture from Camera")
        st.markdown("**Position your face in front of the camera and click the button to capture**")
        
        # Camera input
        camera_image = st.camera_input(
            "Take a picture",
            help="Click the button to capture your face",
            label_visibility="collapsed"
        )
        
        if camera_image is not None:
            try:
                image = Image.open(camera_image)
                image_source = "camera"
            except Exception as e:
                st.error(f"Error processing camera image: {str(e)}")
                st.stop()
    
    # Display image and info if available
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if image is not None:
            st.subheader("📷 Image Preview")
            st.image(image, caption=f"{'Uploaded' if image_source == 'uploaded' else 'Camera'} Image", use_container_width=True)
            
            # Show image info
            st.info(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
            st.info(f"**Image Mode:** {image.mode}")
            st.info(f"**Source:** {'File Upload' if image_source == 'uploaded' else 'Camera'}")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        class_probabilities = None
        
        if image is not None:
            try:
                # Make prediction
                with st.spinner("🔍 Analyzing facial expression..."):
                    predicted_class, confidence, class_probabilities = predict_emotion(image, model)
                
                # Display main prediction
                emoji = EMOTION_EMOJIS.get(predicted_class, '😐')
                st.markdown(f"### {emoji} **{predicted_class.upper()}**")
                st.markdown(f"**Confidence:** {confidence:.2%}")
                
                # Progress bar for confidence
                st.progress(confidence)
                
                st.markdown("---")
                
                # Display all predictions as a table
                st.subheader("📊 All Predictions")
                df = pd.DataFrame({
                    'Emotion': list(class_probabilities.keys()),
                    'Confidence': [f"{v:.2%}" for v in class_probabilities.values()],
                    'Score': list(class_probabilities.values())
                })
                df = df.sort_values('Score', ascending=False)
                df.index = [EMOTION_EMOJIS.get(emotion, '😐') for emotion in df['Emotion']]
                st.dataframe(df[['Emotion', 'Confidence']], use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                class_probabilities = None
        else:
            st.info("👆 Please upload an image or use the camera to see predictions")
    
    # Visualization section
    if image is not None and class_probabilities is not None:
        st.markdown("---")
        st.subheader("📈 Prediction Visualization")
        
        try:
            fig = plot_predictions(class_probabilities)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "FER-2013 Facial Expression Recognition Model | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

