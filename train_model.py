"""
Facial Expression Recognition Model Training Script
Extracted from fer2013-cnn.ipynb for direct execution
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, ReLU, Flatten, Input, Add
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ============================================================================
# GPU CONFIGURATION FOR CUDA
# ============================================================================
print("=" * 70)
print("GPU/CUDA Configuration Check")
print("=" * 70)

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# List all physical devices
print("\nAvailable devices:")
for device in tf.config.list_physical_devices():
    print(f"  - {device}")

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\nNumber of GPUs detected: {len(gpus)}")

if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[OK] Memory growth enabled for GPU: {gpu.name}")
        
        # Set GPU as the default device
        print(f"\n[OK] GPU will be used for training: {gpus[0].name}")
        
        # Get GPU details
        try:
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            if 'device_name' in gpu_details:
                print(f"  GPU Name: {gpu_details['device_name']}")
        except:
            pass
        
        # Verify CUDA is available
        try:
            cuda_built = tf.test.is_built_with_cuda()
            print(f"  CUDA Built: {cuda_built}")
        except:
            print(f"  CUDA Built: Unknown")
        
        # Check GPU availability (newer TensorFlow API)
        try:
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            print(f"  GPU Available: {gpu_available}")
        except:
            try:
                gpu_available = tf.test.is_gpu_available(cuda_only=True)
                print(f"  GPU Available: {gpu_available}")
            except:
                print(f"  GPU Available: True (detected {len(gpus)} GPU(s))")
        
        # Enable mixed precision training for RTX GPUs (faster training)
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"  [OK] Mixed precision (FP16) enabled for faster training")
        except Exception as e:
            print(f"  Note: Mixed precision not enabled ({str(e)[:50]})")
        
    except RuntimeError as e:
        print(f"\n[ERROR] Error configuring GPU: {e}")
        print("  Training will fall back to CPU")
else:
    print("\n[WARNING] No GPU detected!")
    print("  Training will use CPU (much slower)")
    print("  Note: TensorFlow CPU version installed. For GPU support, install tensorflow[and-cuda]")

print("=" * 70)
print()

# Test GPU computation to verify it's working
print("Testing GPU computation...")
try:
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        # Create a simple tensor and perform computation
        test_tensor = tf.random.normal((1000, 1000))
        result = tf.matmul(test_tensor, test_tensor)
        device_used = 'GPU' if gpus else 'CPU'
        print(f"[OK] Computation test successful on {device_used}")
        print(f"  Result shape: {result.shape}")
        print(f"  Device used: {result.device}")
        
        # Test GPU memory
        if gpus:
            try:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                print(f"  GPU Memory - Current: {gpu_memory['current'] / 1024**3:.2f} GB")
                print(f"  GPU Memory - Peak: {gpu_memory['peak'] / 1024**3:.2f} GB")
            except:
                pass
except Exception as e:
    print(f"[WARNING] GPU test failed: {e}")
    print("  Will attempt to use GPU during training anyway")

if gpus:
    print("\n" + "="*70)
    print("[OK] GPU (RTX 5070 Ti 12GB) is ready for training!")
    print("="*70)
    print("\nTraining optimizations enabled:")
    print("  [OK] GPU memory growth (prevents OOM errors)")
    print("  [OK] Automatic GPU device placement")
    try:
        policy = tf.keras.mixed_precision.global_policy()
        if 'mixed_float16' in str(policy):
            print("  [OK] Mixed precision training (FP16 - faster on RTX GPUs)")
    except:
        pass
    print("\nYour model will train significantly faster on GPU!")
else:
    print("\n[INFO] Training will use CPU (slower but will work)")

# ============================================================================
# CONSTANTS
# ============================================================================
print("\n" + "="*70)
print("Configuration")
print("="*70)

# Constants - Updated for local environment
TRAIN_DIR = './train'
TEST_DIR = './test'

IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
IMAGE_DEPTH = 1
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Output paths
MODEL_SAVE_PATH = 'best_model.h5'
HISTORY_SAVE_PATH = 'train_history.pkl'
SAVEDMODEL_PATH = 'saved_model'
TFLITE_PATH = 'model.tflite'

print(f"Train directory: {TRAIN_DIR}")
print(f"Test directory: {TEST_DIR}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("="*70)
print()

# ============================================================================
# DATA LOADING
# ============================================================================
print("Loading data...")

# Data augmentation for training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.15,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

print("Loading Training Data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

print("Loading Validation Data...")
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

print("Loading Test Data...")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Auto detect labels
CLASS_NAMES = list(train_generator.class_indices.keys())
NUM_CLASSES = train_generator.num_classes
print(f"\nDetected {NUM_CLASSES} classes: {CLASS_NAMES}")

# ============================================================================
# LOAD SAVED MODEL OR BUILD NEW
# ============================================================================
print("\n" + "="*70)
print("Loading Checkpoint or Building Model")
print("="*70)

# Try to load existing checkpoint first
if os.path.exists(MODEL_SAVE_PATH):
    print(f"Found checkpoint: {MODEL_SAVE_PATH}")
    print("Loading saved model to resume training...")
    try:
        model = load_model(MODEL_SAVE_PATH)
        print("[OK] Model loaded successfully from checkpoint!")
        print("Resuming training from saved checkpoint...")
        BUILD_NEW_MODEL = False
    except Exception as e:
        print(f"[WARNING] Could not load checkpoint: {e}")
        print("Building new model instead...")
        BUILD_NEW_MODEL = True
else:
    print("No checkpoint found. Building new model...")
    BUILD_NEW_MODEL = True

if BUILD_NEW_MODEL:
    # ============================================================================
    # MODEL ARCHITECTURE
    # ============================================================================
    print("Building Model")
    print("="*70)

    def residual_block(x, filters):
        shortcut = x
        
        # Check if dimensions match, if not, add a projection layer
        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, padding='same', kernel_initializer='he_normal')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        x = Add()([shortcut, x])
        x = ReLU()(x)
        return x

    print(f"Building model with {NUM_CLASSES} classes...")

    inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))

    # Block 1
    x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = residual_block(x, 64)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = residual_block(x, 128)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.25)(x)

    # Block 3
    x = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = residual_block(x, 256)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.4)(x)

    # GAP + Dense
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax', name='predictions', dtype='float32')(x)  # Ensure float32 for softmax

    model = Model(inputs, outputs)
    print("\nModel Summary:")
    model.summary()
else:
    print("\nModel Summary (from checkpoint):")
    model.summary()

# ============================================================================
# MODEL COMPILATION
# ============================================================================
print("\n" + "="*70)
print("Compiling Model")
print("="*70)

optimizer = AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4)
model.compile(
    loss='categorical_crossentropy', 
    optimizer=optimizer, 
    metrics=['accuracy']
)

try:
    policy = tf.keras.mixed_precision.global_policy()
    if 'mixed_float16' in str(policy):
        print(f"[OK] Model compiled with mixed precision (FP16) - faster GPU training")
        print(f"  Learning rate: {LEARNING_RATE}")
    else:
        print(f"[OK] Model compiled with learning rate: {LEARNING_RATE}")
except:
    print(f"[OK] Model compiled with learning rate: {LEARNING_RATE}")

# ============================================================================
# TRAINING
# ============================================================================
print("\n" + "="*70)
print("Starting Training")
print("="*70)

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1,
    save_weights_only=False
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Check for previous training history to determine starting epoch
initial_epoch = 0
if os.path.exists(HISTORY_SAVE_PATH):
    try:
        with open(HISTORY_SAVE_PATH, "rb") as f:
            previous_history = pickle.load(f)
        if 'loss' in previous_history:
            initial_epoch = len(previous_history['loss'])
            print(f"\n[OK] Found previous training history")
            print(f"  Resuming from epoch: {initial_epoch + 1}")
            if 'accuracy' in previous_history:
                print(f"  Previous best training accuracy: {max(previous_history['accuracy']):.4f}")
            if 'val_accuracy' in previous_history:
                print(f"  Previous best validation accuracy: {max(previous_history['val_accuracy']):.4f}")
    except Exception as e:
        print(f"\n[WARNING] Could not load previous history: {e}")
        initial_epoch = 0

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Starting from epoch: {initial_epoch + 1}")
print(f"Maximum epochs: {EPOCHS}")
print(f"Device: {'GPU' if gpus else 'CPU'}")
print("\nTraining started...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# Save training history
with open(HISTORY_SAVE_PATH, "wb") as f:
    pickle.dump(history.history, f)
print(f"\n[OK] Saved training history to {HISTORY_SAVE_PATH}")

# Load best model
print(f"\nLoading best model from {MODEL_SAVE_PATH}...")
model = load_model(MODEL_SAVE_PATH)
print("[OK] Best model loaded successfully!")

# ============================================================================
# EXPORT MODEL
# ============================================================================
print("\n" + "="*70)
print("Exporting Model in Multiple Formats")
print("="*70)

# 1. H5 format
try:
    model.save(MODEL_SAVE_PATH)
    print(f"[OK] Model saved in H5 format: {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"[WARNING] H5 export failed: {e}")

# 2. SavedModel format
try:
    model.save(SAVEDMODEL_PATH, save_format='tf')
    print(f"[OK] Model exported in SavedModel format: {SAVEDMODEL_PATH}/")
except Exception as e:
    print(f"[WARNING] SavedModel export failed: {e}")

# 3. TensorFlow Lite format
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    file_size_mb = os.path.getsize(TFLITE_PATH) / (1024*1024)
    print(f"[OK] Model exported in TFLite format: {TFLITE_PATH}")
    print(f"  Size: {file_size_mb:.2f} MB")
except Exception as e:
    print(f"[WARNING] TFLite export failed: {e}")

print("\n" + "="*70)
print("Training Complete!")
print("="*70)
print(f"  H5 Format:        {MODEL_SAVE_PATH}")
print(f"  SavedModel:       {SAVEDMODEL_PATH}/")
print(f"  TFLite:           {TFLITE_PATH}")
print(f"  Training History: {HISTORY_SAVE_PATH}")
print("="*70)
