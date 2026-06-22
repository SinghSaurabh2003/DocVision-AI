# Dataset 
DATASET_PATH = "/kaggle/input/datasets/singhsaurabh03/training-half/Training_half" 

# Image 
IMAGE_SIZE = 224 

# DataLoader 
BATCH_SIZE = 32 
VAL_SPLIT = 0.2 
NUM_WORKERS = 2 

# Training 
EPOCHS = 25 
LEARNING_RATE = 1e-5 
WEIGHT_DECAY = 1e-5 

# Model 
NUM_CLASSES = 9 

# Seed 
SEED = 42 

# Save 
CHECKPOINT_DIR = "/kaggle/working/checkpoints" 
BEST_MODEL_NAME = "best_model.pth" 
LAST_MODEL_NAME = "last_model.pth"
