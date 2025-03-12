import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', 'model')

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Cloudinary Configuration
CLOUDINARY_NAME = os.getenv("CLOUDINARY_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET") 