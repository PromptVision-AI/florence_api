import cloudinary
import cloudinary.uploader
from app.core.config import CLOUDINARY_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET

def configure_cloudinary():
    """
    Configure Cloudinary using environment variables.
    Make sure you have CLOUDINARY_NAME, CLOUDINARY_API_KEY, 
    and CLOUDINARY_API_SECRET set in your .env file.
    """
    cloudinary.config(
        cloud_name=CLOUDINARY_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
    )

def upload_image_to_cloudinary(image_file):
    """
    Uploads an image file to Cloudinary and returns the secure URL.

    Args:
        image_file: The file object to upload

    Returns:
        str: The secure URL of the uploaded image
    """
    result = cloudinary.uploader.upload(image_file)
    secure_url = result.get("secure_url")
    return secure_url 