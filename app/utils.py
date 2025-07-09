import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps
import io
import numpy as np
from typing import Optional, Tuple

# Initialize MTCNN with optimized parameters
mtcnn = MTCNN(
    image_size=160,
    margin=40,
    min_face_size=40,  # Better for small faces
    thresholds=[0.6, 0.7, 0.7],
    post_process=False,  # Faster processing
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


def smart_rotate(image: Image.Image) -> Image.Image:
    """Enhanced rotation with EXIF and size-aware fallback"""
    try:
        # First attempt EXIF correction
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # Size-based rotation logic
    width, height = image.size
    aspect_ratio = width / height

    # Rotate if significantly landscape (with tolerance)
    if aspect_ratio > 1.3:  # More conservative threshold
        image = image.rotate(90, expand=True, resample=Image.BICUBIC)

        # Remove black borders more efficiently
        arr = np.array(image)
        non_black = (arr.mean(axis=2) > 10)  # Threshold for "black"
        if non_black.any():
            coords = np.where(non_black)
            y0, x0 = coords[0].min(), coords[1].min()
            y1, x1 = coords[0].max() + 1, coords[1].max() + 1
            image = Image.fromarray(arr[y0:y1, x0:x1])

    return image


def preprocess_image(upload_file) -> Optional[torch.Tensor]:
    """Enhanced preprocessing pipeline with debug capabilities"""
    try:
        # Reset file pointer and verify content
        upload_file.file.seek(0)
        image_data = upload_file.file.read()
        if len(image_data) < 1024:  # Minimum reasonable file size
            print("⚠️ Suspiciously small file")
            return None

        # Load and orient image
        img = Image.open(io.BytesIO(image_data))
        img = smart_rotate(img).convert('RGB')

        # Debug: Save intermediate image
        img.save("debug_preprocessed.jpg")

        # Face detection with confidence check
        face_tensor = mtcnn(img, save_path="debug_face.jpg")

        if face_tensor is None:
            print("❌ No face detected - trying alternative approach")
            # Fallback: Try with increased minimum face size
            mtcnn_temp = MTCNN(min_face_size=20, device=mtcnn.device)
            face_tensor = mtcnn_temp(img)

        return face_tensor

    except IOError as e:
        print(f"🚨 File error: {str(e)}")
        return None
    except Exception as e:
        print(f"🚨 Processing crashed: {str(e)}")
        return None


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """Stable cosine similarity with input validation"""
    if vec1.dim() != 1 or vec2.dim() != 1:
        raise ValueError("Inputs must be 1D tensors")

    # Normalize vectors for more stable results
    vec1 = vec1 / torch.norm(vec1)
    vec2 = vec2 / torch.norm(vec2)

    return torch.nn.functional.cosine_similarity(
        vec1.unsqueeze(0),
        vec2.unsqueeze(0)
    ).item()