from fastapi import FastAPI, UploadFile, File
from model import load_model, get_embedding
from utils import preprocess_image, cosine_similarity
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()
model = load_model()


@app.post("/compare")
async def compare_faces(
        user_img: UploadFile = File(...),
        celeb_img: UploadFile = File(...)
):
    try:
        # Zurückspulen der Dateien
        await user_img.seek(0)
        await celeb_img.seek(0)

        # Debug-Logging
        print(f"Processing images: {user_img.filename} and {celeb_img.filename}")

        # Gesichtserkennung
        user_face = preprocess_image(user_img)
        celeb_face = preprocess_image(celeb_img)

        if user_face is None or celeb_face is None:
            raise HTTPException(
                status_code=400,
                detail="Could not detect faces in one or both images"
            )

        # Embeddings
        user_emb = get_embedding(model, user_face)
        celeb_emb = get_embedding(model, celeb_face)

        # Ähnlichkeit
        similarity = cosine_similarity(user_emb, celeb_emb)

        return {
            "similarity": round(similarity * 100, 2),
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")  # Wichtig für Debugging
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )