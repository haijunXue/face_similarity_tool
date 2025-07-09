import torch
from facenet_pytorch import InceptionResnetV1

# Load pre-trained FaceNet model
def load_model():
    model = InceptionResnetV1(pretrained='vggface2').eval()
    return model

# Convert image tensor to embedding
def get_embedding(model, aligned_img_tensor):
    with torch.no_grad():
        embedding = model(aligned_img_tensor.unsqueeze(0))
    return embedding[0]