import torch
import pandas as pd
from model import load_model, get_embedding
from utils import preprocess_image, cosine_similarity
from sklearn.metrics import roc_auc_score, accuracy_score
import os

model = load_model()
model.eval()

df = pd.read_csv("app/test_pairs.csv")
image_dir = "app/test_images"
y_true, y_pred = [], []


for _, row in df.iterrows():
    try:
        img1_path = os.path.join(image_dir, row["image1"])
        img2_path = os.path.join(image_dir, row["image2"])

        with open(img1_path, "rb") as f1, open(img2_path, "rb") as f2:
            face1 = preprocess_image(f1)
            face2 = preprocess_image(f2)

        if face1 is None or face2 is None:
            print(f"❌ Gesicht nicht erkannt: {row['image1']} oder {row['image2']}")
            continue

        emb1 = get_embedding(model, face1)
        emb2 = get_embedding(model, face2)
        similarity = cosine_similarity(emb1, emb2)

        y_true.append(row["label"])
        y_pred.append(similarity)

    except Exception as e:
        print(f"Fehler bei {row['image1']} / {row['image2']}: {e}")

# Bewertung
threshold = 0.6
binary_pred = [1 if sim >= threshold else 0 for sim in y_pred]

roc_auc = roc_auc_score(y_true, y_pred)
acc = accuracy_score(y_true, binary_pred)

print(f"\n📈 ROC AUC Score: {roc_auc:.4f}")
print(f"✅ Accuracy (bei Schwelle {threshold}): {acc:.2%}")
