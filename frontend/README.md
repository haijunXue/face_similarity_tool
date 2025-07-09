# 🧠 Face Similarity Tool

Ein Deep-Learning-Tool zur Bewertung der Gesichtähnlichkeit zwischen Nutzer- und Prominentenbildern mittels PyTorch, OpenCV und Face Embeddings. Entwickelt zur Unterstützung von Beauty-/Ästhetik-Anwendungen.

## 🔍 Funktionen
- Upload eines Nutzerfotos
- Auswahl eines Prominenten aus der Galerie
- Berechnung der Ähnlichkeit mit vortrainierten Face Embeddings (FaceNet/ArcFace)
- Darstellung der Ähnlichkeitsbewertung als Prozentwert
- Optional: GAN-Visualisierung für morphende Gesichter

## 🛠 Technologien
- PyTorch, OpenCV
- FaceNet / ArcFace Embeddings
- FastAPI (Backend)
- Streamlit (Frontend)
- FAISS (optional für schnelle Ähnlichkeitssuche)

## 📦 Setup
```bash
git clone https://github.com/deinname/face-similarity-tool
cd face_similarity_tool
pip install -r requirements.txt
