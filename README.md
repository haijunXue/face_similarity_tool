<img width="1076" height="1306" alt="image" src="https://github.com/user-attachments/assets/6ff33215-f6cc-4214-b564-fb1775c86471" />


Face Similarity Tool
Ein KI-basiertes Tool zur Gesichtsvergleichsanalyse, das auf FaceNet und MTCNN aufbaut. Die Anwendung besteht aus einem FastAPI-Backend und einem Streamlit-Frontend.

https://img.shields.io/badge/Face-Comparison-blue https://img.shields.io/badge/AI-Powered-green https://img.shields.io/badge/Privacy-First-brightgreen

✨ Funktionen
Gesichtserkennung: Automatische Erkennung von Gesichtern in Bildern

Ähnlichkeitsanalyse: Berechnung der Gesichtsähnlichkeit mittels KI

100% Privatsphäre: Alle Verarbeitungen erfolgen lokal ohne Cloud-Uploads

Benutzerfreundliches Interface: Intuitive Web-Oberfläche mit Streamlit

RESTful API: Vollständige API für Integrationen

🛠️ Technologien
Backend: FastAPI, Python, Uvicorn

Frontend: Streamlit

KI-Modelle: FaceNet (InceptionResnetV1), MTCNN

Bildverarbeitung: OpenCV, Pillow, TorchVision

Machine Learning: PyTorch, scikit-learn

📦 Installation
Voraussetzungen
Python 3.8 oder höher

pip (Python Package Manager)

Schritt-für-Schritt Installation
Repository klonen:

bash
git clone <repository-url>
cd face_similarity_tool
Virtuelle Umgebung erstellen und aktivieren:

bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
Abhängigkeiten installieren:

bash
pip install -r requirements.txt
🚀 Verwendung
Backend-Server starten
Server starten:

bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
Das Backend ist jetzt erreichbar unter:

API: http://localhost:8000

Dokumentation: http://localhost:8000/docs

Alternative Dokumentation: http://localhost:8000/redoc

Frontend starten
In einem neuen Terminal (virtuelle Umgebung aktivieren):

bash
cd frontend
streamlit run app.py
Das Frontend ist jetzt erreichbar unter: http://localhost:8501

📖 API-Endpunkte
POST /compare
Vergleicht zwei Gesichter und gibt Ähnlichkeits-Score zurück.

Request:

user_img: Bilddatei (JPG/PNG)

celeb_img: Bilddatei (JPG/PNG)

Response:

json
{
  "similarity": 85.42,
  "status": "success"
}
GET /
Health-Check Endpunkt gibt API-Status zurück.

🖼️ Wie man es benutzt
Öffnen Sie das Frontend im Browser (http://localhost:8501)

Laden Sie zwei Bilder hoch (je eines pro Spalte)

Klicken Sie auf "Compare Faces"

Sehen Sie sich das Ähnlichkeitsergebnis an:

> 75%: Hohe Ähnlichkeit (wahrscheinlich dieselbe Person)

50-75%: Mittlere Ähnlichkeit (mögliche Übereinstimmung)

< 50%: Geringe Ähnlichkeit (wahrscheinlich verschiedene Personen)

🧪 Testing
Führen Sie die Tests mit dem bereitgestellten Test-Skript aus:

bash
cd app
python evaluate_similarity.py
Dies führt eine Auswertung mit den Testbildern durch und gibt ROC-AUC und Accuracy aus.

🔧 Konfiguration
Modelleinstellungen (app/model.py)
pretrained='vggface2': Vortrainiertes FaceNet-Modell

Modell kann auf GPU beschleunigt werden (ändere device='cpu' zu device='cuda')

Gesichtserkennung (app/utils.py)
MTCNN-Parameter können angepasst werden:

keep_all=True: Erkennt mehrere Gesichter

Bildgröße: (160, 160) für FaceNet kompatibel

🤝 Beitragen
Beiträge sind willkommen! Bitte:

Forken Sie das Repository

Erstellen Sie einen Feature-Branch

Committen Sie Ihre Änderungen

Pushen Sie den Branch

Erstellen Sie einen Pull Request
