import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATA_PATH = "data/Audio_Speech_Actors_01-24"


EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

X = []
y = []

print("🔍 Scanning dataset...")

for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")
            if len(parts) > 2:
                emotion_code = parts[2]
                emotion = EMOTIONS.get(emotion_code)

                if emotion:
                    file_path = os.path.join(root, file)
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(emotion)

print(f"✅ Total audio samples loaded: {len(X)}")

if len(X) == 0:
    print("❌ No audio files loaded. Check data folder path.")
    exit()

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🧠 Training SVM model...")
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Model Accuracy: {accuracy*100:.2f}%")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/emotion_model.pkl")

print("💾 Model saved successfully in models/emotion_model.pkl")
