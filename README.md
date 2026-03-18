# Gesture-Detection-system-using-OpenCV
📌 Features
Real-time hand tracking using webcam
Detects up to 2 hands simultaneously
Extracts 21 hand landmarks per hand
Dataset creation using CSV
Machine Learning model using Random Forest Classifier
Easy to extend for custom gestures
🛠️ Technologies Used
Python
OpenCV (cv2)
MediaPipe
NumPy
Scikit-learn
CSV / OS
Pickle (for model saving)

🚀 How It Works
1️⃣ Hand Detection
The system uses MediaPipe Hands to detect hand landmarks in real-time.
👉 Example from your code:
Converts webcam frames to RGB
Detects landmarks
Draws connections between points
2️⃣ Data Collection
Captures landmark coordinates
Stores them in a CSV file
Each row represents one gesture
3️⃣ Model Training
Splits dataset using train_test_split
Trains using RandomForestClassifier
Saves model using pickle
4️⃣ Gesture Prediction
Loads trained model
Predicts gesture from live input
