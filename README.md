# 🧠 Stress Detection Using Image Processing and Deep Learning

A machine learning-based system that detects stress levels from facial images using advanced image processing techniques and deep learning models.

## 📋 Overview

This project implements a stress detection system that analyzes facial features to determine stress levels. It combines traditional image processing techniques (Local Binary Patterns) with modern machine learning algorithms (MLP and XGBoost) to achieve accurate stress classification.

## ✨ Features

- **Real-time Stress Detection**: Analyze facial images to detect stress levels
- **Multiple ML Models**: Utilizes both MLP (Multi-Layer Perceptron) and XGBoost classifiers
- **Facial Landmark Detection**: Uses dlib's 68-point facial landmark detector
- **LBP Feature Extraction**: Implements Local Binary Patterns for texture analysis
- **Web Interface**: Flask-based web application for easy interaction
- **Pre-trained Models**: Includes trained models ready for inference

## 🛠️ Technologies Used

- **Python 3.x**
- **Machine Learning**: scikit-learn, XGBoost
- **Deep Learning**: Neural Networks (MLP)
- **Computer Vision**: OpenCV, dlib
- **Web Framework**: Flask
- **Data Processing**: NumPy, Pandas, joblib

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/MishalHQ/Stress-Detection.git
cd Stress-Detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dlib shape predictor** (if not included)
```bash
# The shape_predictor_68_face_landmarks.dat file is required
# Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract and place in the project root directory
```

## 🚀 Usage

### Running the Web Application

```bash
python main.py
```

The application will start on `http://localhost:5000`

### Using the Models

```python
import joblib
from lbp import extract_features  # Your feature extraction module

# Load pre-trained models
mlp_model = joblib.load('mlp_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

# Extract features from image
features = extract_features(image_path)

# Scale features
features_scaled = scaler.transform(features)

# Predict stress level
prediction = mlp_model.predict(features_scaled)
```

## 📁 Project Structure

```
Stress-Detection/
├── main.py                                    # Flask web application
├── lbp.py                                     # LBP feature extraction
├── real.py                                    # Real-time detection module
├── mlp_model.joblib                          # Trained MLP model
├── xgb_model.joblib                          # Trained XGBoost model
├── scaler.joblib                             # Feature scaler
├── shape_predictor_68_face_landmarks.dat     # Facial landmark detector
├── requirements.txt                          # Python dependencies
├── IEEE_MiniProject.pdf                      # Project documentation
├── static/                                   # Static files (CSS, JS, images)
├── templates/                                # HTML templates
└── README.md                                 # This file
```

## 🧪 How It Works

1. **Image Input**: User provides a facial image
2. **Face Detection**: System detects face and extracts 68 facial landmarks using dlib
3. **Feature Extraction**: Local Binary Patterns (LBP) are computed from facial regions
4. **Preprocessing**: Features are scaled using the pre-trained scaler
5. **Classification**: MLP or XGBoost model predicts stress level
6. **Output**: System returns stress classification result

### Models

- **MLP (Multi-Layer Perceptron)**: Neural network-based classifier for pattern recognition
- **XGBoost**: Gradient boosting algorithm for robust classification
- Both models are trained on facial feature datasets and achieve high accuracy

## 📊 Dependencies

Key dependencies (see `requirements.txt` for complete list):
- Flask - Web framework
- OpenCV - Image processing
- dlib - Facial landmark detection
- scikit-learn - Machine learning algorithms
- XGBoost - Gradient boosting
- NumPy - Numerical computing
- joblib - Model serialization

## 📄 Documentation

For detailed project information, refer to `IEEE_MiniProject.pdf` included in the repository.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Mohammed Mishal**
- GitHub: [@MishalHQ](https://github.com/MishalHQ)

## 🙏 Acknowledgments

- dlib library for facial landmark detection
- scikit-learn and XGBoost communities
- Research papers on stress detection and facial analysis

---

⭐ If you find this project useful, please consider giving it a star!