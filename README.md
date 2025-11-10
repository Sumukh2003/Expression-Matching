⚡ Expression Matching
Real-Time Emotion Recognition and Image Response Using Deep Learning

Expression Matching is a real-time system that detects human facial expressions from a live webcam feed and displays a matching image that reflects the detected mood.
Built from scratch with PyTorch, OpenCV, and Python, it demonstrates how deep learning and computer vision can merge to create interactive, expressive applications.

🧩 Project Overview

This project uses a custom Convolutional Neural Network (CNN) trained on a dataset of facial images to classify emotions such as happy, neutral, surprised, thinking, and others.
Each webcam frame is processed in real time, passed through the trained model, and paired with an image that visually represents the detected emotion.

It’s an engaging way to explore real-time inference, image processing, and neural network integration in Python.

🛠️ Technologies Used
🐍 Python 3.13

Main programming language used for all scripts and pipeline logic.

🔬 PyTorch

Framework used to design and train the CNN model.

Manages tensors, backpropagation, and classification layers.

Supports GPU acceleration for faster model execution.

🎞️ OpenCV

Handles webcam access and real-time video capture.

Combines video frames with predicted emotion visuals.

Performs color conversion, resizing, and display handling.

🧮 NumPy

Performs fast numerical operations and matrix manipulation.

Merges video and image arrays for real-time display.

🖌️ Pillow (PIL)

Converts and resizes images before feeding them to the model.

Bridges compatibility between OpenCV and PyTorch pipelines.

📦 Torchvision

Simplifies dataset loading and preprocessing with ImageFolder and transform utilities.

🎯 Key Takeaways

Developed a CNN architecture fully from scratch without pre-trained models.

Integrated a trained model with OpenCV for real-time prediction.

Learned dataset creation, labeling, and model evaluation techniques.

Optimized the frame-processing pipeline for live performance.

🚀 How to Run the Project

# Create a Virtual Environment

python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install Dependencies

python3 -m pip install --upgrade pip --break-system-packages
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages
python3 -m pip install opencv-python pillow numpy

# Capture Training Data

python3 capture_dataset.py

# Train the Model

python3 train_model.py

# Run Real-Time Detection

python3 detect_expression.py
