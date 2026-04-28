import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import time
import os

# Classes (VERY IMPORTANT — must match folder names)
# Check if we can automatically read classes from data folder, otherwise use fallback
if os.path.exists("data"):
    # Ignore hidden files or directories
    classes = sorted([d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))])
else:
    classes = ['sitting', 'standing']

print("Using classes:", classes)

if len(classes) == 0:
    print("No classes found. Make sure you collected data!")
    classes = ['sitting', 'standing']

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))

try:
    model.load_state_dict(torch.load("model.pth"))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you trained the model first!")
    exit()

model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Webcam
cap = cv2.VideoCapture(0)

# Variables for FPS tracking
prev_time = 0

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # Prepare image for the model
    img = transform(frame).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        outputs = model(img)
        
        # Calculate probabilities to get confidence score
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        label = classes[predicted.item()]
        confidence_percent = confidence.item() * 100

    # Draw a black rectangle background for better text visibility
    cv2.rectangle(frame, (10, 10), (350, 80), (0, 0, 0), -1)

    # Display prediction and confidence
    text_pred = f"Activity: {label} ({confidence_percent:.1f}%)"
    cv2.putText(frame, text_pred, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
                
    # Display FPS
    text_fps = f"FPS: {int(fps)}"
    cv2.putText(frame, text_fps, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    cv2.imshow("Activity Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()