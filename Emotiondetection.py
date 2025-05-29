import streamlit as st
from PIL import Image # Pillow library for image handling
import io
import numpy as np # To convert PIL image to numpy array for dlib
import dlib # Dlib for face detection and landmark prediction
import cv2 # OpenCV for drawing rectangles (optional, but good for visualization)
# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F # For ReLU and other functional operations

# --- Set page name for the app ---
st.set_page_config(layout="wide", page_title="Emotion Detection")

# Define the Convolutional Neural Network (CNN) Model
class EmotdetCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotdetCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Batch Normalization
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces image size by half (48 -> 24)
        self.dropout1 = nn.Dropout(0.25)

        # Second convolutional block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces image size by half (24 -> 12)
        self.dropout2 = nn.Dropout(0.25)

        # Third convolutional block (optional, but often helpful)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces image size by half (12 -> 6)
        self.dropout3 = nn.Dropout(0.25)

        # Fully connected layers
        # Calculate the input features for the first linear layer:
        # Image size: 48x48 -> (pool1) 24x24 -> (pool2) 12x12 -> (pool3) 6x6
        # Number of channels after last conv: 256
        self.fc1 = nn.Linear(256 * 6 * 6, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes) # num_classes is 7 for FER-2013

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten the feature maps
        x = x.view(x.size(0), -1) # Flatten for fully connected layers

        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        x = self.fc2(x) # Output layer (no activation here, as CrossEntropyLoss will apply softmax)
        return x

# Instantiate the model and move it to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotdetCNN(num_classes=7).to(device)

#Print the model architecture to verify
#print(model)

# --- Initialize Dlib's HOG face detector ---
# This line loads the pre-trained default face detector from dlib
detector = dlib.get_frontal_face_detector()

try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except Exception as e:
    st.error(f"Error loading dlib shape predictor model. Make sure 'shape_predictor_68_face_landmarks.dat' is in the same directory as your script. Error: {e}")
    st.stop() # Stop the app if the model isn't found

# --- Load the Trained PyTorch Model ---

MODEL_PATH = 'emotion_cnn_model.pth' 

@st.cache_resource # Cache the model loading for performance
def load_emotion_model():
    model = EmotdetCNN(num_classes=7)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval() # Set to evaluation mode
        st.success("Emotion CNN model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading emotion model: {e}")
        st.stop()

# Load the model at the start of the app
emotion_model = load_emotion_model()

# Define emotion labels (must match the order used during training by ImageFolder)
emotion_labels = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}

# Define the preprocessing transform for inference
preprocess_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to predict emotion
def predict_emotion(face_image_pil):
    # Apply the same transformations used during training
    input_tensor = preprocess_transform(face_image_pil)
    # Add a batch dimension (B, C, H, W)
    input_batch = input_tensor.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = emotion_model(input_batch)

    # Get probabilities
    probabilities = F.softmax(output, dim=1)
    # Get the predicted class
    _, predicted_idx = torch.max(probabilities, 1)
    predicted_emotion = emotion_labels[predicted_idx.item()]
    
    # Get the confidence for the predicted emotion
    confidence = probabilities[0, predicted_idx.item()].item()

    return predicted_emotion, confidence


def main():
    st.title("Detect Emotions from Images")
    st.write("Upload an image to detect the emotion!")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # --- File Format and Size Checks ---
        # Get file details
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
        st.write("File Details:")
        st.json(file_details) # Display file details as JSON

        # Check file type
        if not uploaded_file.type.startswith('image/'):
            st.error("Invalid file type. Please upload an image (JPG, JPEG, PNG).")
            return # Stop execution if not an image

        # Check file size (e.g., max 10MB)
        # 1 MB = 1,024 * 1,024 bytes
        max_file_size_mb = 10
        if uploaded_file.size > (max_file_size_mb * 1024 * 1024):
            st.error(f"Image size exceeds the limit of {max_file_size_mb} MB. Please upload a smaller image.")
            return # Stop execution if too large

        # Read the image and convert to RGB
        try:
            image = Image.open(uploaded_file).convert('RGB') 
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.success("Image uploaded successfully!")
            st.subheader("Processing Image and Detecting faces...")
            # --- Convert PIL Image to NumPy array for Dlib/OpenCV ---
            # Dlib and OpenCV typically work with NumPy arrays.
            # PIL images are (Height, Width, Channels) - RGB
            # Dlib/OpenCV expect (Height, Width, Channels) - BGR for some ops, but detector takes RGB fine
            # Let's keep it RGB for now, as dlib's detector works well with it.
            image_np = np.array(image) # Convert PIL image to NumPy array
            image_for_display = image_np.copy() # Use a copy for drawing on
            # --- Perform Face Detection ---
            # The '1' means we want to upsample the image 1 time.
            # Upsampling makes faces larger and can help detect smaller or distant faces,
            # but it also increases computation time.
            faces = detector(image_np, 1)

            st.write(f"Found {len(faces)} face(s) in the image.")

            # --- Draw Bounding Boxes on the image ---
            # Create a copy to draw on, so the original uploaded image isn't modified
            image_with_boxes = image_np.copy()

            if len(faces) > 0:
                for i, face in enumerate(faces):
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                    
                    # Ensure coordinates are within image bounds before cropping
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.width, x2), min(image.height, y2)

                    # Crop the face region from the original PIL image
                    # PIL's crop method takes (left, upper, right, lower)
                    face_crop_pil = image.crop((x1, y1, x2, y2)) 
                    
                    # --- Emotion Prediction ---
                    if face_crop_pil.width > 0 and face_crop_pil.height > 0: # Ensure valid crop
                        predicted_emotion, confidence = predict_emotion(face_crop_pil)
                        emotion_text = f"{predicted_emotion} ({confidence:.2f})"
                    else:
                        emotion_text = "N/A" # No valid crop

                    # Draw rectangle and emotion text on the image_for_display
                    cv2.rectangle(image_for_display, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
                    cv2.putText(image_for_display, f"Face {i+1}: {emotion_text}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # --- Predict 68 facial landmarks ---
                    landmarks = predictor(image_np, face) # Use the original image_np and the detected face

                    # Draw each landmark
                    for n in range(0, 68): # There are 68 landmarks
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        cv2.circle(image_for_display, (x, y), 2, (255, 0, 0), -1) # Red circle for landmarks

                st.image(image_for_display, caption='Detected Faces, Landmarks and Emotions', use_column_width=True)

            else:
                st.info("No faces detected in the uploaded image.")

        except Exception as e:
            st.error(f"Error during image processing: {e}")
            st.info("Please ensure a clear image with visible faces is uploaded.")


if __name__ == "__main__":
    main()