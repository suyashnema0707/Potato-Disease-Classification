import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os

# --- MODEL DEFINITION & LOADING ---
# The best performing model was ResNet18 with transfer learning.
# We need to define the same architecture here to load the weights correctly.

# 1. Load the pre-trained ResNet18 model structure.
#    We pass weights=None because we are about to load our own fine-tuned weights.
model = torchvision.models.resnet18(weights=None)

# 2. Get the number of input features for the classifier
num_ftrs = model.fc.in_features

# 3. Replace the final layer to match the saved model's structure (3 output classes)
model.fc = nn.Linear(num_ftrs, 3)

# 4. Define the model path and class names
# This assumes it's in a 'models' subfolder within the backend.
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_potato_disease_model_new_dataset.pth')
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# 5. Load the saved weights into the ResNet18 architecture
# We are loading onto CPU, as the backend server might not have a GPU
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# --- IMAGE TRANSFORMATION ---
# This transform should be the same as the validation transform used during training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- PREDICTION FUNCTION ---
def predict_image(image_path):
    """
    Takes an image path, processes it, and returns the predicted class and confidence.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        img_transformed = transform(image)
        # Add a batch dimension (B, C, H, W) as the model expects it
        img_batch = img_transformed.unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_batch)

            # Apply Softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

            # Get the top probability (confidence) and its index
            confidence, predicted_idx_tensor = torch.max(probabilities, 0)

            predicted_class = CLASS_NAMES[predicted_idx_tensor.item()]
            confidence_score = confidence.item()

        # The function now returns two values
        return predicted_class, confidence_score

    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None

