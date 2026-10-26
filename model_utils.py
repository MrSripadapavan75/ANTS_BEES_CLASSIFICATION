from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# Locate the project directory
BASE_DIR = Path(__file__).resolve().parent

# Path to the saved model checkpoint
MODEL_PATH = BASE_DIR / "saved_model" / "ants_bees_resnet18.pth"
# Run predictions using CPU
DEVICE = torch.device("cpu")


def load_model():
    # Check whether the checkpoint exists
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    # Load the saved checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Support both checkpoint formats
    class_names = checkpoint.get(
        "class_names",
        checkpoint.get("classes")
    )

    if class_names is None:
        raise KeyError(
            "Checkpoint must contain 'class_names' or 'classes'."
        )

    # Read preprocessing configuration, using defaults if missing
    image_size = checkpoint.get("image_size", 224)

    normalization_mean = checkpoint.get(
        "normalization_mean",
        [0.485, 0.456, 0.406]
    )

    normalization_std = checkpoint.get(
        "normalization_std",
        [0.229, 0.224, 0.225]
    )

    # Re-create the ResNet18 architecture
    model = models.resnet18(weights=None)

    # Configure the final layer for the saved classes
    number_of_features = model.fc.in_features
    model.fc = nn.Linear(
        number_of_features,
        len(class_names)
    )

    # Load the trained weights
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)

    # Prepare the model for inference
    model.to(DEVICE)
    model.eval()

    return (
        model,
        class_names,
        image_size,
        normalization_mean,
        normalization_std
    )


# Load the model once when this module is imported
(
    model,
    CLASS_NAMES,
    IMAGE_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD
) = load_model()


# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=NORMALIZATION_MEAN,
        std=NORMALIZATION_STD
    ),
])


def predict_image(image: Image.Image):
    if image is None:
        raise ValueError("Please upload an image.")

    # Convert image to RGB
    image = image.convert("RGB")

    # Preprocess the image
    input_tensor = image_transform(image).unsqueeze(0)

    # Run prediction
    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Return class probabilities
    results = {
        CLASS_NAMES[index]: float(probabilities[index].item())
        for index in range(len(CLASS_NAMES))
    }

    return results
