import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Load pre-trained CNN model (e.g., ResNet50)
model = models.resnet50(pretrained=True)
model.eval()

# Remove the classification layer
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# Define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pictures of the celebrity you want to compare
compare_path = './images/paolo'
reference_image_paths = sorted([os.path.join(compare_path, img_path) for img_path in os.listdir(compare_path)])
reference_images = [Image.open(img_path).convert('RGB') for img_path in reference_image_paths[:5]]  # Load first 5 images

# Extract features for reference images
reference_features = []
for image in reference_images:
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        features = model(input_batch)
    reference_features.append(features.squeeze().numpy())

# Load celebrity pictures
database_image_path = './dataset/fake'
celebrity_image_paths = sorted([os.path.join(database_image_path, img_path) for img_path in os.listdir(database_image_path)])
celebrity_images = [Image.open(img_path).convert('RGB') for img_path in celebrity_image_paths]

# Extract features for celebrity images
celebrity_features = []
for image in celebrity_images:
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        features = model(input_batch)
    celebrity_features.append(features.squeeze().numpy())

# Calculate cosine similarity between features of celebrity pictures and reference images
similarities = np.dot(celebrity_features, np.array(reference_features).T)
norms = np.linalg.norm(celebrity_features, axis=1)[:, np.newaxis] * np.linalg.norm(reference_features, axis=1)
similarities = similarities / norms

# Find the indices of the top most similar reference images for each celebrity picture
top_k_indices = np.argsort(similarities, axis=1)[:, ::-1]

# Filter and print only the matches with similarity score above 70%
threshold = 0.7
for i, indices in enumerate(top_k_indices):
    print(f"Matches for celebrity picture {celebrity_image_paths[i]}:")
    for idx in indices:
        similarity_score = similarities[i][idx]
        if similarity_score > threshold:
            print(f"- {reference_image_paths[idx]} (Similarity Score: {similarity_score:.4f})")
