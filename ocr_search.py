import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import cv2
import os

# Charger le modèle CLIP pré-entraîné
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

compare_path = './images/paolo'
database_image_path = './dataset/fake'

# Load reference images from compare_path directory
reference_images = []
reference_image_paths = sorted(os.listdir(compare_path))
for img_path in reference_image_paths:
    image = cv2.imread(os.path.join(compare_path, img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reference_images.append(image)

# Preprocess reference images
inputs = processor(text=None, images=reference_images, return_tensors="pt", padding=True)

# Get features of reference images
with torch.no_grad():
    features = model.get_image_features(**inputs)

# Load images to compare from database_image_path directory
all_images = []
all_image_paths = sorted(os.listdir(database_image_path))
for img_path in all_image_paths:
    image = cv2.imread(os.path.join(database_image_path, img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    all_images.append(image)

# Preprocess images to compare
inputs_all = processor(text=None, images=all_images, return_tensors="pt", padding=True)

# Get features of images to compare
with torch.no_grad():
    features_all = model.get_image_features(**inputs_all)

# Calculate similarity between features of reference images and images to compare
similarities = torch.matmul(features, features_all.T)

# Find indices of most similar images for each reference image
top_k = 5  # Number of similar images to retrieve
similar_images_indices = torch.topk(similarities, top_k, dim=1).indices.numpy()

# Print the indices of similar images for each reference image
for i, indices in enumerate(similar_images_indices):
    print(f"Similar images for reference image {reference_image_paths[i]}:")
    for idx in indices:
        similarity_percentage = similarities[i][idx].item() * 100
        print(f"- {all_image_paths[idx]} (Similarity: {similarity_percentage:.2f}%)")

        # Get the position of the similarity in the picture
        position = np.where(similarities[i].numpy() == similarities[i][idx].item())
        print(f"  - Position in the picture: {position[0]}")