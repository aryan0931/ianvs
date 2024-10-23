# -*- coding: utf-8 -*-
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import torch
import textwrap
import matplotlib.pyplot as plt
from PIL import Image
import clip

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch version:", torch.__version__, "device", device)

# Load CLIP model and preprocess functions
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

# Load dataset
dataset_dir = "path_to_dataset_directory"
file_path = os.path.join(dataset_dir, 'captions.txt')
df = pd.read_csv(file_path, delimiter=',')
df.columns = ['image', 'caption']
df['caption'] = df['caption'].str.lstrip()

# Randomly select a sample of 12 images for visualization
random_sample = df.sample(n=12, random_state=1)
image_names = random_sample['image'].tolist()
descriptions = random_sample['caption'].tolist()

# Initialize lists for images and captions
original_images = []
texts = []
plt.figure(figsize=(16, 16))
wrapper = textwrap.TextWrapper(width=30)
image_dir = os.path.join(dataset_dir, 'Images')

# Loop over the selected sample of images and captions
for index, filename in enumerate(image_names):
    image = Image.open(os.path.join(image_dir, filename)).convert("RGB")
    plt.subplot(3, 4, len(original_images) + 1)
    plt.imshow(image)
    wrapped_text = wrapper.fill(text=descriptions[index])
    plt.title(f"{filename}\n{wrapped_text}", fontsize=10)
    plt.xticks([]), plt.yticks([])

    original_images.append(image)
    texts.append(descriptions[index])

    plt.tight_layout()

# Preprocess images for CLIP
images = [preprocess(image) for image in original_images]

# Convert to tensor and send to device
image_input = torch.stack(images).to(device)

# Tokenize text input
text_tokens = clip.tokenize([desc for desc in texts]).to(device)

# Run the forward pass to get image and text features
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()

# Normalize features
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Calculate cosine similarity
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

# Display similarity matrix
plt.figure(figsize=(20, 14))
plt.imshow(similarity, vmin=0.1, vmax=0.3)
plt.xticks([]), plt.yticks([])

for i, image in enumerate(original_images):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side].set_visible(False)

plt.xlim([-0.5, len(descriptions) - 0.5])
plt.ylim([len(descriptions) + 0.5, -2])
plt.title("Cosine similarity between text and image features", size=20)

# Function to compute top-k accuracy
def compute_accuracy(similarity, k, count):
    top_k_indices = np.argsort(similarity, axis=1)[:, -k:]
    correct_indices = np.arange(count).reshape(-1, 1)
    matches_top_k = np.any(top_k_indices == correct_indices, axis=1)
    top_k_accuracy = matches_top_k.mean()
    print(f"Evaluating {count} images for Top-{k} Accuracy: {top_k_accuracy:.2f}")

# Compute accuracy for various values of k
compute_accuracy(similarity, 1, len(descriptions))
compute_accuracy(similarity, 3, len(descriptions))
compute_accuracy(similarity, 5, len(descriptions))
compute_accuracy(similarity, 10, len(descriptions))
