import requests
import sys
import torch

from PIL import Image
from uri_validator import uri_validator
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

input_path_or_url = sys.argv[1]
# check if input_path_or_url contains url or path
is_url = uri_validator(input_path_or_url)

if(is_url):
  # url = "https://www.telegraph.co.uk/multimedia/archive/03363/panther-tank_3363684b.jpg?imwidth=680"
  image = Image.open(requests.get(input_path_or_url, stream=True).raw)
else:
  # path = "./images/Panther Tank.webp"
  image = Image.open(input_path_or_url)

candidate_labels = ["tank", "train","truk","tree", "car", "bike", "cat"]
inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)


with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits_per_image[0]
probs = logits.softmax(dim=-1).numpy()
scores = probs.tolist()

result = [
    {"score": score, "label": candidate_label}
    for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
]

print(result)