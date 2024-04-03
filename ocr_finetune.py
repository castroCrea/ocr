# NO_WORKING
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import sys
from urllib.parse import urlparse

def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except AttributeError:
        return False


input_path_or_url = sys.argv[1]
# check if input_path_or_url contains url or path
is_url = uri_validator(input_path_or_url)

if(is_url):
  # url = "https://www.telegraph.co.uk/multimedia/archive/03363/panther-tank_3363684b.jpg?imwidth=680"
  image = Image.open(requests.get(input_path_or_url, stream=True).raw)
else:
  # path = "./images/Panther Tank.webp"
  image = Image.open(input_path_or_url)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("paolocl/detr-resnet-50_finetuned_cppe5")
model = DetrForObjectDetection.from_pretrained("paolocl/detr-resnet-50_finetuned_cppe5")

with torch.no_grad():
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

print(results)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )