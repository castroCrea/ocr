from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
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
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )

draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    class_label = model.config.id2label[label.item()]
    confidence = round(score.item(), 3)

    # Draw rectangle
    draw.rectangle(box, outline="red")
    
    # Add text
    text = f"{class_label}: {confidence}"
    text_width = draw.textlength(text)
    text_height = 10
    text_location = [box[0], box[1] - text_height - 4]
    draw.rectangle([text_location[0], text_location[1], text_location[0] + text_width, text_location[1] + text_height], fill="red")
    draw.text(text_location, text, fill="white")

image.show()