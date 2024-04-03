# Generate a uniq image depending on a word

# install

```bash 
pip install -r requirements.txt
```

# Transformers

If you want to save a model locally 
```python
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

processor.save_pretrained("./models/facebook/detr-resnet-50")
model.save_pretrained("./models/facebook/detr-resnet-50")
```

If you want to load a model via hugging face

```bash
git clone https://huggingface.co/facebook/detr-resnet-50
pip install timm 
```

```python
processor = DetrImageProcessor.from_pretrained("./models/detr-resnet-50", local_files_only=True)
model = DetrForObjectDetection.from_pretrained("./models/detr-resnet-50", local_files_only=True)
```