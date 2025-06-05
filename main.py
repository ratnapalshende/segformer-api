from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model and feature extractor once
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model.eval()

def decode_segmentation(segmentation, num_classes=150):
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    color_seg = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    for label in range(num_classes):
        color_seg[segmentation == label] = colors[label]
    return Image.fromarray(color_seg)

@app.post("/segment/")
async def segment_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    seg = logits.argmax(dim=1)[0].detach().cpu().numpy()

    seg_image = decode_segmentation(seg)

    # Convert to bytes
    img_byte_arr = io.BytesIO()
    seg_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")
