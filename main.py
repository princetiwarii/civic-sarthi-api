# =====================
# IMPORTS
# =====================
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageFile
import io
import json
import re
from google import genai   # ✅ NEW SDK
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# =====================
# FIX CORRUPTED IMAGES
# =====================
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =====================
# CONFIG
# =====================
app = FastAPI()

# 🔑 DIRECT API KEY
client = genai.Client(api_key="AIzaSyDYSQvvrINyZliGqTTeMEfs-41UReKETNw")

# =====================
# LOAD MODEL
# =====================
model = load_model("model/waste_classifier_finetuned.keras")

# Class labels (avoid json issue)
classes = ['cardboard', 'glass', 'hazardous', 'metal', 'paper', 'plastic', 'trash']

# =====================
# CATEGORY MAP
# =====================
CATEGORY_MAP = {
    "plastic": "Recyclable",
    "paper": "Recyclable",
    "cardboard": "Recyclable",
    "glass": "Recyclable",
    "metal": "Recyclable",
    "hazardous": "Hazardous",
    "trash": "General Waste"
}

# =====================
# PREDICTION
# =====================
def predict_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    class_index = np.argmax(preds)
    confidence = float(np.max(preds))

    return classes[class_index], confidence

# =====================
# CLEAN JSON
# =====================
def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group() if match else "{}"

# =====================
# GEMINI 2.5 FLASH (NEW SDK)
# =====================
def generate_llm_response(label, category):

    prompt = f"""
    You are an AI assistant for smart waste management.

    Waste Type: {label}
    Category: {category}

    Respond ONLY in valid JSON:
    {{
      "item": "",
      "disposal_method": "",
      "reuse_ideas": [],
      "warnings": []
    }}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text
        clean_text = extract_json(text)

        return json.loads(clean_text)

    except Exception:
        return {
            "item": label,
            "disposal_method": "Dispose responsibly",
            "reuse_ideas": [],
            "warnings": []
        }

# =====================
# SUMMARY
# =====================
def generate_summary(item, category, disposal):
    return f"{item} is classified as {category}. {disposal}"

# =====================
# ROUTES
# =====================
@app.get("/")
def home():
    return {"message": "Civic Sarthi AI API running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        label, confidence = predict_image(img)
        category = CATEGORY_MAP.get(label, "Unknown")

        llm_data = generate_llm_response(label, category)

        if confidence < 0.6:
            category = "Uncertain / Mixed Waste"

        return {
            "prediction": label,
            "category": category,
            "confidence": round(confidence, 3),
            "item": llm_data.get("item", label),
            "disposal_method": llm_data.get("disposal_method"),
            "reuse_ideas": llm_data.get("reuse_ideas", []),
            "warnings": llm_data.get("warnings", []),
            "nearby_facilities": [
                {
                    "name": "Nearest Recycling Center",
                    "distance": "2 km"
                }
            ],
            "summary": generate_summary(
                llm_data.get("item", label),
                category,
                llm_data.get("disposal_method", "")
            )
        }

    except Exception as e:
        return {"error": str(e)}