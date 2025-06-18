from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model
model_kualitas = YOLO("train21/weights/best.pt")   # T1-T4
model_penyakit = YOLO("train6/weights/best.pt")    # penyakit

CONFIDENCE_THRESHOLD = 0.3

penyakit_deskripsi = {
    "Blister Blight": "Si perusak daun muda. Jamur Exobasidium vexans menyebabkan gelembung kecil pada daun muda, terutama saat musim hujan.",
    "Brown Blight": "Noda coklat tidak beraturan akibat jamur Colletotrichum camelliae.",
    "Gray blight": "Bercak abu-abu hingga kehitaman karena jamur Pestalotiopsis theae.",
    "Red rust": "Bercak jingga kemerahan karena ganggang Cephaleuros parasiticus."
}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return {"error": "File yang diunggah bukan gambar"}

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Deteksi kualitas
        results_kualitas = model_kualitas(image)
        best_kualitas_conf = 0
        kualitas_label = None

        for box in results_kualitas[0].boxes:
            conf = float(box.conf)
            if conf > CONFIDENCE_THRESHOLD and conf > best_kualitas_conf:
                best_kualitas_conf = conf
                kualitas_label = results_kualitas[0].names[int(box.cls)]

        # Deteksi penyakit
        results_penyakit = model_penyakit(image)
        best_penyakit_conf = 0
        penyakit_label = None

        for box in results_penyakit[0].boxes:
            conf = float(box.conf)
            if conf > CONFIDENCE_THRESHOLD and conf > best_penyakit_conf:
                best_penyakit_conf = conf
                penyakit_label = results_penyakit[0].names[int(box.cls)]

        # LOGIKA AKHIR
        if not kualitas_label and not penyakit_label:
            return {
                "status": "Unknown",
                "message": "Gambar bukan daun teh atau ambil gambar lebih dekat.",
            }

        if best_kualitas_conf >= best_penyakit_conf:
            return {
                "status": "Healthy",
                "kualitas": kualitas_label,
                "confidence": round(best_kualitas_conf, 2)
            }
        else:
            return {
                "status": "Sick",
                "penyakit": penyakit_label,
                "confidence": round(best_penyakit_conf, 2),
                "deskripsi": penyakit_deskripsi.get(penyakit_label, "Deskripsi tidak ditemukan.")
            }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("API:app", host="127.0.0.1", port=8000, reload=True)
