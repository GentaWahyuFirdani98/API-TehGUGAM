from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load dua model
model_kualitas = YOLO("train21/weights/best.pt")   # T1-T4
model_penyakit = YOLO("train6/weights/best.pt")    # penyakit

CONFIDENCE_THRESHOLD = 0.3

# Deskripsi penyakit
penyakit_deskripsi = {
    "Blister Blight": "Si perusak daun muda. Jamur Exobasidium vexans menyebabkan gelembung kecil pada daun muda, terutama saat musim hujan. Jika dibiarkan, dapat menurunkan kualitas panen.",
    "Brown Blight": "Si Penyebar Noda Cokelat. Jamur Colletotrichum camelliae membentuk bercak coklat tak beraturan yang membuat daun rontok. Umumnya muncul di cuaca lembab dan minim sinar matahari.",
    "Gray blight": "Bercak abu-abu hingga cokelat kehitaman dikelilingi warna kuning yang membuat daun terlihat muram. Disebabkan jamur Pestalotiopsis theae dan menyerang daun yang lebih tua.",
    "Red rust": "Bercak jingga kemerahan mencolok akibat ganggang Cephaleuros parasiticus. Mengganggu jaringan tanaman dan mengurangi hasil panen. Sering muncul di tempat lembab dengan sirkulasi udara buruk."
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # ----------------------
        # Cek model kualitas (T1–T4)
        # ----------------------
        results_kualitas = model_kualitas(image)
        kualitas_boxes = results_kualitas[0].boxes
        kualitas_prediksi = []

        best_kualitas_conf = 0
        kualitas_label = None

        for box in kualitas_boxes:
            conf = float(box.conf)
            if conf > CONFIDENCE_THRESHOLD:
                if conf > best_kualitas_conf:
                    best_kualitas_conf = conf
                    kualitas_label = results_kualitas[0].names[int(box.cls)]

                kualitas_prediksi.append({
                    "kualitas": results_kualitas[0].names[int(box.cls)],
                    "confidence": round(conf, 2)
                })

        # ----------------------
        # Cek model penyakit
        # ----------------------
        results_penyakit = model_penyakit(image)
        penyakit_boxes = results_penyakit[0].boxes
        penyakit_prediksi = []

        best_penyakit_conf = 0
        penyakit_label = None

        for box in penyakit_boxes:
            conf = float(box.conf)
            if conf > CONFIDENCE_THRESHOLD:
                if conf > best_penyakit_conf:
                    best_penyakit_conf = conf
                    penyakit_label = results_penyakit[0].names[int(box.cls)]

                penyakit_prediksi.append({
                    "penyakit": results_penyakit[0].names[int(box.cls)],
                    "confidence": round(conf, 2)
                })

        # ----------------------
        # LOGIKA AKHIR
        # ----------------------
        if not kualitas_prediksi and not penyakit_prediksi:
            return {
                "status": "Unknown",
                "message": "Gambar bukan daun teh atau foto lebih dekat.",
            }

        if best_kualitas_conf >= best_penyakit_conf:
            return {
                "status": "Healthy",
                "kualitas": kualitas_label,
                "confidence": round(best_kualitas_conf, 2)
                
            }

        return {
            "status": "Sick",
            "penyakit": penyakit_label,
            "confidence": round(best_penyakit_conf, 2),
            "deskripsi": penyakit_deskripsi.get(penyakit_label, "Deskripsi tidak ditemukan.")
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("API:app", host="127.0.0.1", port=8000, reload=True)