from fastapi import FastAPI, File, UploadFile
from model.model import predict_emotion  
import shutil
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/classify-audio")
async def analyze_audio(file: UploadFile = File(...)):
    print("Received file:", file.filename)
    print(f"Content type: {file.content_type}")

    file_location = f"uploaded_file/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Need Preprocessing
    emotion = predict_emotion(file_location)
    os.remove(file_location)

    return {"emotion": emotion}


# C:\Users\MS24-1\Environments\environments\speech_recognition\Scripts\activate.bat
# uvicorn main:app --reload --port 8080