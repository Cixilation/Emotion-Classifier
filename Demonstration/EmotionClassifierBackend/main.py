from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import librosa
import soundfile as sf
import io
import torch
import torchaudio
import torch.nn.functional as F
import torch.nn as nn
from pydub import AudioSegment
import tempfile

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables - loaded once when server starts
model = None
device = None
mel_transform = None

# Emotion labels - must match training order
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

def extract_features(waveform, sr=16000):
    """Extract mel-spectrogram features exactly as in training"""
    if isinstance(waveform, torch.Tensor):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
    else:
        waveform = torch.tensor(waveform).unsqueeze(0)
    
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    
    # Ensure consistent padding/truncation as in training
    MAX_AUDIO_LENGTH = 160000  # 10 seconds at 16kHz
    if waveform.shape[1] > MAX_AUDIO_LENGTH:
        waveform = waveform[:, :MAX_AUDIO_LENGTH]
    elif waveform.shape[1] < MAX_AUDIO_LENGTH:
        padding = torch.zeros(1, MAX_AUDIO_LENGTH - waveform.shape[1])
        waveform = torch.cat([waveform, padding], dim=1)
    
    mel_spec = mel_transform(waveform)  # [1, n_mels, time]
    mel_spec = mel_spec.transpose(1, 2)  # [1, time, n_mels]
    return mel_spec.squeeze(0)           # [time, 80]

class ConformerBlock(nn.Module):
    def __init__(self, encoder_dim, conv_expansion_factor, num_attention_heads, 
                 feed_forward_expansion_factor, feed_forward_dropout_p, 
                 attention_dropout_p, conv_dropout_p):
        super().__init__()
        self.conv1 = nn.Conv1d(encoder_dim, encoder_dim * conv_expansion_factor, 
                              kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(encoder_dim * conv_expansion_factor, encoder_dim, 
                              kernel_size=3, padding=1)
        self.conv_dropout = nn.Dropout(conv_dropout_p)
        self.norm1 = nn.LayerNorm(encoder_dim)
        self.norm2 = nn.LayerNorm(encoder_dim)
        self.norm3 = nn.LayerNorm(encoder_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=encoder_dim, 
            num_heads=num_attention_heads, 
            dropout=attention_dropout_p
        )
        self.dropout = nn.Dropout(attention_dropout_p)
        self.ffn = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * feed_forward_expansion_factor),
            nn.ReLU(),
            nn.Dropout(feed_forward_dropout_p),
            nn.Linear(encoder_dim * feed_forward_expansion_factor, encoder_dim),
            nn.Dropout(feed_forward_dropout_p)
        )

    def forward(self, x):
        # Conv block
        residual = x
        x_t = x.transpose(1, 2)
        x_t = F.relu(self.conv1(x_t))
        x_t = self.conv_dropout(x_t)
        x_t = F.relu(self.conv2(x_t))
        x_t = self.conv_dropout(x_t)
        x = self.norm1(x + x_t.transpose(1, 2))

        # Attention block
        residual = x
        attn_output, _ = self.attention(
            x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)
        x = self.norm2(x + self.dropout(attn_output))

        # FFN block
        residual = x
        ff_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class ConformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, encoder_dim, num_blocks, 
                 num_attention_heads, feed_forward_expansion_factor, 
                 conv_expansion_factor, input_dropout_p, feed_forward_dropout_p, 
                 attention_dropout_p, conv_dropout_p):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, encoder_dim)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.encoder_dim = encoder_dim
        self.num_blocks = num_blocks
        
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                encoder_dim=encoder_dim,
                conv_expansion_factor=conv_expansion_factor,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                feed_forward_dropout_p=feed_forward_dropout_p,
                attention_dropout_p=attention_dropout_p,
                conv_dropout_p=conv_dropout_p
            ) for _ in range(num_blocks)
        ])
        self.global_norm = nn.LayerNorm(encoder_dim)
        self.fc_out = nn.Linear(encoder_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_dropout(x)
        for block in self.conformer_blocks:
            x = block(x)
        x = self.global_norm(x)
        x = x.mean(dim=1)
        return self.fc_out(x)

def load_model_on_startup():
    """Load the model when the server starts"""
    global model, device, mel_transform
    
    print("Loading emotion recognition model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize mel transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=80
    )
    
    try:
        # Load model checkpoint
        model_path = ".\EmotionClassifierBackend\model\emotion_classifier_conformer.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['model_config']
        
        print(f"Model config: {config}")
        
        # Create model
        model = ConformerModel(
            input_dim=80,
            num_classes=config['num_classes'],
            encoder_dim=config['encoder_dim'],
            num_blocks=config['num_layers'],
            num_attention_heads=4,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            input_dropout_p=0.1,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def predict_emotion_from_waveform(waveform, sr=16000):
    """
    Predict emotion from audio waveform
    """
    global model, device
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Extract features
        features = extract_features(waveform, sr).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            logits = model(features)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_id = torch.max(probabilities, dim=1)
            
            predicted_id = predicted_id.item()
            confidence = confidence.item()
        
        predicted_emotion = emotion_labels[predicted_id]
        
        # Get all probabilities for debugging
        all_probs = {
            emotion: float(prob) 
            for emotion, prob in zip(emotion_labels, probabilities[0])
        }
        
        return {
            "emotion": predicted_emotion,
            "confidence": confidence,
            "all_probabilities": all_probs
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def convert_webm_to_wav(file_content: bytes) -> tuple:
    """Convert WebM audio to WAV format using pydub"""
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
            temp_webm.write(file_content)
            temp_webm_path = temp_webm.name
        
        try:
            # Load WebM using pydub (which uses FFmpeg under the hood)
            audio = AudioSegment.from_file(temp_webm_path, format="webm")
            
            # Convert to mono and set sample rate to 16kHz
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Export to WAV format in memory
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            # Load the WAV data using librosa
            audio_data, sample_rate = librosa.load(wav_buffer, sr=16000, mono=True)
            
            print(f"Successfully converted WebM to WAV: length={len(audio_data)/16000:.2f}s, sr={sample_rate}")
            
            return audio_data, sample_rate
            
        finally:
            # Clean up temporary file
            os.unlink(temp_webm_path)
            
    except Exception as e:
        print(f"WebM conversion error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to convert WebM audio: {str(e)}")

def load_audio_from_bytes(file_content: bytes, content_type: str = None) -> tuple:
    """Load audio from bytes with format detection"""
    
    # Try WebM conversion first if content type suggests it
    if content_type and 'webm' in content_type.lower():
        try:
            return convert_webm_to_wav(file_content)
        except Exception as webm_error:
            print(f"WebM conversion failed: {webm_error}")
    
    # Try direct librosa loading (works for many formats)
    try:
        audio, sample_rate = librosa.load(io.BytesIO(file_content), sr=16000, mono=True)
        print(f"Direct librosa loading successful: length={len(audio)/16000:.2f}s")
        return audio, sample_rate
    except Exception as librosa_error:
        print(f"Direct librosa loading failed: {librosa_error}")
    
    # Try using pydub for other formats
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Try to load with pydub (auto-detects format)
            audio = AudioSegment.from_file(temp_file_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Convert to numpy array
            audio_data = audio.get_array_of_samples()
            audio_array = librosa.util.buf_to_float(audio_data, n_bytes=2)
            
            print(f"Pydub loading successful: length={len(audio_array)/16000:.2f}s")
            return audio_array, 16000
            
        finally:
            os.unlink(temp_file_path)
            
    except Exception as pydub_error:
        print(f"Pydub loading failed: {pydub_error}")
        raise HTTPException(status_code=500, detail="Unable to process audio format")

# Load model when server starts
@app.on_event("startup")
async def startup_event():
    load_model_on_startup()

@app.get("/")
async def root():
    return {"message": "Emotion Recognition API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.post("/classify-audio")
async def analyze_audio(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    print(f"Content type: {file.content_type}")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read file content
        file_content = await file.read()
        print(f"File size: {len(file_content)} bytes")
        
        # Load audio with format detection
        audio, sample_rate = load_audio_from_bytes(file_content, file.content_type)
        
        # Predict emotion
        result = predict_emotion_from_waveform(audio, sample_rate)
        
        print(f"Predicted emotion: {result['emotion']} (confidence: {result['confidence']:.3f})")
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# To run: uvicorn main:app --reload --port 8080
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)