def preprocess_audio(file_path, sample_rate=16000, n_mels=512):
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

    waveform = waveform.mean(dim=0)  
    waveform = waveform.numpy()

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalisasi
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)

    return torch.tensor(mel_spec_db).unsqueeze(0)  