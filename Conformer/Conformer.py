# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import librosa
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

import gc

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define emotion labels
emotion_labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
label_to_id = {label: i for i, label in enumerate(emotion_labels)}
id_to_label = {i: label for i, label in enumerate(emotion_labels)}

# Audio Processing Configuration
MAX_AUDIO_LENGTH = 160000  # 10 seconds at 16kHz
TARGET_SAMPLE_RATE = 16000

# Caching mechanism for processed audio
class AudioCache:
    def __init__(self, cache_size=1000):
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get(self, path):
        if path in self.cache:
            self.cache_hits += 1
            return self.cache[path]
        self.cache_misses += 1
        return None
    
    def put(self, path, data):
        # Simple LRU-like behavior - clear cache if it gets too big
        if len(self.cache) >= self.cache_size:
            # Keep only 75% of the cache (remove oldest entries)
            keys_to_keep = list(self.cache.keys())[-int(self.cache_size * 0.75):]
            new_cache = {k: self.cache[k] for k in keys_to_keep}
            self.cache = new_cache
        self.cache[path] = data
        
    def stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total) * 100 if total > 0 else 0
        return f"Cache hits: {self.cache_hits}, misses: {self.cache_misses}, hit rate: {hit_rate:.2f}%, size: {len(self.cache)}"

# Global cache instance
audio_cache = AudioCache(cache_size=5000)  # Adjust based on your memory constraints

# Feature extraction is done once during dataset preparation
from transformers import Wav2Vec2FeatureExtractor
global_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

# Custom Dataset with optimizations
class EmotionAudioDataset(Dataset):
    def __init__(self, csv_file, transform=None, preload=False, preprocess_workers=4):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            preload (bool): Whether to preload all audio files (for small datasets).
            preprocess_workers (int): Number of workers for preprocessing (if preload=True).
        """
        self.data = pd.read_csv(csv_file)
        self.processor = global_processor  # Use the global processor
        self.transform = transform
        self.invalid_files = set()
        self.preload = preload
        self.preloaded_data = {}
        
        # Extract labels from paths
        self.labels = []
        for path in self.data.iloc[:, 0]:
            if 'anger' in path:
                self.labels.append(label_to_id['anger'])
            elif 'disgust' in path:
                self.labels.append(label_to_id['disgust'])
            elif 'fear' in path:
                self.labels.append(label_to_id['fear'])
            elif 'happy' in path:
                self.labels.append(label_to_id['happy'])
            elif 'neutral' in path:
                self.labels.append(label_to_id['neutral'])
            elif 'sad' in path:
                self.labels.append(label_to_id['sad'])
            else:
                self.labels.append(0)  # Default
        
        # Preload data if specified (for small datasets)
        if preload:
            print(f"Preloading dataset with {preprocess_workers} workers...")
            with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
                paths = self.data.iloc[:, 0].tolist()
                audio_data = list(tqdm(executor.map(self._load_audio, paths), total=len(paths)))
                self.preloaded_data = {path: data for path, data in zip(paths, audio_data) if data is not None}
            print(f"Preloaded {len(self.preloaded_data)} valid audio files out of {len(paths)}")
        
    def _load_audio(self, audio_path):
        try:
            # Check cache first
            cached_data = audio_cache.get(audio_path)
            if cached_data is not None:
                return cached_data
            
            # Load and preprocess audio
            import os
            if not os.path.exists(audio_path):
                return None
                
            # Use torchaudio for faster loading when possible
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample if needed
                if sample_rate != TARGET_SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
                    waveform = resampler(waveform)
            except:
                # Fallback to librosa
                waveform, sample_rate = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE)
                waveform = torch.tensor(waveform).unsqueeze(0)
            
            # Truncate or pad to standard length
            if waveform.shape[1] > MAX_AUDIO_LENGTH:
                waveform = waveform[:, :MAX_AUDIO_LENGTH]
            elif waveform.shape[1] < MAX_AUDIO_LENGTH:
                # Zero padding
                padding = torch.zeros(1, MAX_AUDIO_LENGTH - waveform.shape[1])
                waveform = torch.cat([waveform, padding], dim=1)
            
            result = waveform.squeeze(0)
            # Store in cache
            audio_cache.put(audio_path, result)
            return result
        except Exception as e:
            return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        try:
            audio_path = self.data.iloc[idx, 0]  # Assuming first column is the file path
            label_id = self.labels[idx]
            
            # Get preloaded or cached data if available
            if self.preload and audio_path in self.preloaded_data:
                waveform = self.preloaded_data[audio_path]
            else:
                waveform = self._load_audio(audio_path)
            
            # Handle invalid files
            if waveform is None:
                if audio_path not in self.invalid_files:
                    self.invalid_files.add(audio_path)
                # Return a dummy waveform
                waveform = torch.zeros(MAX_AUDIO_LENGTH)
            
            # Apply transform if specified
            if self.transform:
                waveform = self.transform(waveform)
            
            return {'waveform': waveform, 'label': label_id, 'path': audio_path}
                
        except Exception as e:
            # Return a dummy sample as fallback
            return {'waveform': torch.zeros(MAX_AUDIO_LENGTH), 'label': 0, 'path': 'error_path'}

# Batched processing for feature extraction
def batch_process_features(waveforms, sampling_rate=16000):
    """Process a batch of waveforms with the feature extractor"""
    with torch.no_grad():
        inputs = global_processor(waveforms, sampling_rate=sampling_rate, padding=True, return_tensors="pt")
    return inputs

# Optimized collate function for batching
def collate_fn(batch):
    """
    Optimized collate function that processes features in batch mode.
    """
    try:
        # Get waveforms and labels
        waveforms = [item['waveform'].numpy() for item in batch]
        labels = torch.tensor([item['label'] for item in batch])
        paths = [item['path'] for item in batch]
        
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_mels=80
        )

        waveforms_np = np.array(waveforms)   # convert list of numpy arrays to single ndarray once
        waveforms_tensor = torch.from_numpy(waveforms_np)  # shape [B, T]
        if waveforms_tensor.dim() == 2:
            waveforms_tensor = waveforms_tensor.unsqueeze(1)  # shape [B, 1, T]

        mel_specs = mel_transform(waveforms_tensor).squeeze(1).transpose(1, 2)  # [B, Time, 80]

        return {
            'input_values': mel_specs,  # Now shape: [B, T, 80]
            'labels': labels,
            'paths': paths
        }   

    except Exception as e:
        print(f"Error in collate_fn: {e}")
        # Return a minimal batch to prevent crash
        dummy_inputs = torch.zeros((len(batch), MAX_AUDIO_LENGTH))
        dummy_labels = torch.zeros(len(batch), dtype=torch.long)
        return {
            'input_values': dummy_inputs,
            'labels': dummy_labels,
            'paths': ["error"] * len(batch)
        }



import torch
import torch.nn as nn
import torch.nn.functional as F

class ConformerBlock(nn.Module):
    def __init__(
        self,
        encoder_dim,
        conv_expansion_factor,
        num_attention_heads,
        feed_forward_expansion_factor,
        feed_forward_dropout_p,
        attention_dropout_p,
        conv_dropout_p,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            encoder_dim,
            encoder_dim * conv_expansion_factor,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            encoder_dim * conv_expansion_factor,
            encoder_dim,
            kernel_size=3,
            padding=1,
        )
        self.conv_dropout = nn.Dropout(conv_dropout_p)
        self.norm1 = nn.LayerNorm(encoder_dim)
        self.norm2 = nn.LayerNorm(encoder_dim)
        self.norm3 = nn.LayerNorm(encoder_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout_p,
        )
        self.dropout = nn.Dropout(attention_dropout_p)
        self.ffn = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * feed_forward_expansion_factor),
            nn.ReLU(),
            nn.Dropout(feed_forward_dropout_p),
            nn.Linear(encoder_dim * feed_forward_expansion_factor, encoder_dim),
            nn.Dropout(feed_forward_dropout_p),
        )

    def forward(self, x):
        residual = x
        x_t = x.transpose(1, 2)
        x_t = F.relu(self.conv1(x_t))
        x_t = self.conv_dropout(x_t)
        x_t = F.relu(self.conv2(x_t))
        x_t = self.conv_dropout(x_t)
        x = self.norm1(x + x_t.transpose(1, 2))

        residual = x
        attn_output, _ = self.attention(
            x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)
        x = self.norm2(x + self.dropout(attn_output))

        residual = x
        ff_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class ConformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        encoder_dim,
        num_blocks,
        num_attention_heads,
        feed_forward_expansion_factor,
        conv_expansion_factor,
        input_dropout_p,
        feed_forward_dropout_p,
        attention_dropout_p,
        conv_dropout_p,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, encoder_dim)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.encoder_dim = encoder_dim
        self.num_blocks = num_blocks 

        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=encoder_dim,
                    conv_expansion_factor=conv_expansion_factor,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                )
                for _ in range(num_blocks)
            ]
        )
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




# Training function with mixed precision and optimization
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=3, 
                grad_accum_steps=2, mixed_precision=True, scheduler_type='cosine'):
    best_val_loss = float('inf')
    best_model_path = 'best_emotion_model_conformer.pth'
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None
    
    # Learning rate scheduler
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    elif scheduler_type == 'warmup':
        from transformers import get_cosine_schedule_with_warmup
        num_training_steps = len(train_loader) // grad_accum_steps * num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps for warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, 
                                                   num_training_steps=num_training_steps)
    
    print(f"Starting training with mixed precision: {mixed_precision}, gradient accumulation steps: {grad_accum_steps}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        valid_batches = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for i, batch in enumerate(pbar):
            try:
                input_values = batch['input_values'].to(device)
                labels = batch['labels'].to(device)
                
                # Mixed precision forward pass
                if mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(input_values)
                        loss = criterion(outputs, labels)
                        # Scale loss by gradient accumulation steps
                        loss = loss / grad_accum_steps
                    
                    # Scale gradients and backprop
                    scaler.scale(loss).backward()
                    
                    # Step every grad_accum_steps or at the end of an epoch
                    if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                        # Unscale before clipping
                        scaler.unscale_(optimizer)
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        # Optimizer step with scaler
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                        # Update LR scheduler if using warmup
                        if scheduler_type == 'warmup':
                            scheduler.step()
                else:
                    # Standard precision training
                    outputs = model(input_values)
                    loss = criterion(outputs, labels)
                    loss = loss / grad_accum_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Step every grad_accum_steps or at the end of an epoch
                    if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # Update LR scheduler if using warmup
                        if scheduler_type == 'warmup':
                            scheduler.step()
                
                # Track statistics (use the unscaled loss)
                running_loss += loss.item() * grad_accum_steps
                valid_batches += 1
                
                # Get predictions for accuracy calculation
                with torch.no_grad():
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item() * grad_accum_steps,
                    'batch_acc': (predicted == labels).sum().item() / len(labels)
                })
            except Exception as e:
                print(f"Error during training batch: {e}")
                continue
        
        if valid_batches == 0:
            print("No valid batches in epoch, skipping")
            continue
            
        epoch_loss = running_loss / valid_batches
        epoch_acc = accuracy_score(all_labels, all_preds) if len(all_preds) > 0 else 0
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        # Print cache stats
        print(audio_cache.stats())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        valid_val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                try:
                    input_values = batch['input_values'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Mixed precision validation (optional)
                    if mixed_precision and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            outputs = model(input_values)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(input_values)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    valid_val_batches += 1
                    _, predicted = torch.max(outputs.data, 1)
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())
                except Exception as e:
                    print(f"Error during validation batch: {e}")
                    continue
        
        if valid_val_batches == 0:
            print("No valid validation batches, skipping validation")
            continue
            
        val_epoch_loss = val_loss / valid_val_batches
        val_epoch_acc = accuracy_score(all_val_labels, all_val_preds) if len(all_val_preds) > 0 else 0
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.4f}")
        
        # Update learning rate schedulers
        if scheduler_type == 'cosine':
            scheduler.step()
        elif scheduler_type == 'plateau':
            scheduler.step(val_epoch_loss)
        
        # Save the best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"Model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training and validation metrics
    if len(train_losses) > 0 and len(val_losses) > 0:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()
    
    return best_model_path

# Evaluation function with optimizations
def evaluate_model(model, test_loader, criterion, mixed_precision=True):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Mixed precision evaluation (optional)
            if mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(input_values)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(input_values)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_labels, 
                yticklabels=emotion_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return test_loss, accuracy, f1, cm

# Main function with optimizations
def main():  
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True

    csv_file = "labeled_data.csv"
    batch_size = 16
    mixed_precision = True
    grad_accum_steps = 4
    num_workers = 4

    # Create dataset 
    dataset = EmotionAudioDataset(csv_file)  # Make sure dataset supports this

    # Split
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                             num_workers=num_workers, pin_memory=True)

    input_dim = 80  # Mel-spectrogram feature dimension
    model = ConformerModel(
        input_dim=input_dim,
        num_classes=len(emotion_labels),
        encoder_dim=256,
        num_blocks=12,
        num_attention_heads=4,
        feed_forward_expansion_factor=4,
        conv_expansion_factor=2,
        input_dropout_p=0.1,
        feed_forward_dropout_p=0.1,
        attention_dropout_p=0.1,
        conv_dropout_p=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Train
    best_model_path = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=100, patience=10,
        grad_accum_steps=grad_accum_steps,
        mixed_precision=mixed_precision,
        scheduler_type='warmup'
    )

    gc.collect()
    torch.cuda.empty_cache()

    model.load_state_dict(torch.load(best_model_path))

    # Evaluate
    test_loss, test_acc, test_f1, cm = evaluate_model(model, test_loader, criterion, mixed_precision=mixed_precision)

    print("Training and evaluation complete!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test F1 score: {test_f1:.4f}")

    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_mapping': id_to_label,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'model_config': {
            'encoder_dim': model.encoder_dim,
            'num_layers': model.num_blocks,
            'num_classes': len(emotion_labels)
        }
    }, 'emotion_classifier_conformer.pth')


    print("Final model saved to: emotion_classifier_conformer.pth")

if __name__ == "__main__":
    main()