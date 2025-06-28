import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import whisper
from whisper.model import Whisper
from whisper.tokenizer import get_tokenizer, Tokenizer
from whisper.decoding import DecodingOptions, decode
import os
import yaml
import argparse
import wandb
import gc
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from typing import Literal, List, Tuple
from torch.utils.data import Subset
import torchaudio
import numpy as np
import json


class MetricRegressorModel(nn.Module):
    """
    Version that uses the decoder's final layer norm and adds a small projection head.
    Most similar to how Whisper normally works.
    """
    def __init__(self, whisper_model: Whisper, freeze_whisper: bool = True):
        super().__init__()
        self.whisper = whisper_model
        self.dims = whisper_model.dims
        
        # Freeze whisper parameters if requested
        if freeze_whisper:
            for param in self.whisper.parameters():
                param.requires_grad = False

        # Simple pooling layers
        self.audio_pool = nn.AdaptiveAvgPool1d(1)
        self.text_pool = nn.AdaptiveAvgPool1d(1)
        
        # Layer normalization (similar to what Whisper uses)
        self.audio_ln = nn.LayerNorm(self.dims.n_audio_state)
        self.text_ln = nn.LayerNorm(self.dims.n_text_state)
        
        # Regression head with residual connection
        self.regressor = nn.Sequential(
            nn.Linear(self.dims.n_audio_state + self.dims.n_text_state, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using simple but effective pooling.
        """
        # Get audio features from encoder
        audio_features = self.whisper.encoder(mel)  # (batch_size, n_audio_ctx, n_audio_state)
        
        # Forward pass through the decoder
        decoder_output = self.whisper.decoder(tokens, audio_features)  # (batch_size, seq_len, n_text_state)
        
        # Simple average pooling
        audio_pooled = self.audio_pool(audio_features.transpose(1, 2)).squeeze(-1)  # (batch_size, n_audio_state)
        text_pooled = self.text_pool(decoder_output.transpose(1, 2)).squeeze(-1)  # (batch_size, n_text_state)
        
        # Apply layer normalization
        audio_pooled = self.audio_ln(audio_pooled)
        text_pooled = self.text_ln(text_pooled)
        
        # Concatenate and predict
        combined_features = torch.cat([audio_pooled, text_pooled], dim=-1)
        metric_pred = self.regressor(combined_features)
        
        return metric_pred


class MetricRegressorDataset(Dataset):
    """
    Dataset for Metric regression training.
    Generates beam search candidates and computes their Metric scores.
    """
    def __init__(
        self,
        wav_scp: str,
        hypothesis: str,
        stats: str,
        whisper_model: Whisper,
        tokenizer: Tokenizer,
        task: str = "transcribe",
        language: str = "en"
    ):
        self.whisper_model = whisper_model
        self.tokenizer = tokenizer
        self.task = task
        self.language = language
        self.dims = whisper_model.dims
        
        # Read data files
        with open(wav_scp, 'r') as f:
            audio_lines = [line.strip().split("\t") for line in f]
        
        with open(hypothesis, 'r') as f:
            text_lines = [line.strip().split("\t") for line in f]

        with open(stats, 'r') as f:
            stats = [line.strip().split("\t") for line in f]
            stats = stats[1:]  # Skip header
        
        # Create mappings
        audio_map = {id: path for id, path in audio_lines}
        text_map = {id: text for id, text in text_lines}
        stats_map = {id: float(score) for id, _, _, score in stats}
        
        # Create samples
        self.samples = []
        for audio_id, audio_path in audio_map.items():
            if audio_id in text_map and audio_id in stats_map:
                self.samples.append((audio_id, audio_path, text_map[audio_id], stats_map[audio_id]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_id, audio_path, hypothesis_text, stats = self.samples[idx]
        
        # Load and preprocess audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.dims.n_mels).to(torch.float)
        
        # convert to tokens
        hypothesis_tokens = self.tokenizer.encode(hypothesis_text)        
        
        return {
            "mel": mel,
            "tokens": hypothesis_tokens,
            "metric_target": stats,
            "audio_id": audio_id
        }
    
    def collate_fn(self, batch):
        """Collate function"""
        mel = pad_sequence([item["mel"] for item in batch], batch_first=True)
        tokens = pad_sequence([torch.tensor(item["tokens"]) for item in batch], batch_first=True, padding_value=self.tokenizer.eot)
        metric_targets = torch.tensor([item["metric_target"] for item in batch], dtype=torch.float) 
        audio_ids = [item["audio_id"] for item in batch]
        return {
            "mels": mel,
            "tokens": tokens,
            "metric_targets": metric_targets,
            "audio_ids": audio_ids
        }


class MetricRegressorTrainer:
    def __init__(
        self,
        model: MetricRegressorModel,
        optimizer: torch.optim.Optimizer,
        scheduler_step_strategy: Literal['val', 'step'],
        lr_scheduler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mixed_precision=True,
        batch_size=16,
        gradient_accumulation_steps=1,
        validation_fraction=0.1
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler_step_strategy = scheduler_step_strategy
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.validation_fraction = validation_fraction
        
        # Mixed precision
        self.use_mixed_precision = use_mixed_precision and device.startswith("cuda")
        self.scaler = GradScaler(device="cuda") if self.use_mixed_precision else None
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    @staticmethod
    def create_scheduler(optimizer, scheduler_config, training_steps):
        """Create learning rate scheduler from config (same as original)."""
        if not scheduler_config:
            return None
        
        scheduler_name = scheduler_config.get('name')
        scheduler_params = scheduler_config.get('params', {})
        
        if not scheduler_name:
            return None
        
        if scheduler_name == "CosineAnnealingLR":
            if 'T_max' not in scheduler_params:
                scheduler_params['T_max'] = training_steps
        
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name, None)
        
        if scheduler_class is None:
            raise ValueError(f"Scheduler '{scheduler_name}' not found")
        
        return scheduler_class(optimizer, **scheduler_params)
    
    def train_step(self, batch, accumulation_step):
        """Perform a single training step."""
        if accumulation_step == 0:
            self.optimizer.zero_grad()
        
        mel = batch["mels"].to(self.device)
        tokens = batch["tokens"].to(self.device)
        metric_targets = batch["metric_targets"].to(self.device)
        
        with autocast(enabled=self.use_mixed_precision, device_type="cuda"):
            metric_pred = self.model(mel, tokens).squeeze(-1)
            loss = self.criterion(metric_pred, metric_targets) / self.gradient_accumulation_steps
        
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if accumulation_step == self.gradient_accumulation_steps - 1:
            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            if self.lr_scheduler is not None and self.scheduler_step_strategy == 'step':
                self.lr_scheduler.step()
        
        return loss.item() * self.gradient_accumulation_steps
    
    @torch.no_grad()
    def validate(self, dataloader, validation_fraction=0.1):
        """Validate the model."""
        self.model.eval()
        
        total_samples = len(dataloader.dataset)
        val_samples = max(1, int(total_samples * validation_fraction))
        
        subset_indices = list(range(val_samples))
        subset_dataset = Subset(dataloader.dataset, subset_indices)
        
        subset_dataloader = DataLoader(
            subset_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=dataloader.dataset.collate_fn,
            num_workers=1,
            pin_memory=True if self.device == "cuda" else False
        )
        
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch in tqdm(subset_dataloader, desc="Validating"):
            mel = batch["mel"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            metric_targets = batch["metric_targets"].to(self.device)
            
            metric_pred = self.model(mel, tokens).squeeze(-1)
            
            loss = self.criterion(metric_pred, metric_targets)
            mae = F.l1_loss(metric_pred, metric_targets)
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return {
            "val_loss": avg_loss,
            "val_mae": avg_mae
        }
    
    def train(self, train_dataloader, val_dataloader, training_steps, log_interval=10, 
              use_wandb=False, output_dir="output", start_step=0, validate_steps=1000):
        """Train the Metric regressor."""
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": []
        }
        
        print(f"Training Metric Regressor")
        print(f"Batch size: {self.batch_size}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        
        train_iterator = iter(train_dataloader)
        global_step = start_step
        
        pbar = tqdm(range(global_step, training_steps), 
                   desc="Training Metric Regressor", 
                   initial=global_step, 
                   total=training_steps)
        
        while global_step < training_steps:
            accumulated_loss = 0.0
            
            for accumulation_step in range(self.gradient_accumulation_steps):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_dataloader)
                    batch = next(train_iterator)
                
                loss = self.train_step(batch, accumulation_step)
                accumulated_loss += loss
            
            avg_loss = accumulated_loss / self.gradient_accumulation_steps
            global_step += 1
            
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            history["train_loss"].append(avg_loss)
            
            if use_wandb:
                wandb.log({
                    "step": global_step,
                    "train_loss": avg_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            if global_step % log_interval == 0:
                print(f"\nStep {global_step}/{training_steps}, Loss: {avg_loss:.4f}")
            
            # Validation
            if global_step % validate_steps == 0 and val_dataloader is not None:
                print(f"\nValidating at step {global_step}...")
                
                val_metrics = self.validate(val_dataloader, self.validation_fraction)
                
                if self.lr_scheduler is not None and self.scheduler_step_strategy == 'val':
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(val_metrics["val_loss"])
                    else:
                        self.lr_scheduler.step()
                
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_mae"].append(val_metrics["val_mae"])
                
                print(f"Val Loss: {val_metrics['val_loss']:.4f}, Val MAE: {val_metrics['val_mae']:.4f}")
                
                if use_wandb:
                    wandb.log({
                        "step": global_step,
                        "val_loss": val_metrics["val_loss"],
                        "val_mae": val_metrics["val_mae"]
                    })
                
                # Save checkpoint
                checkpoint_path = os.path.join(output_dir, f"metric_regressor_checkpoint_{global_step}.pt")
                checkpoint = {
                    "global_step": global_step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                
                if self.lr_scheduler is not None:
                    checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
                
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            del batch
            if global_step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        pbar.close()
        
        # Save final checkpoint
        final_checkpoint_path = os.path.join(output_dir, f"metric_regressor_final_{global_step}.pt")
        final_checkpoint = {
            "global_step": global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        
        if self.lr_scheduler is not None:
            final_checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        torch.save(final_checkpoint, final_checkpoint_path)
        print(f"Saved final checkpoint to {final_checkpoint_path}")
        
        return history


def train_metric_regressor_from_config(config_path):
    """Train Metric regressor from config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load base Whisper model
    whisper_model_name = config['base_model']['name']  # e.g., "base"
    whisper_model = whisper.load_model(whisper_model_name)
    
    # Load fine-tuned checkpoint if specified
    if 'checkpoint_path' in config['base_model']:
        checkpoint = torch.load(config['base_model']['checkpoint_path'])
        whisper_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded fine-tuned model from {config['base_model']['checkpoint_path']}")
    
    # Create tokenizer
    task = config['task']['type']
    language = config['task']['target_language']
    tokenizer = get_tokenizer(
        whisper_model.is_multilingual,
        num_languages=whisper_model.num_languages,
        language=language,
        task=task
    )
    
    # Create Metric regressor model
    freeze_whisper = config['model'].get('freeze_whisper', True)
    model = MetricRegressorModel(whisper_model, freeze_whisper=freeze_whisper)
    
    # Data paths
    data_root = config['data']['root']
    train_wav_scp = f"{data_root}/train/wav.scp"
    train_transcript = f"{data_root}/train/text.tsv"
    val_wav_scp = f"{data_root}/dev/wav.scp"
    val_transcript = f"{data_root}/dev/text.tsv"
    
    # Training settings
    batch_size = config['training']['batch_size']
    training_steps = config['training']['training_steps']
    learning_rate = config['training']['learning_rate']
    device = config['training']['device']
    validate_steps = config['training'].get('validate_steps', 1000)
    beam_size = config['training'].get('beam_size', 5)
    max_samples = config['training'].get('max_samples', None)
    
    # Create datasets
    train_dataset = MetricRegressorDataset(
        train_wav_scp, train_transcript, whisper_model, tokenizer,
        task=task, language=language, beam_size=beam_size, max_samples=max_samples
    )
    
    val_dataset = MetricRegressorDataset(
        val_wav_scp, val_transcript, whisper_model, tokenizer,
        task=task, language=language, beam_size=beam_size, max_samples=max_samples//5 if max_samples else None
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=train_dataset.collate_fn, num_workers=2
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=val_dataset.collate_fn, num_workers=2
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    
    # Create scheduler
    scheduler_config = config['training'].get('scheduler_config')
    scheduler_step_strategy = config['training'].get('scheduler_step_strategy', 'step')
    lr_scheduler = MetricRegressorTrainer.create_scheduler(
        optimizer, scheduler_config, training_steps
    )
    
    # Create trainer
    trainer = MetricRegressorTrainer(
        model=model,
        optimizer=optimizer,
        scheduler_step_strategy=scheduler_step_strategy,
        lr_scheduler=lr_scheduler,
        device=device,
        batch_size=batch_size,
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1)
    )
    
    # Train
    output_dir = config['output']['dir']
    os.makedirs(output_dir, exist_ok=True)
    
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        training_steps=training_steps,
        validate_steps=validate_steps,
        output_dir=output_dir
    )
    
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Metric Regressor")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    
    args = parser.parse_args()
    
    history = train_metric_regressor_from_config(args.config)