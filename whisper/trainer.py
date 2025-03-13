import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import whisper
from whisper.model import Whisper
from whisper.tokenizer import get_tokenizer, Tokenizer
import os
import yaml
import argparse
import wandb
import gc
from tqdm import tqdm
from torch.amp import autocast, GradScaler  # Import for mixed precision

class WhisperDataset(Dataset):
    def __init__(
        self,
        wav_scp: str,
        transcript: str,
        model: Whisper,
        tokenizer: Tokenizer,
        task: str = "transcribe",
        source_language: str = None,
        target_language: str = None,
    ):
        """
        Dataset for Whisper fine-tuning.
        
        Parameters
        ----------
        wav_scp : str
            Path to wav.scp file with audio paths
        transcript : str
            Path to transcript file
        model : Whisper
            The Whisper model
        task : str
            Task to perform ("transcribe" or "translate")
        source_language : str
            Language code (e.g., "en" for English)
        target_language : str
            Language code (e.g., "en" for English)
        tokenizer : Tokenizer
            Tokenizer to use (required)
        """
        self.model = model
        self.dims = model.dims

        # Create tokenizer
        self.tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=target_language,
            task=task
        )
        
        self.task = task
        self.source_language = source_language
        self.target_language = target_language

        # Read wav.scp
        with open(wav_scp, 'r') as f:
            audio_lines = [(line.strip().split("\t")) for line in f]
        
        # Read transcript
        with open(transcript, 'r') as f:
            text_lines = [(line.strip().split("\t") )for line in f]

        assert len(audio_lines) == len(text_lines), "Number of audio files and transcripts do not match"
        # Create audio_id to filepath and text mappings
        audio_map = {id: path for id, path in audio_lines}
        text_map = {id: text for id, text in text_lines}
        
        # Ensure all audio files have corresponding transcripts
        self.samples = []
        for audio_id, audio_path in audio_map.items():
            if audio_id in text_map:
                self.samples.append((audio_id, audio_path, text_map[audio_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, audio_path, text = self.samples[idx]
        
        # Load and preprocess audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.dims.n_mels).to(torch.float)
        
        # Get special tokens
        sot_token = [self.tokenizer.sot] # Start of token sequence
        source_lang_token = [self.tokenizer.to_language_token(self.source_language)]
        target_lang_token = [self.tokenizer.to_language_token(self.target_language)]
        task_token = [self.tokenizer.transcribe if self.task == "transcribe" else self.tokenizer.translate]
        eot_token = [self.tokenizer.eot]
        
        # Encode the text
        text_tokens = self.tokenizer.encode(" " + text.strip())
        
        # Create input and target tokens
        # Input: SOT sequence followed by text tokens (for teacher forcing)
        input_tokens = torch.tensor(sot_token + target_lang_token + task_token + text_tokens)
        
        # Target: text tokens followed by EOT (shifted right from input)
        target_tokens = torch.tensor(source_lang_token + task_token + text_tokens + eot_token)
        
        return {
            "mel": mel,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens
        }


    def collate_fn(self, batch):
        """
        Collate function for the DataLoader.
        Pads sequences in the batch to the same length.
        """
        mels = [item["mel"] for item in batch]
        input_tokens = [item["input_tokens"] for item in batch]
        target_tokens = [item["target_tokens"] for item in batch]
        
        mels = torch.stack(mels)
        
        # Pad token sequences
        input_tokens = pad_sequence(input_tokens, batch_first=True, padding_value=self.tokenizer.eot)
        target_tokens = pad_sequence(target_tokens, batch_first=True, padding_value=self.tokenizer.eot)
        
        return {
            "mel": mels,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens
        }


class WhisperTrainer:
    def __init__(
        self,
        model: Whisper,
        optimizer: torch.optim.Optimizer,
        lr_scheduler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mixed_precision=True,  # Add mixed precision flag
    ):
        """
        Trainer for Whisper models.
        
        Parameters
        ----------
        model : Whisper
            The Whisper model to fine-tune
        optimizer : torch.optim.Optimizer
            Optimizer to use for training
        lr_scheduler : 
            Learning rate scheduler (optional)
        device : str
            Device to use for training
        use_mixed_precision : bool
            Whether to use mixed precision training (only works on CUDA)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        
        # Mixed precision settings
        self.use_mixed_precision = use_mixed_precision and device.startswith("cuda")
        self.scaler = GradScaler(device="cuda") if self.use_mixed_precision else None
        
    def train_step(self, batch):
        """
        Perform a single training step.
        
        Parameters
        ----------
        batch : dict
            Batch of data from the DataLoader
            
        Returns
        -------
        loss : float
            Loss value for this batch
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        mel = batch["mel"].to(self.device)
        input_tokens = batch["input_tokens"].to(self.device)
        target_tokens = batch["target_tokens"].to(self.device)
        
        # Mixed precision forward pass
        with autocast(enabled=self.use_mixed_precision, device_type="cuda"):
            # Get encoder output
            audio_features = self.model.encoder(mel)
            
            # Forward pass through the decoder
            logits = self.model.decoder(input_tokens, audio_features)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                target_tokens.view(-1),
                ignore_index=-100
            )
        
        # Mixed precision backward pass and optimization
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        return loss.item()
    
    def validate(self, dataloader):
        """
        Validate the model on validation data.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for validation data
            
        Returns
        -------
        metrics : dict
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct_tokens = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                mel = batch["mel"].to(self.device)
                input_tokens = batch["input_tokens"].to(self.device)
                target_tokens = batch["target_tokens"].to(self.device)
                
                # Mixed precision evaluation
                with autocast(enabled=self.use_mixed_precision, device_type="cuda"):
                    # Get encoder output
                    audio_features = self.model.encoder(mel)
                    
                    # Forward pass through the decoder
                    logits = self.model.decoder(input_tokens, audio_features)
                    
                    # Compute loss
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        target_tokens.view(-1),
                        ignore_index=-100
                    )
                
                total_loss += loss.item() * mel.size(0)
                
                # Compute accuracy
                pred_tokens = logits.argmax(dim=-1)
                mask = target_tokens != -100
                correct_tokens += (pred_tokens[mask] == target_tokens[mask]).sum().item()
                total_tokens += mask.sum().item()
            
            del batch
            torch.cuda.empty_cache()
            gc.collect()
        
        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def train(self, train_dataloader, val_dataloader, num_epochs, log_interval=10, use_wandb=False, output_dir="output"):
        """
        Train the model.
        
        Parameters
        ----------
        train_dataloader : DataLoader
            DataLoader for training data
        val_dataloader : DataLoader
            DataLoader for validation data
        num_epochs : int
            Number of epochs to train for
        log_interval : int
            Interval for logging training progress
        use_wandb : bool
            Whether to log metrics to Weights & Biases
            
        Returns
        -------
        history : dict
            Dictionary containing training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        # Print mixed precision status
        if self.use_mixed_precision:
            print("Using mixed precision training (FP16)")
        else:
            print("Using full precision training (FP32)")
        
        for epoch in range(num_epochs):
            # Training
            epoch_loss = 0
            for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                loss = self.train_step(batch)
                epoch_loss += loss
                
                # Log batch-level metrics
                if use_wandb:
                    wandb.log({
                        "step": epoch * len(train_dataloader) + i,
                        "train_loss": loss,
                        "learning_rate": self.optimizer.param_groups[0]['lr'] if self.lr_scheduler else self.optimizer.param_groups[0]['lr']
                    })
                
                if (i + 1) % log_interval == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_dataloader)}, Loss: {loss:.4f}")

                del batch
                torch.cuda.empty_cache()
                gc.collect()
            
            avg_train_loss = epoch_loss / len(train_dataloader)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            val_metrics = self.validate(val_dataloader)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Log epoch-level metrics
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"]
                })
            save_epoch = epoch + 1
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{save_epoch}.pt")
            
            # Create checkpoint with both model state and dimensions
            checkpoint = {
                "dims": self.model.dims.__dict__,  # Extract dimensions from the model
                "model_state_dict": self.model.state_dict()
            }
            
            # Save the checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Log checkpoint as artifact to wandb if enabled
            if use_wandb:
                artifact = wandb.Artifact(
                    name=f"model-checkpoint-{save_epoch}", 
                    type="model"
                )
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
            
        return history

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_from_config(config, model, tokenizer):
    
    # Data paths
    data_root = config['data']['root']
    train_wav_scp = f"{data_root}/train/wav.scp"
    train_transcript = f"{data_root}/train/text.tsv"
    val_wav_scp = f"{data_root}/dev/wav.scp"
    val_transcript = f"{data_root}/dev/text.tsv"
    
    # Task settings
    task_type = config['task']['type']
    source_language = config['task']['source_language']
    target_language = config['task']['target_language']
    
    # Training settings
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    device = config['training']['device']
    
    # Mixed precision settings (with default if not specified)
    use_mixed_precision = config['training'].get('mixed_precision', True)

    # Logging settings
    logging_interval = config['logging'].get('log_interval', 10)
    
    # WandB settings - look in both potential locations
    use_wandb = config.get('wandb', {}).get('enabled', False) or config.get('logging', {}).get('wandb', False)
    wandb_project = config.get('wandb', {}).get('project', config.get('logging', {}).get('wandb_project', 'whisper-finetuning'))
    wandb_run_name = config.get('wandb', {}).get('run_name', config.get('logging', {}).get('wand_run_name', None))
    wandb_tags = config.get('wandb', {}).get('tags', [])
    
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            tags=wandb_tags,
            config={
                'model': config['model']['name'],
                'task': task_type,
                'source_language': source_language,
                'target_language': target_language,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'mixed_precision': use_mixed_precision,
            }
        )
    
    # Output settings
    output_dir = config['output']['dir']
    # make output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    num_workers = config['training'].get('num_workers', 4)
    
    # Create datasets
    train_dataset = WhisperDataset(
        train_wav_scp, 
        train_transcript, 
        model,
        task=task_type,
        source_language=source_language,
        target_language=target_language,
        tokenizer=tokenizer
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True if device=="cuda" else False
    )
    
    val_dataset = WhisperDataset(
        val_wav_scp, 
        val_transcript, 
        model,
        task=task_type,
        source_language=source_language,
        target_language=target_language,
        tokenizer=tokenizer
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True if device=="cuda" else False
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_dataloader)
    )
    
    # Create trainer with mixed precision support
    trainer = WhisperTrainer(
        model,
        optimizer,
        lr_scheduler,
        device=device,
        use_mixed_precision=use_mixed_precision
    )
    
    # Train model
    history = trainer.train(
        train_dataloader,
        val_dataloader,
        num_epochs,
        log_interval=logging_interval,
        use_wandb=use_wandb,
        output_dir=output_dir
    )
    
    # Finish wandb run if enabled
    if use_wandb:
        wandb.finish()
    
    return history


def run_sanity_check(config_path, tokenizer):
    # Load configuration
    config = load_config(config_path)
    
    # Extract settings from config
    model_name = config['model']['name']
    
    # Data paths
    data_root = config['data']['root']
    train_wav_scp = f"{data_root}/train/wav.scp"
    train_transcript = f"{data_root}/train/text.tsv"
    val_wav_scp = f"{data_root}/dev/wav.scp"
    val_transcript = f"{data_root}/dev/text.tsv"
    
    # Task settings
    task_type = config['task']['type']
    source_language = config['task']['source_language']
    target_language = config['task']['target_language']
    
    device = config['training']['device']
    batch_size = config['training']['batch_size']
    # Load model
    model = whisper.load_model(model_name)

    num_workers = config['training'].get('num_workers', 4)
    
    # Create datasets
    train_dataset = WhisperDataset(
        train_wav_scp, 
        train_transcript, 
        model,
        task=task_type,
        source_language=source_language,
        target_language=target_language,
        tokenizer=tokenizer
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True if device=="cuda" else False
    )
    
    val_dataset = WhisperDataset(
        val_wav_scp, 
        val_transcript, 
        model,
        task=task_type,
        source_language=source_language,
        target_language=target_language,
        tokenizer=tokenizer

    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True if device=="cuda" else False
    )
    
    # Sanity check
    for batch in train_dataloader:
        assert "mel" in batch
        assert "input_tokens" in batch
        assert "target_tokens" in batch

        mel = batch["mel"]
        input_tokens = batch["input_tokens"]
        target_tokens = batch["target_tokens"]

        # decode text
        print("Input tokens:")
        print(tokenizer.decode(input_tokens[0]))
        print("Target tokens:")
        print(tokenizer.decode(target_tokens[0]))

        break
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    
    args = parser.parse_args()

    # load config
    config = load_config(args.config)
    
    # Load model
    model_name = config['model']['name']
    model = whisper.load_model(model_name)

    # Task settings
    task = config['task']['type']
    source_language = config['task']['source_language']
    target_language = config['task']['target_language']

    # create tokenizer
    tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=target_language,
            task=task
        )

    # run sanity check for data loading
    # run_sanity_check(args.config, tokenizer)
    torch.cuda.empty_cache()
    gc.collect()
    
    history = train_from_config(config, model, tokenizer)