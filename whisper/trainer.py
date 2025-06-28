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
from torch.amp import autocast, GradScaler
from typing import Literal
from torch.utils.data import Subset
import torchaudio

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
        language_id: bool = False
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
        language_id : bool
            Whether to include language ID tokens in the input (default is False)
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
        self.language_id = language_id

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
        audio_id, audio_path, text = self.samples[idx]
        
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
        
        if self.language_id:
            # Create input and target tokens
            # Input: SOT sequence followed by text tokens (for teacher forcing)
            input_tokens = torch.tensor(sot_token + target_lang_token + task_token + text_tokens)
            
            # Target: text tokens followed by EOT (shifted right from input)
            target_tokens = torch.tensor(source_lang_token + task_token + text_tokens + eot_token)
        else:
            # Create input and target tokens without language ID
            # Input: SOT sequence followed by text tokens (for teacher forcing)
            input_tokens = torch.tensor(sot_token + task_token + text_tokens)
            
            # Target: text tokens followed by EOT (shifted right from input)
            target_tokens = torch.tensor(task_token + text_tokens + eot_token)
        
        return {
            "mel": mel,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "audio_ids": audio_id,
            "reference_texts": text.strip()  # Store reference text for validation
        }


    def collate_fn(self, batch):
        """
        Collate function for the DataLoader.
        Pads sequences in the batch to the same length.
        """
        mels = [item["mel"] for item in batch]
        input_tokens = [item["input_tokens"] for item in batch]
        target_tokens = [item["target_tokens"] for item in batch]
        audio_ids = [item["audio_ids"] for item in batch]
        reference_texts = [item["reference_texts"] for item in batch]
        
        mels = torch.stack(mels)
        
        # Pad token sequences
        input_tokens = pad_sequence(input_tokens, batch_first=True, padding_value=self.tokenizer.eot)
        target_tokens = pad_sequence(target_tokens, batch_first=True, padding_value=self.tokenizer.eot)
        
        return {
            "mel": mels,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "audio_ids": audio_ids,
            "reference_texts": reference_texts
        }


class WhisperTrainer:
    def __init__(
        self,
        model: Whisper,
        optimizer: torch.optim.Optimizer,
        scheduler_step_strategy:Literal['val', 'step'],
        lr_scheduler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mixed_precision=True,  # Add mixed precision flag
        batch_size=32,
        gradient_accumulation_steps=1,
        validation_fraction=0.05,
        task: str = "transcribe",
        language: str = "en",
        validate_steps: int = 10000

    ):
        """
        Trainer for Whisper models.
        
        Parameters
        ----------
        model : Whisper
            The Whisper model to fine-tune
        optimizer : torch.optim.Optimizer
            Optimizer to use for training
        scheduler_step_strategy:Literal['val', 'step'],
            Strategy for learning rate scheduler step, either 'val' or 'step'
        lr_scheduler : 
            Learning rate scheduler (optional)
        device : str
            Device to use for training
        use_mixed_precision : bool
            Whether to use mixed precision training (only works on CUDA)
        batch_size : int
            Batch size for training
        gradient_accumulation_steps : int
            Number of steps to accumulate gradients before updating (default is 1)
        validation_fraction : float
            Fraction of data to use for validation (default is 0.1)
        task : str
            Task type ("transcribe" or "translate")
        language : str
            Target language code (e.g., "en" for English)
        validate_steps : int
            Number of steps after which to validate the model (default is 10000)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler_step_strategy = scheduler_step_strategy
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.validation_fraction = validation_fraction
        self.task = task
        self.language = language
        self.validate_steps = validate_steps
        
        # Mixed precision settings
        self.use_mixed_precision = use_mixed_precision and device.startswith("cuda")
        self.scaler = GradScaler(device="cuda") if self.use_mixed_precision else None
    
    @staticmethod
    def create_scheduler(optimizer, scheduler_config, training_steps):
        """
        Create a learning rate scheduler from config.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to attach the scheduler to
        scheduler_config : dict
            Scheduler configuration with 'name' and 'params'
        training_steps : int
            Total number of training steps (used for some schedulers)
            
        Returns
        -------
        scheduler : torch.optim.lr_scheduler._LRScheduler or None
            The created scheduler or None if not specified
        """
        if not scheduler_config:
            return None
        
        scheduler_name = scheduler_config.get('name')
        scheduler_params = scheduler_config.get('params', {})
        
        if not scheduler_name:
            return None
        
        # CosineAnnealingLR has T_max which is the total number of steps
        if scheduler_name == "CosineAnnealingLR":
            if 'T_max' not in scheduler_params:
                # Default T_max to total number of steps
                scheduler_params['T_max'] = training_steps

        # Get the scheduler class from torch.optim.lr_scheduler
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name, None)
        
        if scheduler_class is None:
            raise ValueError(f"Scheduler '{scheduler_name}' not found in torch.optim.lr_scheduler")
        
        return scheduler_class(optimizer, **scheduler_params)
        
    def train_step(self, batch, accumulation_step):
        """
        Perform a single training step.
        
        Parameters
        ----------
        batch : dict
            Batch of data from the DataLoader
        accumulation_step : int
            Current step within the accumulation cycle (0-based)
            
        Returns
        -------
        loss : float
            Loss value for this batch
        """
        self.model.train()
        # Only zero gradients at the start of accumulation cycle
        if accumulation_step == 0:
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
            ) / self.gradient_accumulation_steps

        # Mixed precision backward pass
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # Only update optimizer at the end of accumulation cycle
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
    def validate(self, dataloader, task_type, target_language, validation_fraction=0.05):
        """
        Validate the model on validation data.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for validation data
        task_type : str
            Task type ("transcribe" or "translate")
        target_language : str
            Target language code (e.g., "en" for English)
        validation_fraction : float
            Fraction of data to use for validation (default is 0.1)
            
        Returns
        -------
        metrics : dict
            Dictionary of validation metrics
        """
        self.model.eval()
        
        # Calculate number of samples to use for validation
        total_samples = len(dataloader.dataset)
        val_samples = max(1, int(total_samples * validation_fraction))
        
        # Create a subset of the validation dataset
        indices = list(range(total_samples))
        subset_indices = indices[:val_samples]
        subset_dataset = Subset(dataloader.dataset, subset_indices)
        
        # Create a new dataloader for the subset with batch_size=1 for decoding
        subset_dataloader = DataLoader(
            subset_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=dataloader.dataset.collate_fn,
            num_workers=1,
            pin_memory= True if self.device == "cuda" else False
        )
        
        print(f"Validating on {val_samples} samples ({validation_fraction*100:.1f}% of validation set)")
        
        # Initialize metrics lists
        if task_type == "transcribe":
            wers = []
            cers = []
            scores = []
        else:  # translate
            raise NotImplementedError("Translation task not implemented in this method")
        
        processed_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(subset_dataloader, desc="Validating with decoding"):
                # Get batch data
                audio_ids = batch["audio_ids"]
                mel = batch["mel"].to(self.device)
                reference_texts = batch["reference_texts"]
                
                # Perform decoding
                options = whisper.DecodingOptions(
                    task=task_type,
                    language=target_language,
                    temperature=0.0,  # Greedy decoding
                    beam_size=1,      # Greedy search
                    sample_len=256    # Maximum tokens to decode
                )
                
                # Decode using Whisper
                results = whisper.decode(self.model, mel, options)
                
                # Process results
                for i, (result, reference) in enumerate(zip(results, reference_texts)):
                    hypothesis = result.text.strip()
                    reference = reference.strip()
                    
                    if task_type == "transcribe":
                        # Calculate WER and CER
                        # Split into words for WER
                        hyp_words = hypothesis.split()
                        ref_words = reference.split()
                        
                        if len(ref_words) > 0:
                            wer = torchaudio.functional.edit_distance(ref_words, hyp_words) / len(ref_words)
                            wers.append(wer)
                        
                        # Character-level for CER
                        if len(reference) > 0:
                            cer = torchaudio.functional.edit_distance(list(reference), list(hypothesis)) / len(reference)
                            cers.append(cer)
                        
                        # Calculate combined score
                        if len(ref_words) > 0 and len(reference) > 0:
                            combined_error = 0.4 * wers[-1] + 0.6 * cers[-1]
                            score = (1 - combined_error) * 100
                            scores.append(score)
                            print(f"Audio ID: {audio_ids[i]}\nHyp: {hypothesis}\nRef: {reference}\nWER: {wer:.4f}, CER: {cer:.4f}, Score: {score:.2f}")
                            print("==" * 80)
                    else:  # translate
                        raise NotImplementedError("Translation task not implemented in this method")

                
                
                processed_samples += len(audio_ids)
                
                # Clean up
                del batch, mel
                torch.cuda.empty_cache()
                gc.collect()
                tqdm.write(f"Processed {processed_samples}/{val_samples} validation samples")
        
        metrics = {}
        
        if task_type == "transcribe" and wers and cers:
            avg_wer = sum(wers) / len(wers)
            avg_cer = sum(cers) / len(cers)
            avg_score = sum(scores) / len(scores)
            
            metrics.update({
                "wer": avg_wer,
                "cer": avg_cer,
                "score": avg_score,
            })
            
        elif task_type == "translate":
            raise NotImplementedError("Translation task not implemented in this method")
        
        return metrics
    
    def train(self, train_dataloader, val_dataloader, training_steps, log_interval=10, use_wandb=False, output_dir="output", start_step=0):
        """
        Train the model using step-based training.
        
        Parameters
        ----------
        train_dataloader : DataLoader
            DataLoader for training data
        val_dataloader : DataLoader
            DataLoader for validation data
        training_steps : int
            Total number of training steps
        log_interval : int
            Interval for logging training progress
        use_wandb : bool
            Whether to log metrics to Weights & Biases
        output_dir : str
            Directory to save model checkpoints
        start_step : int
            Step to start training from (for resuming)
            
        Returns
        -------
        history : dict
            Dictionary containing training history
        """
        history = {
            "train_loss": [],
            "val_score": []
        }

        # Print training configuration
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        print(f"Batch size: {self.batch_size}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        
        # Print mixed precision status
        if self.use_mixed_precision:
            print("Using mixed precision training (FP16)")
        else:
            print("Using full precision training (FP32)")
        
        # Create infinite iterator for training data
        train_iterator = iter(train_dataloader)
        
        global_step = start_step
        epoch = 0
        steps_in_current_epoch = 0
        
        print(f"Starting training from step {global_step} to {training_steps}")
        
        # Create progress bar for total steps
        pbar = tqdm(range(global_step, training_steps), 
                    desc="Training", 
                    initial=global_step, 
                    total=training_steps)
        
        while global_step < training_steps:
            # Accumulate gradients over multiple mini-batches
            accumulated_loss = 0.0
            for accumulation_step in range(self.gradient_accumulation_steps):
                try:
                    # Get next batch from iterator
                    batch = next(train_iterator)
                except StopIteration:
                    # End of epoch, reset iterator and increment epoch counter
                    train_iterator = iter(train_dataloader)
                    batch = next(train_iterator)
                    epoch += 1
                    steps_in_current_epoch = 0
                    print(f"\nStarted epoch {epoch}")
                
                # Perform training step with gradient accumulation
                loss = self.train_step(batch, accumulation_step)
                accumulated_loss += loss
            
            avg_loss = accumulated_loss / self.gradient_accumulation_steps

            global_step += 1
            steps_in_current_epoch += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'epoch': epoch,
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Store loss for history
            history["train_loss"].append(avg_loss)
            
            # Log batch-level metrics
            if use_wandb:
                wandb.log({
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            # Log training progress
            if global_step % log_interval == 0:
                print(f"\nStep {global_step}/{training_steps}, Epoch {epoch}, Loss: {avg_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Validation and checkpointing
            if global_step % self.validate_steps == 0 and val_dataloader is not None:
                print(f"\nValidating at step {global_step}...")
                
                # Validation
                val_metrics = self.validate(
                    val_dataloader, 
                    task_type=self.task,
                    target_language=self.language,
                    validation_fraction=self.validation_fraction
                )
                
                # Step learning rate scheduler based on validation
                if self.lr_scheduler is not None and self.scheduler_step_strategy == 'val':
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(val_metrics["score"])
                    else:
                        self.lr_scheduler.step()
                
                # Store validation metrics
                history["val_score"].append(val_metrics["score"])
                
                print(f"Validation Score: {val_metrics['score']:.4f}")
                
                # Log validation metrics
                if use_wandb:
                    wandb.log({
                        "step": global_step,
                        "val_score": val_metrics["score"],
                        "val_wer": val_metrics.get("wer", 0),
                        "val_cer": val_metrics.get("cer", 0)
                    })
                
                # Save checkpoint
                checkpoint_path = os.path.join(output_dir, f"checkpoint_{global_step}.pt")
                checkpoint = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "dims": self.model.dims.__dict__,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                
                # Add scheduler state if it exists
                if self.lr_scheduler is not None:
                    checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
                
                # Save the checkpoint
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Clean up batch to prevent memory accumulation
            del batch
            if global_step % 50 == 0:  # Clean up every 50 steps
                torch.cuda.empty_cache()
                gc.collect()
        
        # Close progress bar
        pbar.close()
        
        # Save final checkpoint
        final_checkpoint_path = os.path.join(output_dir, f"final_checkpoint_{global_step}.pt")
        final_checkpoint = {
            "global_step": global_step,
            "epoch": epoch,
            "dims": self.model.dims.__dict__,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        
        if self.lr_scheduler is not None:
            final_checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        torch.save(final_checkpoint, final_checkpoint_path)
        print(f"Saved final checkpoint to {final_checkpoint_path}")
        
        return history

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_checkpoint(checkpoint_path, model, optimizer=None, lr_scheduler=None):
    """
    Load a checkpoint and resume training state.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file
    model : Whisper
        The model to load state into
    optimizer : torch.optim.Optimizer, optional
        Optimizer to load state into
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler to load state into
        
    Returns
    -------
    start_step : int
        The step to resume from
    
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get the step to resume from
    start_step = checkpoint.get('step', 0)
    
    print(f"Resumed from step {start_step}")
    return start_step

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
    language_id = config['task'].get('language_id', False)
    
    # Training settings
    batch_size = config['training']['batch_size']
    training_steps = config['training']['training_steps']
    learning_rate = config['training']['learning_rate']
    device = config['training']['device']
    resume_from = config['training'].get('resume_from', None)
    validate_steps = config['training'].get('validate_steps', 10000)
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)

    
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
                'training_steps': training_steps,
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
        tokenizer=tokenizer,
        language_id=language_id
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
        tokenizer=tokenizer,
        language_id=language_id
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
    scheduler_step_strategy = config['training'].get('scheduler_step_strategy')
    scheduler_config = config['training'].get('scheduler_config', None)
    # ensure the scheduler config is a dict
    if scheduler_config is not None and not isinstance(scheduler_config, dict):
        raise ValueError("Scheduler config must be a dictionary")
    lr_scheduler = None
    lr_scheduler = WhisperTrainer.create_scheduler(
        optimizer, 
        scheduler_config,
        training_steps=training_steps,
    )
    # ensure it is a valid scheduler
    assert lr_scheduler is not None, "Failed to create a valid learning rate scheduler"
    print(lr_scheduler.state_dict())

    # Resume from checkpoint if specified
    start_step = 0
    if resume_from:
        start_step = load_checkpoint(resume_from, model, optimizer, lr_scheduler)
        print(f"Resuming training from step {start_step + 1}")
    
    # Create trainer with mixed precision support
    trainer = WhisperTrainer(
        model=model,
        optimizer=optimizer,
        scheduler_step_strategy=scheduler_step_strategy,
        lr_scheduler=lr_scheduler,
        device=device,
        use_mixed_precision=use_mixed_precision,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        validation_fraction=config['training'].get('validation_fraction', 0.05),
        task=task_type,
        language=target_language,
        validate_steps=validate_steps
        )
    
    # Train model
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        training_steps=training_steps,
        log_interval=logging_interval,
        use_wandb=use_wandb,
        output_dir=output_dir,
        start_step=start_step
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