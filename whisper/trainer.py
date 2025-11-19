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
import csv
import sacrebleu

class WhisperDataset(Dataset):
    def __init__(
        self,
        wav_scp: str,
        metadata_csv: str,
        model: Whisper,
    ):
        """
        Dataset for Whisper multitask fine-tuning.
        
        Parameters
        ----------
        wav_scp : str
            Path to wav.scp file with audio paths
        metadata_csv : str
            Path to CSV file with columns: audio_id, task, source_language, target_language, text
        model : Whisper
            The Whisper model
        """
        self.model = model
        self.dims = model.dims
        
        # Create a single tokenizer (language/task doesn't matter, we just need access to all tokens)
        self.tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language='en',
            task='transcribe'
        )
        
        # Read wav.scp
        with open(wav_scp, 'r') as f:
            audio_lines = [line.strip().split("\t") for line in f]
        
        # Create audio_id to filepath mapping
        self.audio_map = {id: path for id, path in audio_lines}
        
        # Read metadata CSV
        self.samples = []
        with open(metadata_csv, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                audio_id = row['audio_id']
                if audio_id in self.audio_map:
                    self.samples.append({
                        'audio_id': audio_id,
                        'audio_path': self.audio_map[audio_id],
                        'task': row['task'],
                        'source_language': row['source_language'],
                        'target_language': row['target_language'],
                        'text': row['text']
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        audio_id = sample['audio_id']
        audio_path = sample['audio_path']
        task = sample['task']
        source_language = sample['source_language']
        target_language = sample['target_language']
        text = sample['text']
        
        # Load and preprocess audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.dims.n_mels).to(torch.float)
        
        # Get special tokens using the single tokenizer
        sot_token = [self.tokenizer.sot]
        source_lang_token = [self.tokenizer.to_language_token(source_language)]
        target_lang_token = [self.tokenizer.to_language_token(target_language)]
        task_token = [self.tokenizer.transcribe if task == "transcribe" else self.tokenizer.translate]
        eot_token = [self.tokenizer.eot]
        
        # Encode the text
        text_tokens = self.tokenizer.encode(" " + text.strip())
        
        # Create input and target tokens with language ID
        # Input: SOT + target_lang + task + text tokens (for teacher forcing)
        input_tokens = torch.tensor(sot_token + target_lang_token + task_token + text_tokens)
        
        # Target: source_lang + task + text tokens + EOT (shifted right from input)
        target_tokens = torch.tensor(source_lang_token + task_token + text_tokens + eot_token)
        
        return {
            "mel": mel,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "audio_id": audio_id,
            "task": task,
            "source_language": source_language,
            "target_language": target_language,
            "reference_text": text.strip()
        }
    
    def collate_fn(self, batch):
        """
        Collate function for the DataLoader.
        Pads sequences in the batch to the same length.
        """
        mels = [item["mel"] for item in batch]
        input_tokens = [item["input_tokens"] for item in batch]
        target_tokens = [item["target_tokens"] for item in batch]
        audio_ids = [item["audio_id"] for item in batch]
        tasks = [item["task"] for item in batch]
        source_languages = [item["source_language"] for item in batch]
        target_languages = [item["target_language"] for item in batch]
        reference_texts = [item["reference_text"] for item in batch]
        
        mels = torch.stack(mels)
        
        # Pad token sequences (use -100 for input_tokens padding)
        input_tokens = pad_sequence(input_tokens, batch_first=True, padding_value=-100)
        target_tokens = pad_sequence(target_tokens, batch_first=True, padding_value=-100)
        
        return {
            "mel": mels,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "audio_ids": audio_ids,
            "tasks": tasks,
            "source_languages": source_languages,
            "target_languages": target_languages,
            "reference_texts": reference_texts
        }


class WhisperTrainer:
    def __init__(
        self,
        model: Whisper,
        optimizer: torch.optim.Optimizer,
        scheduler_step_strategy: Literal['val', 'step'],
        lr_scheduler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mixed_precision=True,
        batch_size=32,
        gradient_accumulation_steps=1,
        validation_fraction=0.05,
        validate_steps: int = 10000
    ):
        """
        Trainer for Whisper models with multitask support.
        
        Parameters
        ----------
        model : Whisper
            The Whisper model to fine-tune
        optimizer : torch.optim.Optimizer
            Optimizer to use for training
        scheduler_step_strategy : Literal['val', 'step']
            Strategy for learning rate scheduler step
        lr_scheduler : optional
            Learning rate scheduler
        device : str
            Device to use for training
        use_mixed_precision : bool
            Whether to use mixed precision training
        batch_size : int
            Batch size for training
        gradient_accumulation_steps : int
            Number of steps to accumulate gradients
        validation_fraction : float
            Fraction of data to use for validation
        validate_steps : int
            Number of steps after which to validate
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler_step_strategy = scheduler_step_strategy
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.validation_fraction = validation_fraction
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
    def validate(self, dataloader, validation_fraction=0.05):
        """
        Validate the model on validation data with separate metrics per task.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for validation data
        validation_fraction : float
            Fraction of data to use for validation (default is 0.05)
            
        Returns
        -------
        metrics : dict
            Dictionary of validation metrics per task
        """
        self.model.eval()
        
        # Calculate number of samples to use for validation
        total_samples = len(dataloader.dataset)
        val_samples = max(1, int(total_samples * validation_fraction))
        
        # Create a subset of the validation dataset
        indices = list(range(total_samples))
        subset_indices = indices[:val_samples]
        subset_dataset = Subset(dataloader.dataset, subset_indices)
        
        # Create a new dataloader for the subset
        subset_dataloader = DataLoader(
            subset_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=dataloader.dataset.collate_fn,
            num_workers=1,
            pin_memory=True if self.device == "cuda" else False
        )
        
        print(f"Validating on {val_samples} samples ({validation_fraction*100:.1f}% of validation set)")
        
        # Initialize metrics per task
        task_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(subset_dataloader, desc="Validating"):
                # Get batch data
                audio_ids = batch["audio_ids"]
                mel = batch["mel"].to(self.device)
                tasks = batch["tasks"]
                source_languages = batch["source_languages"]
                target_languages = batch["target_languages"]
                reference_texts = batch["reference_texts"]
                
                # Group by (task, target_language) tuple for efficient processing
                task_lang_indices = {}
                for i, (task, target_lang) in enumerate(zip(tasks, target_languages)):
                    key = (task, target_lang)
                    if key not in task_lang_indices:
                        task_lang_indices[key] = []
                    task_lang_indices[key].append(i)
                
                # Process each (task, target_language) group separately
                for (task, target_lang), indices in task_lang_indices.items():
                    if task not in task_metrics:
                        if task == "transcribe":
                            task_metrics[task] = {"wers": [], "cers": [], "scores": []}
                        else:  # translate
                            task_metrics[task] = {"bleus": [], "chrfs": []}
                    
                    # Get samples for this task-language combination
                    task_mel = mel[indices]
                    task_references = [reference_texts[i] for i in indices]
                    task_audio_ids = [audio_ids[i] for i in indices]
                    
                    # Perform decoding
                    options = whisper.DecodingOptions(
                        task=task,
                        language=target_lang,  # Now safe - all samples have same target language
                        temperature=0.0,
                        beam_size=1,
                        sample_len=256
                    )
                    
                    results = whisper.decode(self.model, task_mel, options)
                    
                    # Compute metrics
                    for i, (result, reference, audio_id) in enumerate(zip(results, task_references, task_audio_ids)):
                        hypothesis = result.text.strip()
                        reference = reference.strip()
                        
                        if task == "transcribe":
                            # Calculate WER and CER
                            hyp_words = hypothesis.split()
                            ref_words = reference.split()
                            
                            if len(ref_words) > 0:
                                wer = torchaudio.functional.edit_distance(ref_words, hyp_words) / len(ref_words)
                                task_metrics[task]["wers"].append(wer)
                            
                            if len(reference) > 0:
                                cer = torchaudio.functional.edit_distance(list(reference), list(hypothesis)) / len(reference)
                                task_metrics[task]["cers"].append(cer)
                            
                            if len(ref_words) > 0 and len(reference) > 0:
                                combined_error = 0.4 * task_metrics[task]["wers"][-1] + 0.6 * task_metrics[task]["cers"][-1]
                                score = (1 - combined_error) * 100
                                task_metrics[task]["scores"].append(score)
                                
                                print(f"[{task}] Audio ID: {audio_id}")
                                print(f"Hyp: {hypothesis}")
                                print(f"Ref: {reference}")
                                print(f"WER: {wer:.4f}, CER: {cer:.4f}, Score: {score:.2f}")
                                print("==" * 40)
                        
                        else:  # translate
                            # Calculate BLEU and chrF
                            bleu = sacrebleu.corpus_bleu(
                                [hypothesis], 
                                [[reference]], 
                                lowercase=True,
                                tokenize='13a'
                            )
                            
                            chrf = sacrebleu.corpus_chrf(
                                [hypothesis], 
                                [[reference]]
                            )
                            
                            task_metrics[task]["bleus"].append(bleu.score)
                            task_metrics[task]["chrfs"].append(chrf.score)
                            
                            print(f"[{task}] Audio ID: {audio_id}")
                            print(f"Hyp: {hypothesis}")
                            print(f"Ref: {reference}")
                            print(f"BLEU: {bleu.score:.2f}, chrF: {chrf.score:.2f}")
                            print("==" * 40)
                
                # Clean up
                del batch, mel
                torch.cuda.empty_cache()
                gc.collect()
        
        # Compute average metrics per task
        metrics = {}
        for task, task_data in task_metrics.items():
            if task == "transcribe":
                if task_data["wers"] and task_data["cers"]:
                    avg_wer = sum(task_data["wers"]) / len(task_data["wers"])
                    avg_cer = sum(task_data["cers"]) / len(task_data["cers"])
                    avg_score = sum(task_data["scores"]) / len(task_data["scores"])
                    
                    metrics[f"{task}_wer"] = avg_wer
                    metrics[f"{task}_cer"] = avg_cer
                    metrics[f"{task}_score"] = avg_score
            else:  # translate
                if task_data["bleus"]:
                    avg_bleu = sum(task_data["bleus"]) / len(task_data["bleus"])
                    avg_chrf = sum(task_data["chrfs"]) / len(task_data["chrfs"])
                    
                    metrics[f"{task}_bleu"] = avg_bleu
                    metrics[f"{task}_chrf"] = avg_chrf
        
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
            "train_loss": []
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
                    dataloader=val_dataloader,
                    validation_fraction=self.validation_fraction
                )
                
                # Step learning rate scheduler based on validation
                if self.lr_scheduler is not None and self.scheduler_step_strategy == 'val':
                    # Use first available score metric for scheduler
                    score_metric = val_metrics.get('transcribe_score') or val_metrics.get('translate_chrf', 0)
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(score_metric)
                    else:
                        self.lr_scheduler.step()
                
                # Print validation metrics
                print(f"Validation Metrics: {val_metrics}")
                
                # Log validation metrics
                if use_wandb:
                    wandb.log({
                        "step": global_step,
                        **{f"val_{k}": v for k, v in val_metrics.items()}
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
    start_step = checkpoint.get('global_step', 0)
    
    print(f"Resumed from step {start_step}")
    return start_step

def train_from_config(config, model):
    
    # Data paths
    data_root = config['data']['root']
    train_wav_scp = f"{data_root}/train/wav.scp"
    train_metadata = f"{data_root}/train/metadata.csv"
    val_wav_scp = f"{data_root}/dev/wav.scp"
    val_metadata = f"{data_root}/dev/metadata.csv"
    
    # Training settings
    batch_size = config['training']['batch_size']
    training_steps = config['training']['training_steps']
    learning_rate = config['training']['learning_rate']
    device = config['training']['device']
    resume_from = config['training'].get('resume_from', None)
    validate_steps = config['training'].get('validate_steps', 10000)
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    # Mixed precision settings
    use_mixed_precision = config['training'].get('mixed_precision', True)
    
    # Logging settings
    logging_interval = config['logging'].get('log_interval', 10)
    
    # WandB settings
    use_wandb = config.get('wandb', {}).get('enabled', False)
    wandb_project = config.get('wandb', {}).get('project', 'whisper-multitask')
    wandb_run_name = config.get('wandb', {}).get('run_name', None)
    wandb_tags = config.get('wandb', {}).get('tags', [])
    
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            tags=wandb_tags,
            config={
                'model': config['model']['name'],
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'training_steps': training_steps,
                'mixed_precision': use_mixed_precision,
            }
        )
    
    # Output settings
    output_dir = config['output']['dir']
    os.makedirs(output_dir, exist_ok=True)
    
    num_workers = config['training'].get('num_workers', 4)
    
    # Create datasets
    train_dataset = WhisperDataset(
        train_wav_scp, 
        train_metadata, 
        model
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
        val_metadata, 
        model
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
    
    if scheduler_config is not None and not isinstance(scheduler_config, dict):
        raise ValueError("Scheduler config must be a dictionary")
    
    lr_scheduler = WhisperTrainer.create_scheduler(
        optimizer, 
        scheduler_config,
        training_steps=training_steps,
    )
    
    assert lr_scheduler is not None, "Failed to create a valid learning rate scheduler"
    print(lr_scheduler.state_dict())
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_from:
        start_step = load_checkpoint(resume_from, model, optimizer, lr_scheduler)
        print(f"Resuming training from step {start_step + 1}")
    
    # Create trainer
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    
    args = parser.parse_args()

    # load config
    config = load_config(args.config)
    
    # Load model
    model_name = config['model']['name']
    model = whisper.load_model(model_name)

    torch.cuda.empty_cache()
    gc.collect()
    
    history = train_from_config(config, model)