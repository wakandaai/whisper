import torch
import whisper
import os
import glob
from collections import OrderedDict
import re
from typing import List, Dict, Union
from tqdm import tqdm

class ModelAverager:
    """
    Class for averaging Whisper model checkpoints to improve performance.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the model averager.
        
        Parameters
        ----------
        model_name : str
            Name of the base Whisper model (e.g., 'base', 'small', 'medium')
        """
        self.model_name = model_name
        self.base_model = whisper.load_model(model_name)
        
    def load_checkpoint_weights(self, checkpoint_path: str) -> Dict:
        """
        Load model weights from a checkpoint file.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file
            
        Returns
        -------
        dict
            Model state dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint['model_state_dict']
    
    def average_checkpoints(self, checkpoint_paths: List[str]) -> Dict:
        """
        Average multiple model checkpoints.
        
        Parameters
        ----------
        checkpoint_paths : List[str]
            List of paths to checkpoint files
            
        Returns
        -------
        dict
            Averaged model state dictionary
        """
        if not checkpoint_paths:
            raise ValueError("No checkpoint paths provided")
        
        print(f"Averaging {len(checkpoint_paths)} checkpoints:")
        for i, path in enumerate(checkpoint_paths):
            print(f"  {i+1}. {path}")
        
        # Load first checkpoint to initialize averaged weights
        averaged_weights = self.load_checkpoint_weights(checkpoint_paths[0])
        
        # Initialize all weights to zero for averaging
        for key in averaged_weights.keys():
            averaged_weights[key] = averaged_weights[key] * 0.0
        
        # Sum all checkpoint weights
        for checkpoint_path in tqdm(checkpoint_paths, desc="Loading checkpoints and accumulating"):
            weights = self.load_checkpoint_weights(checkpoint_path)
            for key in averaged_weights.keys():
                averaged_weights[key] += weights[key]
        
        # Divide by number of checkpoints to get average
        num_checkpoints = len(checkpoint_paths)
        for key in averaged_weights.keys():
            averaged_weights[key] /= num_checkpoints
        
        print(f"Successfully averaged {num_checkpoints} checkpoints")
        return averaged_weights
    
    def select_best_checkpoints(self, checkpoint_dir: str, 
                              metric: str = 'score', 
                              top_k: int = 5,
                              validation_log: str = None) -> List[str]:
        """
        Select the best k checkpoints based on validation metrics.
        
        Parameters
        ----------
        checkpoint_dir : str
            Directory containing checkpoint files
        metric : str
            Metric to use for selection ('score', 'wer', 'cer')
        top_k : int
            Number of top checkpoints to select
        validation_log : str, optional
            Path to validation log file (if metrics are logged separately)
            
        Returns
        -------
        List[str]
            List of paths to the best checkpoints
        """
        # Find all checkpoint files
        checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
        
        # Extract step numbers and sort
        checkpoint_info = []
        for checkpoint_file in checkpoint_files:
            # Extract step number from filename
            match = re.search(r'checkpoint_(\d+)\.pt', checkpoint_file)
            if match:
                step = int(match.group(1))
                checkpoint_info.append((step, checkpoint_file))
        
        # Sort by step number
        checkpoint_info.sort(key=lambda x: x[0])
        
        # If no validation log provided, select the last k checkpoints
        if validation_log is None:
            print(f"No validation log provided. Selecting last {top_k} checkpoints.")
            selected = checkpoint_info[-top_k:]
            return [checkpoint_file for _, checkpoint_file in selected]
        
        # TODO: Implement validation log parsing if you log metrics to a file
        # For now, return the last k checkpoints
        print(f"Selecting last {top_k} checkpoints (validation log parsing not implemented)")
        selected = checkpoint_info[-top_k:]
        return [checkpoint_file for _, checkpoint_file in selected]
    
    def select_checkpoints_by_steps(self, checkpoint_dir: str, 
                                   steps: List[int]) -> List[str]:
        """
        Select checkpoints by specific step numbers.
        
        Parameters
        ----------
        checkpoint_dir : str
            Directory containing checkpoint files
        steps : List[int]
            List of step numbers to select
            
        Returns
        -------
        List[str]
            List of paths to the selected checkpoints
        """
        selected_checkpoints = []
        
        for step in steps:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
            if os.path.exists(checkpoint_path):
                selected_checkpoints.append(checkpoint_path)
            else:
                print(f"Warning: Checkpoint for step {step} not found at {checkpoint_path}")
        
        return selected_checkpoints
    
    def create_averaged_model(self, checkpoint_paths: List[str]) -> whisper.model.Whisper:
        """
        Create a new model with averaged weights.
        
        Parameters
        ----------
        checkpoint_paths : List[str]
            List of paths to checkpoint files to average
            
        Returns
        -------
        whisper.model.Whisper
            Model with averaged weights
        """
        # Average the checkpoints
        averaged_weights = self.average_checkpoints(checkpoint_paths)
        
        # Create a new model instance
        averaged_model = whisper.load_model(self.model_name)
        
        # Load the averaged weights
        averaged_model.load_state_dict(averaged_weights)
        
        return averaged_model
    
    def save_averaged_checkpoint(self, checkpoint_paths: List[str], 
                                output_path: str,
                                include_metadata: bool = True):
        """
        Save an averaged checkpoint to disk.
        
        Parameters
        ----------
        checkpoint_paths : List[str]
            List of paths to checkpoint files to average
        output_path : str
            Path where the averaged checkpoint should be saved
        include_metadata : bool
            Whether to include training metadata in the saved checkpoint
        """
        # Average the weights
        averaged_weights = self.average_checkpoints(checkpoint_paths)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': averaged_weights,
            'averaged_from': checkpoint_paths,
            'num_averaged': len(checkpoint_paths)
        }
        
        # Include metadata from the last checkpoint if requested
        if include_metadata and checkpoint_paths:
            last_checkpoint = torch.load(checkpoint_paths[-1], map_location='cpu')
            # Copy relevant metadata
            for key in ['dims', 'global_step', 'epoch']:
                if key in last_checkpoint:
                    checkpoint_data[key] = last_checkpoint[key]
        
        # Save the averaged checkpoint
        torch.save(checkpoint_data, output_path)
        print(f"Saved averaged checkpoint to {output_path}")


def add_model_averaging_to_trainer(trainer_class):
    """
    Add model averaging functionality to your existing WhisperTrainer class.
    This is a mixin approach that extends your trainer.
    """
    
    def save_best_checkpoints(self, output_dir: str, max_checkpoints: int = 10):
        """
        Keep track of best checkpoints for later averaging.
        Add this to your validation logic.
        """
        if not hasattr(self, '_best_checkpoints'):
            self._best_checkpoints = []
        
        # This would be called during validation with the current score
        current_step = getattr(self, 'current_step', 0)
        current_score = getattr(self, 'current_val_score', 0.0)
        
        checkpoint_info = {
            'step': current_step,
            'score': current_score,
            'path': os.path.join(output_dir, f"checkpoint_{current_step}.pt")
        }
        
        self._best_checkpoints.append(checkpoint_info)
        
        # Sort by score (higher is better) and keep only top k
        self._best_checkpoints.sort(key=lambda x: x['score'], reverse=True)
        self._best_checkpoints = self._best_checkpoints[:max_checkpoints]
    
    def create_final_averaged_model(self, output_dir: str, top_k: int = 5):
        """
        Create final averaged model from best checkpoints.
        Call this at the end of training.
        """
        if not hasattr(self, '_best_checkpoints') or len(self._best_checkpoints) < 2:
            print("Not enough checkpoints for averaging")
            return None
        
        # Get paths of top k checkpoints
        top_checkpoints = self._best_checkpoints[:min(top_k, len(self._best_checkpoints))]
        checkpoint_paths = [ckpt['path'] for ckpt in top_checkpoints]
        
        # Create averager and save averaged model
        averager = ModelAverager(self.model_name)  # You'd need to store model_name
        averaged_checkpoint_path = os.path.join(output_dir, "averaged_model.pt")
        averager.save_averaged_checkpoint(checkpoint_paths, averaged_checkpoint_path)
        
        return averaged_checkpoint_path
    
    # Add methods to the trainer class
    trainer_class.save_best_checkpoints = save_best_checkpoints
    trainer_class.create_final_averaged_model = create_final_averaged_model
    
    return trainer_class


# Example usage functions
def average_last_n_checkpoints(checkpoint_dir: str, model_name: str, n: int = 5):
    """
    Simple function to average the last n checkpoints.
    
    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints
    model_name : str
        Base Whisper model name
    n : int
        Number of last checkpoints to average
        
    Returns
    -------
    whisper.model.Whisper
        Averaged model
    """
    averager = ModelAverager(model_name)
    
    # Find all checkpoints
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if len(checkpoint_files) < n:
        print(f"Warning: Only found {len(checkpoint_files)} checkpoints, using all of them")
        n = len(checkpoint_files)
    
    # Sort by step number and take the last n
    checkpoint_info = []
    for checkpoint_file in checkpoint_files:
        match = re.search(r'checkpoint_(\d+)\.pt', checkpoint_file)
        if match:
            step = int(match.group(1))
            checkpoint_info.append((step, checkpoint_file))
    
    checkpoint_info.sort(key=lambda x: x[0])
    selected_checkpoints = [checkpoint_file for _, checkpoint_file in checkpoint_info[-n:]]
    
    # Create averaged model
    averaged_model = averager.create_averaged_model(selected_checkpoints)
    
    # Save averaged checkpoint
    output_path = os.path.join(checkpoint_dir, "averaged_model.pt")
    averager.save_averaged_checkpoint(selected_checkpoints, output_path)
    
    return averaged_model


def average_specific_checkpoints(checkpoint_dir: str, model_name: str, steps: List[int]):
    """
    Average specific checkpoints by step numbers.
    
    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints
    model_name : str
        Base Whisper model name
    steps : List[int]
        List of step numbers to average
        
    Returns
    -------
    whisper.model.Whisper
        Averaged model
    """
    averager = ModelAverager(model_name)
    
    # Select checkpoints by steps
    selected_checkpoints = averager.select_checkpoints_by_steps(checkpoint_dir, steps)
    
    if not selected_checkpoints:
        raise ValueError("No valid checkpoints found for the specified steps")
    
    # Create averaged model
    averaged_model = averager.create_averaged_model(selected_checkpoints)
    
    # Save averaged checkpoint
    output_path = os.path.join(checkpoint_dir, f"averaged_model_steps_{'_'.join(map(str, steps))}.pt")
    averager.save_averaged_checkpoint(selected_checkpoints, output_path)
    
    return averaged_model


if __name__ == "__main__":
    checkpoint_dir = "results/whisper_large-v3_digital-umuganda_s2tt_ft"
    model_name = "large-v3"  # Base model name   
    averaged_model = average_specific_checkpoints(checkpoint_dir, model_name, steps=[7000, 8000])
    print("Created averaged model")
    # save the averaged model
    averaged_model_path = os.path.join(checkpoint_dir, "averaged_model.pt")
    averaged_model.save(averaged_model_path)