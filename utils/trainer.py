"""
Training utilities with early stopping and checkpointing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    """
    
    def __init__(self, patience=10, min_delta=0.0, mode='max', restore_best_weights=True):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'max' for metrics to maximize (e.g., accuracy), 'min' for minimize (e.g., loss)
            restore_best_weights (bool): Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        if mode == 'max':
            self.best_score = -np.inf
        else:
            self.best_score = np.inf
    
    def __call__(self, score, model):
        """
        Check if training should stop.
        
        Args:
            score (float): Current validation score
            model: PyTorch model
        
        Returns:
            bool: True if training should stop
        """
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore_weights(self, model):
        """Restore best weights to model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ModelTrainer:
    """
    Model trainer with validation, early stopping, and checkpointing.
    """
    
    def __init__(self, model, device, criterion=None, optimizer=None, 
                 scheduler=None, verbose=True):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: torch.device (cuda or cpu)
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (will be created if None)
            scheduler: Learning rate scheduler (optional)
            verbose (bool): Print training progress
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, max_epochs=100, 
              early_stopping=None, save_path=None):
        """
        Train model with validation and early stopping.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            max_epochs (int): Maximum number of epochs
            early_stopping: EarlyStopping instance (optional)
            save_path (str): Path to save best model checkpoint
        
        Returns:
            dict: Training history
        """
        if self.optimizer is None:
            raise ValueError("Optimizer must be set before training")
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = -np.inf
        
        for epoch in range(max_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_path is not None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                    }, save_path)
            
            # Early stopping
            if early_stopping is not None:
                if early_stopping(val_acc, self.model):
                    if self.verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    if early_stopping.restore_best_weights:
                        early_stopping.restore_weights(self.model)
                    break
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{max_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%")
        
        return history
    
    def evaluate(self, test_loader):
        """Evaluate model on test set."""
        test_loss, test_acc = self.validate(test_loader)
        return test_loss, test_acc

