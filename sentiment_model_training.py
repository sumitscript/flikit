import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from datetime import datetime
import os

from sentiment_model_setup import (
    SentimentConfig, 
    SentimentClassifier, 
    prepare_data_splits, 
    setup_model_and_tokenizer, 
    create_data_loaders
)

class SentimentTrainer:
    
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                predictions = torch.argmax(outputs, dim=1)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self):
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print("-" * 60)
        
        best_accuracy = 0
        best_model_state = None
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                best_model_state = self.model.state_dict().copy()
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print("-" * 60)
        
        self.model.load_state_dict(best_model_state)
        print(f"Training completed! Best validation accuracy: {best_accuracy:.4f}")
        
        return best_accuracy

def evaluate_model(trainer):
    print("Performing final model evaluation...")
    
    val_metrics = trainer.validate()
    
    labels = ['Negative', 'Neutral', 'Positive']
    print("\nClassification Report:")
    print(classification_report(
        val_metrics['labels'], 
        val_metrics['predictions'], 
        target_names=labels
    ))
    
    cm = confusion_matrix(val_metrics['labels'], val_metrics['predictions'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return val_metrics

def plot_training_history(trainer):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(trainer.train_losses) + 1)
    ax1.plot(epochs, trainer.train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, trainer.val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, trainer.val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_model_and_artifacts(model, tokenizer, config, val_metrics):
    model_artifacts = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': tokenizer,
        'validation_metrics': val_metrics,
        'label_mapping': {0: 'Negative', 1: 'Neutral', 2: 'Positive'},
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    print("Model saved as sentiment_model.pkl")
    
    with open('model_summary.txt', 'w') as f:
        f.write("Sentiment Classification Model Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Type: {config.model_type}\n")
        f.write(f"Base Model: {config.model_name}\n")
        f.write(f"Training Date: {model_artifacts['training_date']}\n")
        f.write(f"Number of Parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"  Accuracy: {val_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {val_metrics['precision']:.4f}\n")
        f.write(f"  Recall: {val_metrics['recall']:.4f}\n")
        f.write(f"  F1 Score: {val_metrics['f1']:.4f}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Max Length: {config.max_length}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        f.write(f"  Learning Rate: {config.learning_rate}\n")
        f.write(f"  Epochs: {config.num_epochs}\n")
        f.write(f"  Dropout Rate: {config.dropout_rate}\n")

def main():
    print("Starting Sentiment Classification Model Training")
    print("=" * 60)
    
    config = SentimentConfig(model_type='distilbert')
    
    print("Preparing data...")
    X_train, X_val, y_train, y_val = prepare_data_splits(config=config)
    
    print("Setting up model...")
    model, tokenizer, device = setup_model_and_tokenizer(config)
    
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        X_train, X_val, y_train, y_val, tokenizer, config
    )
    
    trainer = SentimentTrainer(model, train_loader, val_loader, config, device)
    
    best_accuracy = trainer.train()
    
    val_metrics = evaluate_model(trainer)
    
    plot_training_history(trainer)
    
    save_model_and_artifacts(model, tokenizer, config, val_metrics)
    
    print(f"\nTraining completed successfully!")
    print(f"Final validation accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Model saved as sentiment_model.pkl")
    print(f"Training plots saved as training_history.png and confusion_matrix.png")

if __name__ == "__main__":
    main()