import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SentimentConfig:
    def __init__(self, model_type='distilbert'):
        self.model_type = model_type
        self.model_name = self._get_model_name(model_type)
        self.max_length = 512
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.num_epochs = 3
        self.dropout_rate = 0.1
        self.num_classes = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _get_model_name(self, model_type):
        model_names = {
            'bert': 'bert-base-uncased',
            'distilbert': 'distilbert-base-uncased',
            'roberta': 'roberta-base'
        }
        return model_names.get(model_type, 'distilbert-base-uncased')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentClassifier(nn.Module):
    def __init__(self, config):
        super(SentimentClassifier, self).__init__()
        self.config = config
        self.transformer = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, config.num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

def prepare_data_splits(file_path='processed_airline_reviews.csv', config=None):
    if config is None:
        config = SentimentConfig()
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records for sentiment analysis")
    except FileNotFoundError:
        print("Creating sample data for demonstration...")
        df = pd.DataFrame({
            'processed_review': ['great flight experience'] * 100 + ['terrible service'] * 100 + ['average flight'] * 100,
            'rating_numeric': [8] * 100 + [2] * 100 + [5] * 100
        })
    
    def rating_to_sentiment(rating):
        if rating <= 4:
            return 0
        elif rating <= 6:
            return 1
        else:
            return 2
    
    df['sentiment_label'] = df['rating_numeric'].apply(rating_to_sentiment)
    
    texts = df['processed_review'].fillna('').tolist()
    labels = df['sentiment_label'].tolist()
    
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val

def setup_model_and_tokenizer(config):
    print(f"Setting up {config.model_type} model...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = SentimentClassifier(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Model loaded on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, device

def create_data_loaders(X_train, X_val, y_train, y_val, tokenizer, config):
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, config.max_length)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer, config.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Created data loaders with batch size: {config.batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader

def test_model_setup():
    print("Testing model setup...")
    
    config = SentimentConfig(model_type='distilbert')
    X_train, X_val, y_train, y_val = prepare_data_splits(config=config)
    model, tokenizer, device = setup_model_and_tokenizer(config)
    train_loader, val_loader = create_data_loaders(X_train, X_val, y_train, y_val, tokenizer, config)
    
    print("Model setup test completed successfully!")
    return model, tokenizer, train_loader, val_loader, config, device

if __name__ == "__main__":
    test_model_setup()