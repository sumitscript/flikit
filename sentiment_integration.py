import pandas as pd
import numpy as np
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class SentimentPredictor:
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
    
    def load_model(self, model_path='sentiment_model.pkl'):
        try:
            with open(model_path, 'rb') as f:
                model_artifacts = pickle.load(f)
            
            self.model = model_artifacts.get('model_state_dict')
            self.tokenizer = model_artifacts.get('tokenizer')
            self.config = model_artifacts.get('config')
            
            if self.model is not None and self.tokenizer is not None:
                self.model_loaded = True
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict_sentiment(self, text):
        if not self.model_loaded:
            return self.rule_based_sentiment(text)
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = torch.max(predictions).item()
            
            label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            sentiment = label_mapping.get(predicted_class, 'Neutral')
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': predictions.numpy()[0].tolist()
            }
            
        except Exception as e:
            print(f"Error in model prediction: {str(e)}")
            return self.rule_based_sentiment(text)
    
    def rule_based_sentiment(self, text):
        if pd.isna(text):
            return {'sentiment': 'Neutral', 'confidence': 0.5, 'probabilities': [0.33, 0.34, 0.33]}
        
        text = str(text).lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best', 'perfect', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'poor', 'disgusting', 'pathetic']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            sentiment = 'Positive'
            confidence = min(0.8, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = 'Negative'
            confidence = min(0.8, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = 'Neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': [0.33, 0.34, 0.33]
        }
    
    def analyze_dataframe(self, df, text_column):
        results = []
        
        print(f"Analyzing sentiment for {len(df)} reviews...")
        
        for idx, text in enumerate(df[text_column]):
            if idx % 100 == 0:
                print(f"Processing review {idx + 1}/{len(df)}")
            
            result = self.predict_sentiment(text)
            results.append(result)
        
        df_result = df.copy()
        df_result['sentiment'] = [r['sentiment'] for r in results]
        df_result['sentiment_confidence'] = [r['confidence'] for r in results]
        
        return df_result
    
    def get_sentiment_distribution(self, df):
        if 'sentiment' not in df.columns:
            return {}
        
        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)
        
        distribution = {}
        for sentiment, count in sentiment_counts.items():
            distribution[sentiment] = {
                'count': count,
                'percentage': (count / total) * 100
            }
        
        return distribution

def main():
    predictor = SentimentPredictor()
    model_loaded = predictor.load_model()
    
    if model_loaded:
        print("Trained sentiment model loaded successfully")
    else:
        print("Could not load trained model, using fallback rule-based approach")
    
    return predictor

if __name__ == "__main__":
    predictor = main()