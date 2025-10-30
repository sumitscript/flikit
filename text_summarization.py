import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Only extractive summarization will work.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string

class TextSummarizer:
    
    def __init__(self, approach='extractive'):
        self.approach = approach
        self.download_nltk_data()
        
        if approach == 'transformer' and TRANSFORMERS_AVAILABLE:
            self.setup_transformer_models()
        elif approach == 'transformer' and not TRANSFORMERS_AVAILABLE:
            print("Transformers not available, falling back to extractive approach")
            self.approach = 'extractive'
        
        if self.approach == 'extractive':
            self.setup_extractive_components()
    
    def download_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def setup_transformer_models(self):
        try:
            print("Loading T5 model for summarization...")
            self.t5_summarizer = pipeline(
                "summarization",
                model="t5-small",
                tokenizer="t5-small",
                framework="pt"
            )
            
            print("Loading BART model for summarization...")
            self.bart_summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn",
                framework="pt"
            )
            
            self.transformer_ready = True
            print("Transformer models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading transformer models: {str(e)}")
            print("Falling back to extractive summarization")
            self.approach = 'extractive'
            self.transformer_ready = False
            self.setup_extractive_components()
    
    def setup_extractive_components(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        print("Extractive summarization components ready!")
    
    def preprocess_text_for_summarization(self, text: str) -> str:
        if pd.isna(text) or text == "":
            return ""
        text = str(text)
        text = re.sub(r'Trip Verified\s*\|\s*', '', text)
        text = re.sub(r'Not Verified\s*\|\s*', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[!.?]{3,}', '...', text)
        return text.strip()
    
    def transformer_summarize(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        if not self.transformer_ready:
            return self.extractive_summarize(text, num_sentences=3)
        
        try:
            clean_text = self.preprocess_text_for_summarization(text)
            
            if len(clean_text.split()) < 10:
                return clean_text
            
            try:
                summary = self.t5_summarizer(
                    clean_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]['summary_text']
            except:
                summary = self.bart_summarizer(
                    clean_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]['summary_text']
            
            return summary.strip()
            
        except Exception as e:
            print(f"Transformer summarization failed: {str(e)}")
            return self.extractive_summarize(text, num_sentences=3)
    
    def extractive_summarize(self, text: str, num_sentences: int = 3) -> str:
        try:
            clean_text = self.preprocess_text_for_summarization(text)
            
            if len(clean_text.split()) < 10:
                return clean_text
            
            sentences = sent_tokenize(clean_text)
            
            if len(sentences) <= num_sentences:
                return clean_text
            
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_sentence_indices.sort()
            summary_sentences = [sentences[i] for i in top_sentence_indices]
            summary = ' '.join(summary_sentences)
            
            return summary.strip()
            
        except Exception as e:
            print(f"Extractive summarization failed: {str(e)}")
            sentences = sent_tokenize(clean_text)
            return ' '.join(sentences[:min(num_sentences, len(sentences))])
    
    def generate_short_summary(self, text: str) -> str:
        if self.approach == 'transformer':
            return self.transformer_summarize(text, max_length=50, min_length=10)
        else:
            return self.extractive_summarize(text, num_sentences=1)
    
    def generate_detailed_summary(self, text: str) -> str:
        if self.approach == 'transformer':
            return self.transformer_summarize(text, max_length=150, min_length=50)
        else:
            return self.extractive_summarize(text, num_sentences=4)
    
    def summarize_feedback_batch(self, feedback_list: List[str]) -> Dict[str, List[str]]:
        short_summaries = []
        detailed_summaries = []
        
        print(f"Summarizing {len(feedback_list)} feedback texts...")
        
        for i, feedback in enumerate(feedback_list):
            if i % 10 == 0:
                print(f"Processing feedback {i+1}/{len(feedback_list)}")
            
            short_summary = self.generate_short_summary(feedback)
            detailed_summary = self.generate_detailed_summary(feedback)
            
            short_summaries.append(short_summary)
            detailed_summaries.append(detailed_summary)
        
        return {
            'short_summaries': short_summaries,
            'detailed_summaries': detailed_summaries
        }
    
    def summarize_by_sentiment(self, df: pd.DataFrame, sentiment_column: str = 'sentiment') -> Dict[str, str]:
        sentiment_summaries = {}
        
        for sentiment in df[sentiment_column].unique():
            sentiment_feedback = df[df[sentiment_column] == sentiment]['Review'].tolist()
            combined_text = ' '.join([str(text) for text in sentiment_feedback[:10]])
            summary = self.generate_detailed_summary(combined_text)
            sentiment_summaries[sentiment] = summary
        
        return sentiment_summaries
    
    def evaluate_summary_quality(self, original_text: str, summary: str) -> Dict[str, float]:
        try:
            original_words = len(original_text.split())
            summary_words = len(summary.split())
            compression_ratio = summary_words / original_words if original_words > 0 else 0
            
            original_words_set = set(original_text.lower().split())
            summary_words_set = set(summary.lower().split())
            coverage = len(summary_words_set.intersection(original_words_set)) / len(original_words_set) if original_words_set else 0
            
            summary_sentences = len(sent_tokenize(summary))
            
            return {
                'compression_ratio': compression_ratio,
                'coverage': coverage,
                'sentence_count': summary_sentences,
                'word_count': summary_words
            }
            
        except Exception as e:
            print(f"Error evaluating summary quality: {str(e)}")
            return {
                'compression_ratio': 0.0,
                'coverage': 0.0,
                'sentence_count': 0,
                'word_count': 0
            }

def load_processed_data(file_path: str = 'processed_airline_reviews.csv') -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} processed reviews")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def main():
    print("=== Text Summarization Engine Demo ===")
    df = load_processed_data()
    if df is None:
        return
    
    print("\nInitializing summarizer...")
    summarizer = TextSummarizer(approach='transformer')
    
    print("\n=== Sample Summarization Results ===")
    sample_reviews = df['Review'].head(3).tolist()
    
    for i, review in enumerate(sample_reviews):
        print(f"\n--- Review {i+1} ---")
        print(f"Original ({len(review.split())} words):")
        print(review[:200] + "..." if len(review) > 200 else review)
        
        short_summary = summarizer.generate_short_summary(review)
        detailed_summary = summarizer.generate_detailed_summary(review)
        
        print(f"\nShort Summary ({len(short_summary.split())} words):")
        print(short_summary)
        
        print(f"\nDetailed Summary ({len(detailed_summary.split())} words):")
        print(detailed_summary)
        
        quality_metrics = summarizer.evaluate_summary_quality(review, detailed_summary)
        print(f"\nQuality Metrics:")
        print(f"  Compression Ratio: {quality_metrics['compression_ratio']:.3f}")
        print(f"  Coverage: {quality_metrics['coverage']:.3f}")
        print(f"  Sentences: {quality_metrics['sentence_count']}")
        print("-" * 60)

if __name__ == "__main__":
    main()