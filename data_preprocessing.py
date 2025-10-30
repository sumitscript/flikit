import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        
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
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        try:
            nltk.data.find('corpora/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng')
    
    def load_dataset(self, file_path='Indian_Domestic_Airline.csv'):
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with {len(df)} records")
            print(f"Columns: {list(df.columns)}")
            return df
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return None
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None
    
    def explore_data_structure(self, df):
        print("=== Dataset Structure ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\n=== Data Types ===")
        print(df.dtypes)
        print("\n=== Missing Values ===")
        print(df.isnull().sum())
        print("\n=== Basic Statistics ===")
        print(df.describe(include='all'))
        print("\n=== Sample Records ===")
        print(df.head())
        duplicates = df.duplicated().sum()
        print(f"\n=== Duplicate Records: {duplicates} ===")
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': duplicates
        }
    
    def remove_duplicates(self, df):
        initial_count = len(df)
        df_clean = df.drop_duplicates()
        final_count = len(df_clean)
        removed_count = initial_count - final_count
        print(f"Removed {removed_count} duplicate records")
        print(f"Dataset size: {initial_count} -> {final_count}")
        return df_clean
    
    def clean_special_characters(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().strip('"').strip("'").strip()
        text = re.sub(r'[!.?]{3,}', '...', text)
        return text
    
    def get_wordnet_pos(self, word):
        try:
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        except:
            return wordnet.NOUN
    
    def tokenize_and_lemmatize(self, text):
        if pd.isna(text) or text == "":
            return []
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [token for token in tokens if len(token) > 1]
        lemmatized_tokens = []
        for token in tokens:
            pos = self.get_wordnet_pos(token)
            lemmatized_token = self.lemmatizer.lemmatize(token, pos)
            lemmatized_tokens.append(lemmatized_token)
        return lemmatized_tokens
    
    def handle_missing_data(self, df):
        df_clean = df.copy()
        if 'Review' in df_clean.columns:
            initial_count = len(df_clean)
            df_clean = df_clean.dropna(subset=['Review'])
            removed_count = initial_count - len(df_clean)
            print(f"Removed {removed_count} rows with missing reviews")
        if 'Rating - 10' in df_clean.columns:
            median_rating = df_clean['Rating - 10'].median()
            missing_ratings = df_clean['Rating - 10'].isnull().sum()
            df_clean['Rating - 10'].fillna(median_rating, inplace=True)
            print(f"Filled {missing_ratings} missing ratings with median value: {median_rating}")
        if 'Title' in df_clean.columns:
            missing_titles = df_clean['Title'].isnull().sum()
            df_clean['Title'].fillna('', inplace=True)
            print(f"Filled {missing_titles} missing titles with empty string")
        if 'Name' in df_clean.columns:
            missing_names = df_clean['Name'].isnull().sum()
            df_clean['Name'].fillna('Anonymous', inplace=True)
            print(f"Filled {missing_names} missing names with 'Anonymous'")
        if 'Date' in df_clean.columns:
            missing_dates = df_clean['Date'].isnull().sum()
            df_clean['Date'].fillna('Unknown', inplace=True)
            print(f"Filled {missing_dates} missing dates with 'Unknown'")
        if 'Recommond' in df_clean.columns:
            missing_recomm = df_clean['Recommond'].isnull().sum()
            df_clean['Recommond'].fillna('unknown', inplace=True)
            print(f"Filled {missing_recomm} missing recommendations with 'unknown'")
        return df_clean
    
    def preprocess_dataset(self, file_path='Indian_Domestic_Airline.csv'):
        print("=== Starting Data Preprocessing Pipeline ===")
        df = self.load_dataset(file_path)
        if df is None:
            return None
        print("\n=== Step 1: Data Exploration ===")
        data_info = self.explore_data_structure(df)
        print("\n=== Step 2: Removing Duplicates ===")
        df = self.remove_duplicates(df)
        print("\n=== Step 3: Handling Missing Data ===")
        df = self.handle_missing_data(df)
        print("\n=== Step 4: Cleaning Special Characters ===")
        text_columns = ['Review', 'Title']
        for col in text_columns:
            if col in df.columns:
                print(f"Cleaning special characters in {col} column...")
                df[col] = df[col].apply(self.clean_special_characters)
        print("\n=== Step 5: Tokenization and Lemmatization ===")
        if 'Review' in df.columns:
            print("Processing review text...")
            df['processed_review'] = df['Review'].apply(lambda x: ' '.join(self.tokenize_and_lemmatize(x)))
            df['review_tokens'] = df['Review'].apply(self.tokenize_and_lemmatize)
        if 'Title' in df.columns:
            print("Processing title text...")
            df['processed_title'] = df['Title'].apply(lambda x: ' '.join(self.tokenize_and_lemmatize(x)))
            df['title_tokens'] = df['Title'].apply(self.tokenize_and_lemmatize)
        print("\n=== Step 6: Creating Additional Features ===")
        if 'Review' in df.columns:
            df['review_length'] = df['Review'].str.len()
            df['review_word_count'] = df['Review'].str.split().str.len()
            df['processed_word_count'] = df['processed_review'].str.split().str.len()
        if 'Rating - 10' in df.columns:
            df['rating_numeric'] = pd.to_numeric(df['Rating - 10'], errors='coerce')
        print(f"\n=== Preprocessing Complete ===")
        print(f"Final dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    
    def save_processed_data(self, df, output_path='processed_airline_reviews.csv'):
        try:
            df.to_csv(output_path, index=False)
            print(f"Processed dataset saved to {output_path}")
        except Exception as e:
            print(f"Error saving processed dataset: {str(e)}")

def main():
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess_dataset('Indian_Domestic_Airline.csv')
    if processed_df is not None:
        preprocessor.save_processed_data(processed_df)
        print("\n=== Final Dataset Statistics ===")
        print(f"Total records: {len(processed_df)}")
        print(f"Average review length: {processed_df['review_length'].mean():.2f} characters")
        print(f"Average word count: {processed_df['review_word_count'].mean():.2f} words")
        print(f"Rating distribution:")
        print(processed_df['rating_numeric'].value_counts().sort_index())
        return processed_df
    else:
        print("Preprocessing failed!")
        return None

if __name__ == "__main__":
    processed_data = main()