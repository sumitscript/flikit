import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor
from text_summarization import TextSummarizer
from predictive_insights import RecurringIssuesIdentifier, CustomerSatisfactionPredictor, InsightsReportGenerator
from sentiment_integration import SentimentPredictor
from web_visualizations import WebVisualizer
from chatbot_interface import ChatbotUI

class WebAppController:
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.summarizer = TextSummarizer(approach='extractive')
        self.issues_identifier = RecurringIssuesIdentifier(n_clusters=6, max_features=50)
        self.satisfaction_predictor = CustomerSatisfactionPredictor()
        self.report_generator = InsightsReportGenerator()
        self.sentiment_predictor = SentimentPredictor()
        self.visualizer = WebVisualizer()
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
        if 'sentiment_model_loaded' not in st.session_state:
            st.session_state.sentiment_model_loaded = False
    
    def load_sentiment_model(self):
        try:
            with open('sentiment_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            st.error("Sentiment model not found. Please ensure sentiment_model.pkl exists.")
            return None
        except Exception as e:
            st.error(f"Error loading sentiment model: {str(e)}")
            return None
    
    def process_uploaded_file(self, uploaded_file):
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} records")
            
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            st.write("**Columns:**", list(df.columns))
            st.write("**Sample Data:**")
            st.dataframe(df.head())
            
            return df
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            return None 
   
    def preprocess_data(self, df):
        with st.spinner("Preprocessing data..."):
            try:
                temp_file = "temp_uploaded_data.csv"
                df.to_csv(temp_file, index=False)
                processed_df = self.preprocessor.preprocess_dataset(temp_file)
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                if processed_df is not None:
                    st.success("Data preprocessing completed successfully!")
                    return processed_df
                else:
                    st.error("Data preprocessing failed!")
                    return None
            except Exception as e:
                st.error(f"Error during preprocessing: {str(e)}")
                return None
    
    def analyze_sentiment(self, df):
        with st.spinner("Analyzing sentiment..."):
            try:
                if not st.session_state.sentiment_model_loaded:
                    model_loaded = self.sentiment_predictor.load_model()
                    st.session_state.sentiment_model_loaded = model_loaded
                    
                    if model_loaded:
                        st.success("Trained sentiment model loaded successfully!")
                    else:
                        st.warning("Using rule-based sentiment analysis (trained model not available)")
                
                if 'Review' in df.columns:
                    df_with_sentiment = self.sentiment_predictor.analyze_dataframe(df, 'Review')
                    st.success("Sentiment analysis completed!")
                    return df_with_sentiment
                else:
                    return self.simple_sentiment_analysis(df)
                
            except Exception as e:
                st.error(f"Error during sentiment analysis: {str(e)}")
                return self.simple_sentiment_analysis(df)
    
    def simple_sentiment_analysis(self, df):
        def classify_sentiment(rating):
            if rating <= 4:
                return "Negative"
            elif rating <= 6:
                return "Neutral"
            else:
                return "Positive"
        
        if 'rating_numeric' in df.columns:
            df['sentiment'] = df['rating_numeric'].apply(classify_sentiment)
        else:
            df['sentiment'] = "Neutral"
        
        return df
    
    def generate_summaries(self, df):
        with st.spinner("Generating summaries..."):
            try:
                if 'Review' in df.columns:
                    sample_size = min(100, len(df))
                    sample_df = df.head(sample_size).copy()
                    
                    sample_df['short_summary'] = sample_df['Review'].apply(
                        lambda x: self.summarizer.generate_short_summary(str(x))
                    )
                    sample_df['detailed_summary'] = sample_df['Review'].apply(
                        lambda x: self.summarizer.generate_detailed_summary(str(x))
                    )
                    
                    df = df.merge(sample_df[['short_summary', 'detailed_summary']], 
                                left_index=True, right_index=True, how='left')
                
                st.success("Summary generation completed!")
                return df
                
            except Exception as e:
                st.error(f"Error generating summaries: {str(e)}")
                return df   
 
    def run_complete_analysis(self, df):
        with st.spinner("Running complete analysis..."):
            try:
                temp_file = "temp_processed_data.csv"
                df.to_csv(temp_file, index=False)
                
                st.write("Identifying recurring issues...")
                issues_results = self.issues_identifier.analyze_recurring_issues(temp_file)
                
                st.write("Predicting customer satisfaction trends...")
                satisfaction_results = self.satisfaction_predictor.predict_customer_satisfaction(temp_file)
                
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                results = {
                    'issues_analysis': issues_results,
                    'satisfaction_prediction': satisfaction_results,
                    'processed_data': df
                }
                
                st.success("Complete analysis finished!")
                return results
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                return None

def create_sidebar():
    st.sidebar.title("Airline Feedback Analysis")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Upload Data", "Analysis Results", "Visualizations", "Reports", "AI Assistant"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application analyzes airline customer feedback using AI to identify "
        "recurring issues and predict satisfaction trends."
    )
    
    return page

def upload_data_page(controller):
    st.title("Upload Customer Feedback Data")
    st.markdown("Upload your airline customer feedback CSV file to begin analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing customer feedback data"
    )
    
    if uploaded_file is not None:
        df = controller.process_uploaded_file(uploaded_file)
        
        if df is not None:
            st.session_state.uploaded_data = df
            
            st.subheader("Preprocessing Options")
            
            col1, col2 = st.columns(2)
            with col1:
                preprocess_data = st.checkbox("Preprocess Data", value=True, 
                                            help="Clean and preprocess the text data")
            with col2:
                analyze_sentiment = st.checkbox("Analyze Sentiment", value=True,
                                              help="Perform sentiment analysis on reviews")
            
            generate_summaries = st.checkbox("Generate Summaries", value=False,
                                           help="Generate text summaries (may take longer)")
            
            if st.button("Start Analysis", type="primary"):
                processed_df = df.copy()
                
                if preprocess_data:
                    processed_df = controller.preprocess_data(processed_df)
                    if processed_df is None:
                        return
                
                if analyze_sentiment:
                    processed_df = controller.analyze_sentiment(processed_df)
                
                if generate_summaries:
                    processed_df = controller.generate_summaries(processed_df)
                
                results = controller.run_complete_analysis(processed_df)
                
                if results is not None:
                    st.session_state.analysis_results = results
                    st.success("Analysis completed! Navigate to 'Analysis Results' to view insights.")
    
    else:
        st.info("Please upload a CSV file to get started.")
        
        st.subheader("Expected Data Format")
        sample_data = pd.DataFrame({
            'Review': ['Great flight experience!', 'Delayed departure was frustrating'],
            'Rating - 10': [8, 3],
            'Date': ['1st January 2024', '2nd January 2024'],
            'Name': ['John Doe', 'Jane Smith'],
            'Recommond': ['yes', 'no']
        })
        st.dataframe(sample_data)

def analysis_results_page():
    st.title("Analysis Results")
    
    if st.session_state.analysis_results is None:
        st.warning("No analysis results available. Please upload and analyze data first.")
        return
    
    results = st.session_state.analysis_results
    
    st.subheader("Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_reviews = len(results['processed_data'])
        st.metric("Total Reviews", total_reviews)
    
    with col2:
        if 'sentiment' in results['processed_data'].columns:
            positive_reviews = len(results['processed_data'][results['processed_data']['sentiment'] == 'Positive'])
            st.metric("Positive Reviews", positive_reviews)
    
    with col3:
        if 'rating_numeric' in results['processed_data'].columns:
            avg_rating = results['processed_data']['rating_numeric'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}")
    
    with col4:
        if results['issues_analysis'] and 'total_negative_reviews' in results['issues_analysis']:
            issues_count = results['issues_analysis']['total_negative_reviews']
            st.metric("Issues Identified", issues_count)
    
    st.subheader("Recurring Issues Analysis")
    
    if results['issues_analysis'] and 'issue_summary' in results['issues_analysis']:
        issue_summary = results['issues_analysis']['issue_summary']
        
        issue_data = []
        for category, summary in issue_summary.items():
            issue_data.append({
                'Issue Category': category,
                'Total Complaints': summary['total_complaints'],
                'Percentage': f"{summary['percentage']:.1f}%",
                'Number of Clusters': summary['num_clusters']
            })
        
        issue_df = pd.DataFrame(issue_data)
        st.dataframe(issue_df, use_container_width=True)
    
    st.subheader("Satisfaction Prediction")
    
    if results['satisfaction_prediction'] and results['satisfaction_prediction']['trend_analysis']:
        trend = results['satisfaction_prediction']['trend_analysis']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Historical Average", f"{trend['historical_average']:.2f}")
        with col2:
            st.metric("Recent Average", f"{trend['recent_average']:.2f}")
        with col3:
            st.metric("Forecast Average", f"{trend['forecast_average']:.2f}")
        
        trend_direction = trend['trend_direction']
        if trend_direction == "Improving":
            st.success(f"Trend Direction: {trend_direction}")
        elif trend_direction == "Declining":
            st.error(f"Trend Direction: {trend_direction}")
        else:
            st.info(f"Trend Direction: {trend_direction}")
    
    if 'short_summary' in results['processed_data'].columns:
        st.subheader("Sample Reviews with Summaries")
        
        sample_reviews = results['processed_data'].dropna(subset=['short_summary']).head(5)
        
        for idx, row in sample_reviews.iterrows():
            with st.expander(f"Review {idx + 1} - Rating: {row.get('rating_numeric', 'N/A')}"):
                st.write("**Original Review:**")
                st.write(row['Review'][:300] + "..." if len(str(row['Review'])) > 300 else row['Review'])
                st.write("**Summary:**")
                st.write(row['short_summary'])
                if 'sentiment' in row:
                    sentiment_color = {"Positive": "green", "Negative": "red", "Neutral": "orange"}
                    st.markdown(f"**Sentiment:** :{sentiment_color.get(row['sentiment'], 'blue')}[{row['sentiment']}]")

def visualizations_page():
    st.title("Interactive Visualizations")
    
    if st.session_state.analysis_results is None:
        st.warning("No analysis results available. Please upload and analyze data first.")
        return
    
    results = st.session_state.analysis_results
    df = results['processed_data']
    
    st.subheader("Sentiment Distribution")
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts()
        
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Customer Sentiment Distribution",
            color_discrete_map={
                'Positive': '#2E8B57',
                'Negative': '#DC143C',
                'Neutral': '#FF8C00'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.subheader("Rating Distribution")
    if 'rating_numeric' in df.columns:
        fig_hist = px.histogram(
            df,
            x='rating_numeric',
            nbins=10,
            title="Distribution of Customer Ratings",
            labels={'rating_numeric': 'Rating', 'count': 'Number of Reviews'}
        )
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    if results['issues_analysis'] and 'issue_summary' in results['issues_analysis']:
        st.subheader("Issue Categories Analysis")
        
        issue_summary = results['issues_analysis']['issue_summary']
        categories = list(issue_summary.keys())
        complaints = [issue_summary[cat]['total_complaints'] for cat in categories]
        percentages = [issue_summary[cat]['percentage'] for cat in categories]
        
        fig_bar = px.bar(
            x=categories,
            y=complaints,
            title="Number of Complaints by Category",
            labels={'x': 'Issue Category', 'y': 'Number of Complaints'}
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

def reports_page():
    st.title("Reports & Downloads")
    
    if st.session_state.analysis_results is None:
        st.warning("No analysis results available. Please upload and analyze data first.")
        return
    
    results = st.session_state.analysis_results
    df = results['processed_data']
    total_reviews = len(df)
    
    report_content = f"""
# Customer Feedback Analysis Report

**Generated on:** {datetime.now().strftime('%B %d, %Y at %H:%M')}

## Executive Summary
- **Total Reviews Analyzed:** {total_reviews:,}
- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}

## Key Findings
"""
    
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts()
        report_content += f"""
### Sentiment Analysis
- **Positive Reviews:** {sentiment_counts.get('Positive', 0):,} ({sentiment_counts.get('Positive', 0)/total_reviews*100:.1f}%)
- **Negative Reviews:** {sentiment_counts.get('Negative', 0):,} ({sentiment_counts.get('Negative', 0)/total_reviews*100:.1f}%)
- **Neutral Reviews:** {sentiment_counts.get('Neutral', 0):,} ({sentiment_counts.get('Neutral', 0)/total_reviews*100:.1f}%)
"""
    
    if 'rating_numeric' in df.columns:
        avg_rating = df['rating_numeric'].mean()
        median_rating = df['rating_numeric'].median()
        report_content += f"""
### Rating Analysis
- **Average Rating:** {avg_rating:.2f}/10
- **Median Rating:** {median_rating:.1f}/10
"""
    
    if results['issues_analysis'] and 'issue_summary' in results['issues_analysis']:
        report_content += "\n### Top Issue Categories\n"
        issue_summary = results['issues_analysis']['issue_summary']
        
        for category, summary in sorted(issue_summary.items(), 
                                      key=lambda x: x[1]['percentage'], reverse=True):
            report_content += f"- **{category}:** {summary['total_complaints']} complaints ({summary['percentage']:.1f}%)\n"
    
    if results['satisfaction_prediction'] and results['satisfaction_prediction']['trend_analysis']:
        trend = results['satisfaction_prediction']['trend_analysis']
        report_content += f"""
### Satisfaction Prediction
- **Historical Average:** {trend['historical_average']:.2f}/10
- **Recent Average:** {trend['recent_average']:.2f}/10
- **Forecast Average:** {trend['forecast_average']:.2f}/10
- **Trend Direction:** {trend['trend_direction']}
"""
    
    st.markdown(report_content)
    
    st.subheader("Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download Processed Data (CSV)",
            data=csv_data,
            file_name=f"processed_feedback_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            label="Download Summary Report (TXT)",
            data=report_content,
            file_name=f"feedback_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

def chatbot_page():
    try:
        chatbot_ui = ChatbotUI()
        chatbot_ui.render_chatbot_interface(st.session_state.analysis_results)
    except Exception as e:
        st.error(f"Chatbot functionality not available: {str(e)}")
        st.info("The chatbot feature requires additional setup. Please check the documentation for instructions.")

def main():
    st.set_page_config(
        page_title="Airline Feedback Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    controller = WebAppController()
    current_page = create_sidebar()
    
    if current_page == "Upload Data":
        upload_data_page(controller)
    elif current_page == "Analysis Results":
        analysis_results_page()
    elif current_page == "Visualizations":
        visualizations_page()
    elif current_page == "Reports":
        reports_page()
    elif current_page == "AI Assistant":
        chatbot_page()

if __name__ == "__main__":
    main()