import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class WebVisualizer:
    
    def __init__(self):
        self.color_palette = {
            'Positive': '#2E8B57',
            'Negative': '#DC143C', 
            'Neutral': '#FF8C00',
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728'
        }
    
    def create_sentiment_dashboard(self, df):
        figures = {}
        
        if 'sentiment' not in df.columns:
            return figures
        
        sentiment_counts = df['sentiment'].value_counts()
        
        figures['sentiment_pie'] = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map=self.color_palette
        )
        
        if 'rating_numeric' in df.columns and 'sentiment' in df.columns:
            figures['sentiment_scatter'] = px.scatter(
                df,
                x='rating_numeric',
                y=df.index,
                color='sentiment',
                title="Sentiment vs Rating",
                color_discrete_map=self.color_palette,
                labels={'rating_numeric': 'Rating', 'y': 'Review Index'}
            )
        
        if 'sentiment_confidence' in df.columns:
            figures['confidence_hist'] = px.histogram(
                df,
                x='sentiment_confidence',
                color='sentiment',
                title="Sentiment Confidence Distribution",
                color_discrete_map=self.color_palette,
                labels={'sentiment_confidence': 'Confidence Score'}
            )
        
        return figures
    
    def create_rating_analysis_charts(self, df):
        figures = {}
        
        if 'rating_numeric' not in df.columns:
            return figures
        
        figures['rating_distribution'] = px.histogram(
            df,
            x='rating_numeric',
            nbins=10,
            title="Rating Distribution",
            labels={'rating_numeric': 'Rating', 'count': 'Number of Reviews'}
        )
        
        if 'review_length' in df.columns:
            figures['length_vs_rating'] = px.scatter(
                df,
                x='review_length',
                y='rating_numeric',
                title="Review Length vs Rating",
                labels={'review_length': 'Review Length (characters)', 'rating_numeric': 'Rating'}
            )
        
        if 'sentiment' in df.columns:
            avg_rating_by_sentiment = df.groupby('sentiment')['rating_numeric'].mean().reset_index()
            figures['avg_rating_by_sentiment'] = px.bar(
                avg_rating_by_sentiment,
                x='sentiment',
                y='rating_numeric',
                title="Average Rating by Sentiment",
                color='sentiment',
                color_discrete_map=self.color_palette
            )
        
        return figures
    
    def create_issues_analysis_charts(self, issues_results):
        figures = {}
        
        if not issues_results or 'issue_summary' not in issues_results:
            return figures
        
        issue_summary = issues_results['issue_summary']
        categories = list(issue_summary.keys())
        complaints = [issue_summary[cat]['total_complaints'] for cat in categories]
        percentages = [issue_summary[cat]['percentage'] for cat in categories]
        
        figures['issues_bar'] = px.bar(
            x=categories,
            y=complaints,
            title="Complaints by Issue Category",
            labels={'x': 'Issue Category', 'y': 'Number of Complaints'}
        )
        figures['issues_bar'].update_layout(xaxis_tickangle=-45)
        
        figures['issues_pie'] = px.pie(
            values=percentages,
            names=categories,
            title="Issue Categories Distribution (%)"
        )
        
        if 'categorized_issues' in issues_results:
            all_keywords = []
            for category, issues in issues_results['categorized_issues'].items():
                for issue in issues:
                    all_keywords.extend(issue['keywords'][:3])
            
            keyword_counts = Counter(all_keywords)
            top_keywords = keyword_counts.most_common(15)
            
            if top_keywords:
                keywords, counts = zip(*top_keywords)
                figures['keywords_bar'] = px.bar(
                    x=list(counts),
                    y=list(keywords),
                    orientation='h',
                    title="Top Issue Keywords",
                    labels={'x': 'Frequency', 'y': 'Keywords'}
                )
        
        return figures
    
    def create_satisfaction_trend_charts(self, satisfaction_results):
        figures = {}
        
        if not satisfaction_results:
            return figures
        
        if satisfaction_results['time_series_data'] is not None:
            ts_data = satisfaction_results['time_series_data']
            
            figures['historical_trend'] = px.line(
                ts_data,
                x='ds',
                y='y',
                title="Historical Satisfaction Trend",
                labels={'ds': 'Date', 'y': 'Average Rating'}
            )
            figures['historical_trend'].update_traces(line_color='#1f77b4', line_width=2)
        
        if satisfaction_results['forecast_results'] is not None:
            forecast = satisfaction_results['forecast_results']
            
            fig_forecast = go.Figure()
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast['date'],
                y=forecast['predicted_satisfaction'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2)
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast['date'],
                y=forecast['upper_bound'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast['date'],
                y=forecast['lower_bound'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Interval',
                fillcolor='rgba(255,0,0,0.2)'
            ))
            
            fig_forecast.update_layout(
                title="30-Day Satisfaction Forecast",
                xaxis_title="Date",
                yaxis_title="Predicted Rating"
            )
            
            figures['forecast_chart'] = fig_forecast
        
        if satisfaction_results['trend_analysis']:
            trend = satisfaction_results['trend_analysis']
            
            comparison_data = {
                'Period': ['Historical', 'Recent', 'Forecast'],
                'Average_Rating': [
                    trend['historical_average'],
                    trend['recent_average'],
                    trend['forecast_average']
                ]
            }
            
            figures['comparison_chart'] = px.bar(
                comparison_data,
                x='Period',
                y='Average_Rating',
                title="Satisfaction Comparison",
                color='Period'
            )
        
        return figures
    
    def display_metrics_cards(self, df, issues_results=None, satisfaction_results=None):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_reviews = len(df)
            st.metric("Total Reviews", f"{total_reviews:,}")
        
        with col2:
            if 'rating_numeric' in df.columns:
                avg_rating = df['rating_numeric'].mean()
                st.metric("Average Rating", f"{avg_rating:.2f}/10")
            else:
                st.metric("Average Rating", "N/A")
        
        with col3:
            if 'sentiment' in df.columns:
                positive_pct = (df['sentiment'] == 'Positive').mean() * 100
                st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            else:
                st.metric("Positive Sentiment", "N/A")
        
        with col4:
            if issues_results and 'total_negative_reviews' in issues_results:
                issues_count = issues_results['total_negative_reviews']
                st.metric("Issues Identified", f"{issues_count:,}")
            else:
                st.metric("Issues Identified", "N/A")
        
        if satisfaction_results and satisfaction_results['trend_analysis']:
            trend = satisfaction_results['trend_analysis']
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric("Historical Avg", f"{trend['historical_average']:.2f}")
            
            with col6:
                st.metric("Recent Avg", f"{trend['recent_average']:.2f}")
            
            with col7:
                forecast_avg = trend['forecast_average']
                recent_avg = trend['recent_average']
                delta = forecast_avg - recent_avg
                st.metric("Forecast Avg", f"{forecast_avg:.2f}", f"{delta:+.2f}")
            
            with col8:
                trend_direction = trend['trend_direction']
                st.metric("Trend", f"{trend_direction}")

def test_visualizations():
    print("Testing Web Visualizations...")
    
    sample_data = pd.DataFrame({
        'sentiment': ['Positive', 'Negative', 'Neutral'] * 10,
        'rating_numeric': [8, 3, 5] * 10,
        'review_length': [100, 200, 150] * 10,
        'sentiment_confidence': [0.9, 0.8, 0.6] * 10
    })
    
    visualizer = WebVisualizer()
    
    sentiment_figs = visualizer.create_sentiment_dashboard(sample_data)
    print(f"Created {len(sentiment_figs)} sentiment visualizations")
    
    rating_figs = visualizer.create_rating_analysis_charts(sample_data)
    print(f"Created {len(rating_figs)} rating visualizations")
    
    print("Web visualizations test completed successfully!")

if __name__ == "__main__":
    test_visualizations()