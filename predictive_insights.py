import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class RecurringIssuesIdentifier:
    
    def __init__(self, n_clusters=8, max_features=100):
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        
    def load_processed_data(self, file_path='processed_airline_reviews.csv'):
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} reviews for analysis")
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def prepare_clustering_data(self, df):
        negative_reviews = df[df['rating_numeric'] <= 5].copy()
        print(f"Analyzing {len(negative_reviews)} negative/neutral reviews for issues")
        
        review_texts = negative_reviews['processed_review'].fillna('').tolist()
        ratings = negative_reviews['rating_numeric'].tolist()
        
        tfidf_features = self.vectorizer.fit_transform(review_texts)
        rating_features = np.array(ratings).reshape(-1, 1)
        rating_features_scaled = self.scaler.fit_transform(rating_features)
        
        feature_matrix = np.hstack([
            tfidf_features.toarray(),
            rating_features_scaled
        ])
        
        return feature_matrix, review_texts, ratings, negative_reviews
    
    def identify_clusters(self, feature_matrix):
        cluster_labels = self.kmeans.fit_predict(feature_matrix)
        return cluster_labels
    
    def extract_cluster_themes(self, review_texts, cluster_labels):
        cluster_themes = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_reviews = [
                review_texts[i] for i in range(len(review_texts)) 
                if cluster_labels[i] == cluster_id
            ]
            
            if not cluster_reviews:
                continue
            
            cluster_vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            try:
                cluster_tfidf = cluster_vectorizer.fit_transform(cluster_reviews)
                feature_names = cluster_vectorizer.get_feature_names_out()
                mean_scores = np.mean(cluster_tfidf.toarray(), axis=0)
                top_indices = np.argsort(mean_scores)[::-1][:10]
                top_keywords = [feature_names[i] for i in top_indices]
                
                cluster_themes[cluster_id] = {
                    'size': len(cluster_reviews),
                    'keywords': top_keywords,
                    'sample_reviews': cluster_reviews[:3]
                }
                
            except Exception as e:
                print(f"Error processing cluster {cluster_id}: {str(e)}")
                cluster_themes[cluster_id] = {
                    'size': len(cluster_reviews),
                    'keywords': [],
                    'sample_reviews': cluster_reviews[:3]
                }
        
        return cluster_themes    

    def categorize_issues(self, cluster_themes):
        issue_categories = {
            'Service Quality': ['staff', 'crew', 'service', 'rude', 'unprofessional', 'customer'],
            'Flight Operations': ['delay', 'delayed', 'cancel', 'cancelled', 'time', 'departure'],
            'Comfort & Seating': ['seat', 'comfort', 'legroom', 'space', 'cramped', 'uncomfortable'],
            'Baggage Issues': ['luggage', 'baggage', 'bag', 'lost', 'damaged'],
            'Food & Amenities': ['food', 'meal', 'water', 'amenities', 'entertainment'],
            'Booking & Check-in': ['booking', 'check', 'reservation', 'website', 'app'],
            'Value for Money': ['expensive', 'price', 'money', 'value', 'cost', 'fare']
        }
        
        categorized_issues = {}
        
        for cluster_id, theme in cluster_themes.items():
            keywords = theme['keywords']
            
            best_category = 'Other Issues'
            max_matches = 0
            
            for category, category_keywords in issue_categories.items():
                matches = sum(1 for keyword in keywords if any(ck in keyword.lower() for ck in category_keywords))
                if matches > max_matches:
                    max_matches = matches
                    best_category = category
            
            if best_category not in categorized_issues:
                categorized_issues[best_category] = []
            
            categorized_issues[best_category].append({
                'cluster_id': cluster_id,
                'size': theme['size'],
                'keywords': keywords,
                'sample_reviews': theme['sample_reviews']
            })
        
        return categorized_issues
    
    def analyze_recurring_issues(self, file_path='processed_airline_reviews.csv'):
        print("=== Starting Recurring Issues Analysis ===")
        
        df = self.load_processed_data(file_path)
        if df is None:
            return None
        
        print("Preparing data for clustering...")
        feature_matrix, review_texts, ratings, negative_reviews = self.prepare_clustering_data(df)
        
        print("Identifying issue clusters...")
        cluster_labels = self.identify_clusters(feature_matrix)
        
        print("Extracting cluster themes...")
        cluster_themes = self.extract_cluster_themes(review_texts, cluster_labels)
        
        print("Categorizing issues...")
        categorized_issues = self.categorize_issues(cluster_themes)
        
        total_negative_reviews = len(negative_reviews)
        issue_summary = {}
        
        for category, issues in categorized_issues.items():
            total_size = sum(issue['size'] for issue in issues)
            issue_summary[category] = {
                'total_complaints': total_size,
                'percentage': (total_size / total_negative_reviews) * 100,
                'num_clusters': len(issues)
            }
        
        results = {
            'categorized_issues': categorized_issues,
            'issue_summary': issue_summary,
            'cluster_labels': cluster_labels,
            'total_negative_reviews': total_negative_reviews,
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print("=== Recurring Issues Analysis Complete ===")
        return results
    
    def visualize_issues(self, results):
        if not results:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        categories = list(results['issue_summary'].keys())
        percentages = [results['issue_summary'][cat]['percentage'] for cat in categories]
        
        ax1.pie(percentages, labels=categories, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribution of Issue Categories')
        
        complaints = [results['issue_summary'][cat]['total_complaints'] for cat in categories]
        
        bars = ax2.bar(range(len(categories)), complaints, color='skyblue')
        ax2.set_xlabel('Issue Categories')
        ax2.set_ylabel('Number of Complaints')
        ax2.set_title('Complaints Count by Category')
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(categories, rotation=45, ha='right')
        
        for bar, complaint in zip(bars, complaints):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{complaint}', ha='center', va='bottom')
        
        all_keywords = []
        for category, issues in results['categorized_issues'].items():
            for issue in issues:
                all_keywords.extend(issue['keywords'][:5])
        
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(10)
        
        if top_keywords:
            keywords, counts = zip(*top_keywords)
            ax3.barh(range(len(keywords)), counts, color='lightcoral')
            ax3.set_yticks(range(len(keywords)))
            ax3.set_yticklabels(keywords)
            ax3.set_xlabel('Frequency')
            ax3.set_title('Top Issue Keywords')
        
        cluster_sizes = []
        for category, issues in results['categorized_issues'].items():
            for issue in issues:
                cluster_sizes.append(issue['size'])
        
        ax4.scatter(range(len(cluster_sizes)), cluster_sizes, alpha=0.7, color='green')
        ax4.set_xlabel('Cluster Index')
        ax4.set_ylabel('Number of Reviews')
        ax4.set_title('Cluster Sizes Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('recurring_issues_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Recurring issues visualization saved as 'recurring_issues_analysis.png'")

class CustomerSatisfactionPredictor:
    
    def __init__(self):
        self.prophet_model = None
        self.regression_model = None
        self.scaler = StandardScaler()
        
    def prepare_time_series_data(self, df):
        df_ts = df.copy()
        
        try:
            df_ts['date_parsed'] = pd.to_datetime(df_ts['Date'], format='%dth %B %Y', errors='coerce')
        except:
            try:
                df_ts['date_parsed'] = pd.to_datetime(df_ts['Date'], format='%dst %B %Y', errors='coerce')
            except:
                try:
                    df_ts['date_parsed'] = pd.to_datetime(df_ts['Date'], format='%dnd %B %Y', errors='coerce')
                except:
                    try:
                        df_ts['date_parsed'] = pd.to_datetime(df_ts['Date'], format='%drd %B %Y', errors='coerce')
                    except:
                        df_ts['date_parsed'] = pd.to_datetime(df_ts['Date'], errors='coerce')
        
        df_ts = df_ts.dropna(subset=['date_parsed'])
        
        if len(df_ts) == 0:
            print("Warning: No valid dates found in the dataset")
            return None
        
        daily_satisfaction = df_ts.groupby('date_parsed').agg({
            'rating_numeric': ['mean', 'count']
        }).reset_index()
        
        daily_satisfaction.columns = ['ds', 'y', 'review_count']
        daily_satisfaction = daily_satisfaction[daily_satisfaction['review_count'] >= 3]
        
        print(f"Prepared time-series data with {len(daily_satisfaction)} data points")
        print(f"Date range: {daily_satisfaction['ds'].min()} to {daily_satisfaction['ds'].max()}")
        
        return daily_satisfaction
    
    def train_prophet_model(self, ts_data):
        if ts_data is None or len(ts_data) < 10:
            print("Insufficient data for time-series forecasting")
            return None
        
        try:
            self.prophet_model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            self.prophet_model.fit(ts_data)
            print("Prophet model trained successfully")
            return self.prophet_model
            
        except Exception as e:
            print(f"Error training Prophet model: {str(e)}")
            return None    

    def create_satisfaction_features(self, df):
        features = []
        
        df['review_length_norm'] = df['review_length'] / df['review_length'].max()
        df['word_count_norm'] = df['review_word_count'] / df['review_word_count'].max()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing']
        
        df['positive_word_count'] = df['processed_review'].apply(
            lambda x: sum(1 for word in positive_words if word in str(x).lower())
        )
        df['negative_word_count'] = df['processed_review'].apply(
            lambda x: sum(1 for word in negative_words if word in str(x).lower())
        )
        
        df['recommends'] = (df['Recommond'] == 'yes').astype(int)
        
        feature_columns = [
            'review_length_norm', 'word_count_norm', 
            'positive_word_count', 'negative_word_count', 'recommends'
        ]
        
        X = df[feature_columns].fillna(0)
        y = df['rating_numeric']
        
        return X, y
    
    def train_regression_model(self, df):
        X, y = self.create_satisfaction_features(df)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.regression_model = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10
        )
        self.regression_model.fit(X_train_scaled, y_train)
        
        y_pred = self.regression_model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'feature_importance': dict(zip(X.columns, self.regression_model.feature_importances_))
        }
        
        print(f"Regression model trained - R²: {r2:.3f}, RMSE: {np.sqrt(mse):.3f}")
        return self.regression_model, metrics
    
    def forecast_satisfaction(self, ts_data, periods=30):
        if self.prophet_model is None:
            print("Prophet model not trained")
            return None
        
        try:
            future = self.prophet_model.make_future_dataframe(periods=periods)
            forecast = self.prophet_model.predict(future)
            
            forecast_results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            forecast_results.columns = ['date', 'predicted_satisfaction', 'lower_bound', 'upper_bound']
            
            print(f"Generated {periods}-day satisfaction forecast")
            return forecast_results
            
        except Exception as e:
            print(f"Error generating forecast: {str(e)}")
            return None
    
    def analyze_trends(self, ts_data, forecast_results):
        if ts_data is None or forecast_results is None:
            return {}
        
        recent_avg = ts_data['y'].tail(7).mean()
        overall_avg = ts_data['y'].mean()
        
        forecast_avg = forecast_results['predicted_satisfaction'].mean()
        forecast_trend = forecast_results['predicted_satisfaction'].iloc[-1] - forecast_results['predicted_satisfaction'].iloc[0]
        
        if forecast_trend > 0.1:
            trend_direction = "Improving"
        elif forecast_trend < -0.1:
            trend_direction = "Declining"
        else:
            trend_direction = "Stable"
        
        trend_analysis = {
            'historical_average': overall_avg,
            'recent_average': recent_avg,
            'forecast_average': forecast_avg,
            'trend_direction': trend_direction,
            'trend_magnitude': abs(forecast_trend),
            'improvement_vs_recent': forecast_avg - recent_avg,
            'improvement_vs_historical': forecast_avg - overall_avg
        }
        
        return trend_analysis
    
    def predict_customer_satisfaction(self, file_path='processed_airline_reviews.csv'):
        print("=== Starting Customer Satisfaction Prediction ===")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} reviews for satisfaction prediction")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
        
        print("Preparing time-series data...")
        ts_data = self.prepare_time_series_data(df)
        
        print("Training time-series forecasting model...")
        prophet_model = self.train_prophet_model(ts_data)
        
        print("Training regression model...")
        regression_model, regression_metrics = self.train_regression_model(df)
        
        forecast_results = None
        trend_analysis = {}
        
        if prophet_model is not None:
            print("Generating satisfaction forecasts...")
            forecast_results = self.forecast_satisfaction(ts_data, periods=30)
            
            if forecast_results is not None:
                trend_analysis = self.analyze_trends(ts_data, forecast_results)
        
        results = {
            'time_series_data': ts_data,
            'forecast_results': forecast_results,
            'trend_analysis': trend_analysis,
            'regression_metrics': regression_metrics,
            'models': {
                'prophet': prophet_model,
                'regression': regression_model
            },
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print("=== Customer Satisfaction Prediction Complete ===")
        return results
    
    def visualize_predictions(self, results):
        if not results:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        if results['time_series_data'] is not None:
            ts_data = results['time_series_data']
            ax1.plot(ts_data['ds'], ts_data['y'], 'b-', alpha=0.7, label='Historical')
            ax1.set_title('Historical Customer Satisfaction Trend')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Average Rating')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        if results['forecast_results'] is not None:
            forecast = results['forecast_results']
            ax2.plot(forecast['date'], forecast['predicted_satisfaction'], 'r-', label='Forecast')
            ax2.fill_between(forecast['date'], 
                           forecast['lower_bound'], 
                           forecast['upper_bound'], 
                           alpha=0.3, color='red', label='Confidence Interval')
            ax2.set_title('30-Day Satisfaction Forecast')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Predicted Rating')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        if 'regression_metrics' in results and 'feature_importance' in results['regression_metrics']:
            importance = results['regression_metrics']['feature_importance']
            features = list(importance.keys())
            importances = list(importance.values())
            
            ax3.barh(features, importances, color='green', alpha=0.7)
            ax3.set_title('Feature Importance for Satisfaction Prediction')
            ax3.set_xlabel('Importance')
        
        if results['trend_analysis']:
            trend = results['trend_analysis']
            
            metrics = ['Historical Avg', 'Recent Avg', 'Forecast Avg']
            values = [trend['historical_average'], trend['recent_average'], trend['forecast_average']]
            
            bars = ax4.bar(metrics, values, color=['blue', 'orange', 'red'], alpha=0.7)
            ax4.set_title('Satisfaction Comparison')
            ax4.set_ylabel('Average Rating')
            ax4.set_ylim(0, 10)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('satisfaction_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Satisfaction prediction visualization saved as 'satisfaction_prediction.png'")

class InsightsReportGenerator:
    
    def __init__(self):
        self.report_data = {}
        
    def create_comprehensive_dashboard(self, issues_results, satisfaction_results):
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        if issues_results and 'issue_summary' in issues_results:
            categories = list(issues_results['issue_summary'].keys())
            percentages = [issues_results['issue_summary'][cat]['percentage'] for cat in categories]
            
            ax1.pie(percentages, labels=categories, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Issue Categories Distribution', fontsize=12, fontweight='bold')
        
        fig.suptitle('Intelligent Customer Feedback Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('comprehensive_insights_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Comprehensive dashboard saved as 'comprehensive_insights_dashboard.png'")
    
    def generate_complete_insights_report(self, issues_results, satisfaction_results):
        print("=== Generating Comprehensive Insights Report ===")
        self.create_comprehensive_dashboard(issues_results, satisfaction_results)
        print("=== Insights Report Generation Complete ===")

def main():
    identifier = RecurringIssuesIdentifier(n_clusters=8, max_features=100)
    results = identifier.analyze_recurring_issues()
    
    if results:
        print("\n=== RECURRING ISSUES SUMMARY ===")
        print(f"Total negative/neutral reviews analyzed: {results['total_negative_reviews']}")
        print("\nIssue Categories:")
        
        for category, summary in results['issue_summary'].items():
            print(f"\n{category}:")
            print(f"  - Total complaints: {summary['total_complaints']}")
            print(f"  - Percentage: {summary['percentage']:.1f}%")
            print(f"  - Number of clusters: {summary['num_clusters']}")
        
        identifier.visualize_issues(results)
        return results
    else:
        print("Failed to analyze recurring issues!")
        return None

if __name__ == "__main__":
    main()