import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
import streamlit as st

class ChatbotFramework:
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY', 'your-groq-api-key-here')
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.1-8b-instant"
        self.conversation_context = []
        self.analysis_context = {}
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def set_analysis_context(self, analysis_results: Dict[str, Any]):
        self.analysis_context = analysis_results
    
    def _create_context_prompt(self, analysis_context: Dict[str, Any]) -> str:
        context_parts = ["You are an AI assistant analyzing airline customer feedback data."]
        
        if 'processed_data' in analysis_context:
            df = analysis_context['processed_data']
            total_reviews = len(df)
            context_parts.append(f"Dataset contains {total_reviews} customer reviews.")
            
            if 'sentiment' in df.columns:
                sentiment_dist = df['sentiment'].value_counts()
                context_parts.append(f"Sentiment distribution: {dict(sentiment_dist)}")
            
            if 'rating_numeric' in df.columns:
                avg_rating = df['rating_numeric'].mean()
                context_parts.append(f"Average rating: {avg_rating:.2f}/10")
        
        if 'issues_analysis' in analysis_context and analysis_context['issues_analysis']:
            issues = analysis_context['issues_analysis']
            if 'issue_summary' in issues:
                context_parts.append("Main issue categories:")
                for category, data in issues['issue_summary'].items():
                    context_parts.append(f"- {category}: {data['total_complaints']} complaints")
        
        if 'satisfaction_prediction' in analysis_context and analysis_context['satisfaction_prediction']:
            satisfaction = analysis_context['satisfaction_prediction']
            if 'trend_analysis' in satisfaction and satisfaction['trend_analysis']:
                trend = satisfaction['trend_analysis']
                context_parts.append(f"Satisfaction trend: {trend['trend_direction']}")
                context_parts.append(f"Forecast average: {trend['forecast_average']:.2f}/10")
        
        return " ".join(context_parts)
    
    def process_query(self, user_query: str, analysis_context: Dict[str, Any] = None) -> str:
        try:
            context = analysis_context or self.analysis_context
            system_prompt = self._create_context_prompt(context)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Error: Unable to process query. Status code: {response.status_code}"
                
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def generate_comprehensive_insights(self, analysis_context: Dict[str, Any]) -> str:
        query = """
        Based on the customer feedback analysis data, provide a comprehensive summary including:
        1. Overall performance assessment
        2. Key strengths and weaknesses
        3. Most critical issues to address
        4. Satisfaction trends and predictions
        5. Specific actionable recommendations
        
        Please provide a detailed analysis in a structured format.
        """
        
        return self.process_query(query, analysis_context)
    
    def get_suggestions_for_improvement(self) -> str:
        query = """
        Based on the customer feedback analysis, provide specific, actionable recommendations for improving customer satisfaction. 
        Focus on practical steps that can be implemented to address the main issues identified in the data.
        """
        
        return self.process_query(query, self.analysis_context)
    
    def analyze_sentiment_trends(self, analysis_context: Dict[str, Any]) -> str:
        query = """
        Analyze the sentiment trends in the customer feedback data. 
        What patterns do you see? What might be causing positive or negative sentiment? 
        Provide insights into customer emotions and satisfaction drivers.
        """
        
        return self.process_query(query, analysis_context)
    
    def identify_priority_actions(self, analysis_context: Dict[str, Any]) -> str:
        query = """
        Based on the analysis results, identify the top 5 priority actions that should be taken immediately 
        to improve customer satisfaction. Rank them by impact and urgency.
        """
        
        return self.process_query(query, analysis_context)

class ConversationContextManager:
    
    def __init__(self):
        self.analysis_context = {}
        self.conversation_history = []
    
    def set_analysis_context(self, analysis_results: Dict[str, Any]):
        self.analysis_context = analysis_results
    
    def get_analysis_context(self) -> Dict[str, Any]:
        return self.analysis_context
    
    def add_conversation_turn(self, user_message: str, assistant_response: str):
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'assistant': assistant_response
        })
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history
    
    def clear_conversation_history(self):
        self.conversation_history = []
    
    def export_conversation(self) -> str:
        return json.dumps(self.conversation_history, indent=2)

def test_chatbot_integration():
    print("Testing Chatbot Integration...")
    
    chatbot = ChatbotFramework()
    
    sample_context = {
        'processed_data': pd.DataFrame({
            'sentiment': ['Positive', 'Negative', 'Neutral'] * 10,
            'rating_numeric': [8, 3, 5] * 10
        })
    }
    
    test_query = "What is the overall sentiment of the customer feedback?"
    response = chatbot.process_query(test_query, sample_context)
    
    print(f"Test Query: {test_query}")
    print(f"Response: {response}")
    print("Chatbot integration test completed!")

if __name__ == "__main__":
    test_chatbot_integration()