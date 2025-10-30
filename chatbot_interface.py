import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from chatbot_integration import ChatbotFramework, ConversationContextManager

class QueryResponseSystem:
    
    def __init__(self, chatbot_framework: ChatbotFramework):
        self.chatbot = chatbot_framework
        self.predefined_queries = self._initialize_predefined_queries()
        self.suggestion_templates = self._initialize_suggestion_templates()
    
    def _initialize_predefined_queries(self) -> Dict[str, str]:
        return {
            "summary": "Can you provide a summary of the customer feedback analysis?",
            "sentiment": "What is the overall sentiment distribution?",
            "issues": "What are the main issues customers are facing?",
            "trends": "What are the satisfaction trends?",
            "recommendations": "What recommendations do you have for improvement?",
            "ratings": "How are the ratings distributed?",
            "performance": "How is our overall performance?"
        }
    
    def _initialize_suggestion_templates(self) -> Dict[str, List[str]]:
        return {
            "improvement_suggestions": [
                "Focus on addressing the most frequent complaint categories",
                "Implement staff training programs for service quality issues",
                "Establish proactive communication for flight delays",
                "Enhance baggage handling procedures",
                "Improve food and amenity offerings based on feedback",
                "Streamline booking and check-in processes",
                "Consider value-for-money improvements"
            ],
            "operational_recommendations": [
                "Monitor satisfaction trends weekly",
                "Set up automated alerts for satisfaction drops",
                "Create customer feedback response protocols",
                "Implement regular staff performance reviews",
                "Establish customer service recovery procedures"
            ]
        }
    
    def process_query(self, query: str, analysis_context: Dict[str, Any]) -> str:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['summary', 'overview', 'general']):
            return self._generate_summary_response(analysis_context)
        elif any(word in query_lower for word in ['sentiment', 'feeling', 'emotion']):
            return self._generate_sentiment_response(analysis_context)
        elif any(word in query_lower for word in ['issue', 'problem', 'complaint']):
            return self._generate_issues_response(analysis_context)
        elif any(word in query_lower for word in ['trend', 'forecast', 'prediction']):
            return self._generate_trend_response(analysis_context)
        elif any(word in query_lower for word in ['recommend', 'suggest', 'improve']):
            return self._generate_recommendations_response(analysis_context)
        elif any(word in query_lower for word in ['rating', 'score', 'performance']):
            return self._generate_rating_response(analysis_context)
        else:
            return self.chatbot.process_query(query, analysis_context)
    
    def _generate_summary_response(self, analysis_context: Dict[str, Any]) -> str:
        summary_parts = ["**Analysis Summary**"]
        
        if 'processed_data' in analysis_context:
            df = analysis_context['processed_data']
            total_reviews = len(df)
            
            summary_parts.append(f"**Performance Summary** (Based on {total_reviews:,} reviews)")
            
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts()
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / total_reviews) * 100
                    summary_parts.append(f"{sentiment}: {count:,} ({percentage:.1f}%)")
            
            if 'rating_numeric' in df.columns:
                avg_rating = df['rating_numeric'].mean()
                summary_parts.append(f"Average Rating: {avg_rating:.2f}/10")
        
        return "\n".join(summary_parts)
    
    def _generate_trend_analysis(self, analysis_context: Dict[str, Any]) -> str:
        if 'satisfaction_prediction' in analysis_context:
            satisfaction_results = analysis_context['satisfaction_prediction']
            if satisfaction_results and 'trend_analysis' in satisfaction_results:
                trend = satisfaction_results['trend_analysis']
                
                trend_parts = [
                    "**Trend Analysis**",
                    f"Historical Average: {trend['historical_average']:.2f}/10",
                    f"Recent Average: {trend['recent_average']:.2f}/10",
                    f"Forecast Average: {trend['forecast_average']:.2f}/10",
                    f"Trend Direction: {trend['trend_direction']}"
                ]
                
                return "\n".join(trend_parts)
        
        return "Trend analysis data is not available."
    
    def _generate_issues_breakdown(self, analysis_context: Dict[str, Any]) -> str:
        if 'issues_analysis' in analysis_context:
            issues = analysis_context['issues_analysis']
            if 'issue_summary' in issues:
                breakdown_parts = ["**Issue Breakdown**"]
                
                for category, data in issues['issue_summary'].items():
                    breakdown_parts.append(f"- {category}: {data['total_complaints']} complaints ({data['percentage']:.1f}%)")
                
                return "\n".join(breakdown_parts)
        
        return "Issue analysis data is not available."
    
    def _generate_sentiment_response(self, analysis_context: Dict[str, Any]) -> str:
        return self._generate_summary_response(analysis_context)
    
    def _generate_issues_response(self, analysis_context: Dict[str, Any]) -> str:
        return self._generate_issues_breakdown(analysis_context)
    
    def _generate_trend_response(self, analysis_context: Dict[str, Any]) -> str:
        return self._generate_trend_analysis(analysis_context)
    
    def _generate_recommendations_response(self, analysis_context: Dict[str, Any]) -> str:
        recommendations = []
        recommendations.extend(self.suggestion_templates['improvement_suggestions'][:3])
        recommendations.extend(self.suggestion_templates['operational_recommendations'][:2])
        
        return "**Recommendations:**\n" + "\n".join([f"- {rec}" for rec in recommendations])
    
    def _generate_rating_response(self, analysis_context: Dict[str, Any]) -> str:
        if 'processed_data' in analysis_context:
            df = analysis_context['processed_data']
            if 'rating_numeric' in df.columns:
                avg_rating = df['rating_numeric'].mean()
                median_rating = df['rating_numeric'].median()
                rating_counts = df['rating_numeric'].value_counts().sort_index()
                
                response_parts = [
                    "**Rating Analysis**",
                    f"Average Rating: {avg_rating:.2f}/10",
                    f"Median Rating: {median_rating:.1f}/10",
                    "**Rating Distribution:**"
                ]
                
                for rating, count in rating_counts.items():
                    percentage = (count / len(df)) * 100
                    response_parts.append(f"- {rating}/10: {count} reviews ({percentage:.1f}%)")
                
                return "\n".join(response_parts)
        
        return "Rating data is not available."

class ChatbotUI:
    
    def __init__(self):
        self.context_manager = ConversationContextManager()
        self.chatbot_framework = ChatbotFramework()
        self.query_system = QueryResponseSystem(self.chatbot_framework)
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'context_set' not in st.session_state:
            st.session_state.context_set = False
    
    def render_chatbot_interface(self, analysis_results: Optional[Dict[str, Any]] = None):
        st.title("AI Customer Feedback Assistant")
        st.markdown("Ask questions about your customer feedback analysis or get improvement suggestions!")
        
        if analysis_results:
            self.context_manager.set_analysis_context(analysis_results)
            st.session_state.context_set = True
        
        current_context = self.context_manager.get_analysis_context()
        if not current_context:
            st.warning("No analysis data available. Please run the feedback analysis first to enable chatbot functionality.")
            return
        
        self._render_quick_insights(current_context)
        self._render_chat_interface(current_context)
        self._render_predefined_queries()
    
    def _render_quick_insights(self, analysis_context: Dict[str, Any]):
        st.subheader("Quick Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Get Summary"):
                summary = self.query_system._generate_summary_response(analysis_context)
                st.session_state.chat_history.append(("You", "Can you provide a summary?"))
                st.session_state.chat_history.append(("Assistant", summary))
        
        with col2:
            if st.button("Get Recommendations"):
                recommendations = self.query_system._generate_recommendations_response(analysis_context)
                st.session_state.chat_history.append(("You", "What are your recommendations?"))
                st.session_state.chat_history.append(("Assistant", recommendations))
        
        st.subheader("AI-Generated Insights")
        
        if st.button("Generate Comprehensive Insights"):
            with st.spinner("Analyzing data and generating insights..."):
                try:
                    comprehensive_response = self.chatbot_framework.generate_comprehensive_insights(analysis_context)
                    st.markdown("### Comprehensive Analysis")
                    st.markdown(comprehensive_response)
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
        
        if st.button("Get Improvement Suggestions"):
            with st.spinner("Generating recommendations..."):
                suggestions_response = self.chatbot.get_suggestions_for_improvement()
                st.markdown("### AI-Generated Recommendations")
                st.markdown(suggestions_response)
    
    def _render_chat_interface(self, analysis_context: Dict[str, Any]):
        st.subheader("Chat with AI Assistant")
        
        for speaker, message in st.session_state.chat_history:
            if speaker == "You":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**Assistant:** {message}")
        
        user_input = st.text_input("Ask a question about your feedback analysis:", key="chat_input")
        
        if st.button("Send") and user_input:
            st.session_state.chat_history.append(("You", user_input))
            
            with st.spinner("Processing your question..."):
                try:
                    response = self.query_system.process_query(user_input, analysis_context)
                    st.session_state.chat_history.append(("Assistant", response))
                except Exception as e:
                    error_response = f"I apologize, but I encountered an error processing your question: {str(e)}"
                    st.session_state.chat_history.append(("Assistant", error_response))
            
            st.rerun()
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    def _render_predefined_queries(self):
        st.subheader("Quick Questions")
        st.markdown("Click on any question below:")
        
        queries = self.query_system.predefined_queries
        
        col1, col2 = st.columns(2)
        
        query_items = list(queries.items())
        mid_point = len(query_items) // 2
        
        with col1:
            for key, query in query_items[:mid_point]:
                if st.button(query, key=f"predefined_{key}"):
                    st.session_state.chat_history.append(("You", query))
                    
                    current_context = self.context_manager.get_analysis_context()
                    if current_context:
                        response = self.query_system.process_query(query, current_context)
                        st.session_state.chat_history.append(("Assistant", response))
                        st.rerun()
        
        with col2:
            for key, query in query_items[mid_point:]:
                if st.button(query, key=f"predefined_{key}"):
                    st.session_state.chat_history.append(("You", query))
                    
                    current_context = self.context_manager.get_analysis_context()
                    if current_context:
                        response = self.query_system.process_query(query, current_context)
                        st.session_state.chat_history.append(("Assistant", response))
                        st.rerun()

def main():
    st.set_page_config(page_title="AI Feedback Assistant", layout="wide")
    
    chatbot_ui = ChatbotUI()
    chatbot_ui.render_chatbot_interface()

if __name__ == "__main__":
    main()