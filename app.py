import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="AI-Powered Analytics Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .ai-response {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AIAnalytics:
    """AI-powered analytics functionality"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
    
    def generate_insights(self, data: pd.DataFrame) -> str:
        """Generate AI insights from data"""
        if data.empty:
            return "No data available for analysis."
        
        # Basic statistical analysis
        insights = []
        insights.append(f"Dataset contains {len(data)} records with {len(data.columns)} features.")
        
        # Analyze numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"Numeric features: {', '.join(numeric_cols)}")
            for col in numeric_cols:
                mean_val = data[col].mean()
                std_val = data[col].std()
                insights.append(f"• {col}: Mean = {mean_val:.2f}, Std = {std_val:.2f}")
        
        # Analyze categorical columns
        cat_cols = data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            insights.append(f"Categorical features: {', '.join(cat_cols)}")
            for col in cat_cols:
                unique_count = data[col].nunique()
                insights.append(f"• {col}: {unique_count} unique values")
        
        return "\n".join(insights)
    
    def predict_trends(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Simple trend prediction using linear regression"""
        if target_column not in data.columns:
            return {"error": f"Column '{target_column}' not found in data"}
        
        try:
            # Simple trend analysis
            values = data[target_column].dropna()
            if len(values) < 2:
                return {"error": "Insufficient data for trend analysis"}
            
            # Calculate trend
            x = np.arange(len(values))
            coefficients = np.polyfit(x, values, 1)
            trend_direction = "increasing" if coefficients[0] > 0 else "decreasing"
            trend_strength = abs(coefficients[0])
            
            # Generate future predictions (simple linear extrapolation)
            future_x = np.arange(len(values), len(values) + 5)
            future_predictions = np.polyval(coefficients, future_x)
            
            return {
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "current_value": values.iloc[-1],
                "predicted_values": future_predictions.tolist(),
                "confidence": "Medium (Linear Extrapolation)"
            }
        except Exception as e:
            return {"error": f"Error in trend analysis: {str(e)}"}

def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate sample business data
    data = {
        'Date': dates,
        'Sales': np.random.normal(1000, 200, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 100,
        'Revenue': np.random.normal(5000, 800, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 500,
        'Customers': np.random.poisson(50, len(dates)),
        'Category': np.random.choice(['A', 'B', 'C'], len(dates)),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], len(dates))
    }
    
    df = pd.DataFrame(data)
    df['Sales'] = np.maximum(df['Sales'], 0)  # Ensure positive values
    df['Revenue'] = np.maximum(df['Revenue'], 0)
    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI-Powered Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize AI Analytics
    ai_analytics = AIAnalytics()
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # Data source selection
        data_source = st.selectbox(
            "Select Data Source",
            ["Sample Data", "Upload CSV", "Connect to API"]
        )
        
        # AI Features toggle
        st.header("🧠 AI Features")
        enable_insights = st.checkbox("Enable AI Insights", value=True)
        enable_predictions = st.checkbox("Enable Trend Predictions", value=True)
        
        # Visualization options
        st.header("📊 Visualization")
        chart_type = st.selectbox(
            "Chart Type",
            ["Line Chart", "Bar Chart", "Scatter Plot", "Heatmap"]
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Data Analysis")
        
        # Load data based on selection
        if data_source == "Sample Data":
            df = generate_sample_data()
            st.success("Sample data loaded successfully!")
        elif data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            else:
                df = generate_sample_data()
                st.info("Using sample data. Please upload a CSV file.")
        else:
            df = generate_sample_data()
            st.info("API connection not implemented. Using sample data.")
        
        # Display data overview
        st.subheader("Data Overview")
        st.dataframe(df.head(10))
        
        # Data metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total Records", len(df))
        with col_b:
            st.metric("Features", len(df.columns))
        with col_c:
            if 'Sales' in df.columns:
                st.metric("Avg Sales", f"${df['Sales'].mean():.2f}")
        with col_d:
            if 'Revenue' in df.columns:
                st.metric("Total Revenue", f"${df['Revenue'].sum():.2f}")
    
    with col2:
        st.header("🤖 AI Insights")
        
        if enable_insights:
            with st.spinner("Generating AI insights..."):
                insights = ai_analytics.generate_insights(df)
                st.markdown(f'<div class="ai-response"><strong>🧠 AI Analysis:</strong><br>{insights}</div>', unsafe_allow_html=True)
        
        if enable_predictions and len(df) > 0:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                target_col = st.selectbox("Select column for prediction", numeric_columns)
                
                if st.button("🔮 Generate Prediction"):
                    with st.spinner("Analyzing trends..."):
                        prediction = ai_analytics.predict_trends(df, target_col)
                        
                        if "error" not in prediction:
                            st.success(f"Trend: {prediction['trend_direction']}")
                            st.info(f"Current Value: {prediction['current_value']:.2f}")
                            
                            # Display predictions
                            st.write("**Next 5 Predictions:**")
                            for i, pred in enumerate(prediction['predicted_values'], 1):
                                st.write(f"Period {i}: {pred:.2f}")
    
    # Visualization section
    st.header("📊 Interactive Visualizations")
    
    if len(df) > 0:
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Time series visualization
            if 'Date' in df.columns and 'Sales' in df.columns:
                fig_line = px.line(df, x='Date', y='Sales', title='Sales Over Time')
                fig_line.update_layout(height=400)
                st.plotly_chart(fig_line, use_container_width=True)
        
        with viz_col2:
            # Category analysis
            if 'Category' in df.columns and 'Revenue' in df.columns:
                category_data = df.groupby('Category')['Revenue'].sum().reset_index()
                fig_bar = px.bar(category_data, x='Category', y='Revenue', title='Revenue by Category')
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Additional charts based on selection
        if chart_type == "Scatter Plot" and len(df.select_dtypes(include=[np.number]).columns) >= 2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col], key="scatter_y")
            
            fig_scatter = px.scatter(df, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        elif chart_type == "Heatmap":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                correlation_matrix = numeric_df.corr()
                fig_heatmap = px.imshow(correlation_matrix, title='Correlation Heatmap')
                st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**🚀 AI-Powered Analytics Dashboard** | Built with Streamlit and Python")

if __name__ == "__main__":
    main()