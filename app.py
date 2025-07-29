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
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

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

class FinancialAIAgent:
    """Advanced AI agent specifically designed for financial accounting analysis"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.financial_ratios = {}
        self.anomalies = []
        
    def analyze_financial_health(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive financial health analysis"""
        analysis = {
            "overall_health": "Unknown",
            "key_metrics": {},
            "recommendations": [],
            "risk_indicators": [],
            "strengths": []
        }
        
        try:
            # Calculate key financial ratios if we have the right columns
            if all(col in data.columns for col in ['Revenue', 'Expenses']):
                data['Net_Income'] = data['Revenue'] - data['Expenses']
                profit_margin = (data['Net_Income'].mean() / data['Revenue'].mean()) * 100
                analysis["key_metrics"]["Profit Margin %"] = round(profit_margin, 2)
                
                if profit_margin > 20:
                    analysis["strengths"].append("Strong profit margins")
                elif profit_margin < 5:
                    analysis["risk_indicators"].append("Low profit margins - review cost structure")
            
            # Revenue growth analysis
            if 'Revenue' in data.columns and len(data) > 1:
                revenue_growth = ((data['Revenue'].iloc[-1] - data['Revenue'].iloc[0]) / data['Revenue'].iloc[0]) * 100
                analysis["key_metrics"]["Revenue Growth %"] = round(revenue_growth, 2)
                
                if revenue_growth > 10:
                    analysis["strengths"].append("Strong revenue growth")
                elif revenue_growth < 0:
                    analysis["risk_indicators"].append("Declining revenue trend")
            
            # Cash flow analysis
            if 'Cash_Flow' in data.columns:
                avg_cash_flow = data['Cash_Flow'].mean()
                analysis["key_metrics"]["Average Cash Flow"] = round(avg_cash_flow, 2)
                
                if avg_cash_flow > 0:
                    analysis["strengths"].append("Positive cash flow")
                else:
                    analysis["risk_indicators"].append("Negative cash flow - monitor liquidity")
            
            # Determine overall health
            risk_count = len(analysis["risk_indicators"])
            strength_count = len(analysis["strengths"])
            
            if strength_count > risk_count:
                analysis["overall_health"] = "Good"
            elif risk_count > strength_count:
                analysis["overall_health"] = "Needs Attention"
            else:
                analysis["overall_health"] = "Moderate"
                
            # Generate recommendations
            if analysis["risk_indicators"]:
                analysis["recommendations"].append("Focus on addressing identified risk factors")
            if "Strong revenue growth" in analysis["strengths"]:
                analysis["recommendations"].append("Maintain current growth strategies")
            if risk_count == 0:
                analysis["recommendations"].append("Consider expansion opportunities")
                
        except Exception as e:
            analysis["error"] = f"Analysis error: {str(e)}"
            
        return analysis
    
    def detect_anomalies(self, data: pd.DataFrame, column: str) -> Dict[str, Any]:
        """AI-powered anomaly detection for financial data"""
        if column not in data.columns:
            return {"error": f"Column '{column}' not found"}
        
        try:
            values = data[column].dropna()
            if len(values) < 3:
                return {"error": "Insufficient data for anomaly detection"}
            
            # Use statistical methods for anomaly detection
            z_scores = np.abs(stats.zscore(values))
            threshold = 2.5
            anomaly_indices = np.where(z_scores > threshold)[0]
            
            anomalies = []
            for idx in anomaly_indices:
                anomalies.append({
                    "index": int(idx),
                    "value": float(values.iloc[idx]),
                    "z_score": float(z_scores[idx]),
                    "severity": "High" if z_scores[idx] > 3 else "Medium"
                })
            
            return {
                "total_anomalies": len(anomalies),
                "anomalies": anomalies,
                "percentage": round((len(anomalies) / len(values)) * 100, 2),
                "threshold_used": threshold
            }
            
        except Exception as e:
            return {"error": f"Anomaly detection failed: {str(e)}"}
    
    def forecast_financial_metrics(self, data: pd.DataFrame, target_column: str, periods: int = 12) -> Dict[str, Any]:
        """Advanced financial forecasting using multiple methods"""
        if target_column not in data.columns:
            return {"error": f"Column '{target_column}' not found"}
        
        try:
            values = data[target_column].dropna()
            if len(values) < 5:
                return {"error": "Insufficient data for forecasting"}
            
            # Prepare data for forecasting
            X = np.arange(len(values)).reshape(-1, 1)
            y = values.values
            
            # Linear regression forecast
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate future predictions
            future_X = np.arange(len(values), len(values) + periods).reshape(-1, 1)
            predictions = model.predict(future_X)
            
            # Calculate confidence intervals (simplified)
            residuals = y - model.predict(X)
            std_error = np.std(residuals)
            confidence_interval = 1.96 * std_error  # 95% confidence
            
            # Seasonal adjustment (simple)
            if len(values) >= 12:
                seasonal_component = np.mean([values[i::12] for i in range(min(12, len(values)))], axis=1)
                seasonal_factor = seasonal_component / np.mean(seasonal_component)
                # Apply seasonal adjustment to predictions
                for i, pred in enumerate(predictions):
                    season_idx = i % len(seasonal_factor)
                    predictions[i] = pred * seasonal_factor[season_idx]
            
            return {
                "predictions": predictions.tolist(),
                "confidence_interval": confidence_interval,
                "model_score": model.score(X, y),
                "trend": "increasing" if model.coef_[0] > 0 else "decreasing",
                "seasonal_adjusted": len(values) >= 12,
                "forecast_periods": periods
            }
            
        except Exception as e:
            return {"error": f"Forecasting failed: {str(e)}"}
    
    def generate_financial_insights(self, data: pd.DataFrame) -> str:
        """Generate comprehensive financial insights using AI analysis"""
        if data.empty:
            return "No financial data available for analysis."
        
        insights = []
        insights.append("🏦 FINANCIAL AI ANALYSIS REPORT")
        insights.append("=" * 40)
        
        # Basic financial metrics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        financial_keywords = ['revenue', 'sales', 'income', 'profit', 'expense', 'cost', 'cash', 'flow']
        financial_cols = [col for col in numeric_cols if any(keyword in col.lower() for keyword in financial_keywords)]
        
        if financial_cols:
            insights.append(f"📊 Key Financial Metrics Identified: {', '.join(financial_cols)}")
            
            for col in financial_cols:
                if data[col].notna().sum() > 0:
                    total = data[col].sum()
                    mean_val = data[col].mean()
                    trend = "↗️" if data[col].iloc[-1] > data[col].iloc[0] else "↘️"
                    insights.append(f"• {col}: Total=${total:,.2f}, Avg=${mean_val:,.2f} {trend}")
        
        # Performance analysis
        if 'Revenue' in data.columns and 'Expenses' in data.columns:
            net_income = data['Revenue'] - data['Expenses']
            profitability = (net_income.mean() / data['Revenue'].mean()) * 100
            insights.append(f"💰 Average Profitability: {profitability:.1f}%")
            
            if profitability > 15:
                insights.append("✅ Strong profitability - excellent financial health")
            elif profitability > 5:
                insights.append("⚠️ Moderate profitability - room for improvement")
            else:
                insights.append("🚨 Low profitability - requires immediate attention")
        
        # Trend analysis
        if len(data) > 1:
            insights.append("\n📈 TREND ANALYSIS:")
            for col in financial_cols[:3]:  # Analyze top 3 financial metrics
                if len(data[col].dropna()) > 1:
                    recent_avg = data[col].tail(len(data)//4).mean()
                    early_avg = data[col].head(len(data)//4).mean()
                    change = ((recent_avg - early_avg) / early_avg) * 100 if early_avg != 0 else 0
                    
                    if abs(change) > 10:
                        direction = "increased" if change > 0 else "decreased"
                        insights.append(f"• {col} has {direction} by {abs(change):.1f}% over the period")
        
        # Risk assessment
        insights.append("\n⚠️ RISK ASSESSMENT:")
        risk_factors = []
        
        for col in financial_cols:
            if data[col].notna().sum() > 2:
                volatility = data[col].std() / data[col].mean() if data[col].mean() != 0 else 0
                if volatility > 0.3:
                    risk_factors.append(f"High volatility in {col} ({volatility:.1%})")
        
        if risk_factors:
            for risk in risk_factors:
                insights.append(f"• {risk}")
        else:
            insights.append("• Low volatility detected - stable financial performance")
        
        return "\n".join(insights)
    
    def calculate_financial_ratios(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate standard financial ratios"""
        ratios = {}
        
        try:
            # Profitability ratios
            if 'Revenue' in data.columns and 'Expenses' in data.columns:
                net_income = data['Revenue'] - data['Expenses']
                ratios['Profit_Margin'] = (net_income.mean() / data['Revenue'].mean()) * 100
                
            if 'Gross_Profit' in data.columns and 'Revenue' in data.columns:
                ratios['Gross_Margin'] = (data['Gross_Profit'].mean() / data['Revenue'].mean()) * 100
            
            # Liquidity ratios
            if 'Current_Assets' in data.columns and 'Current_Liabilities' in data.columns:
                ratios['Current_Ratio'] = data['Current_Assets'].mean() / data['Current_Liabilities'].mean()
            
            # Efficiency ratios
            if 'Revenue' in data.columns and 'Total_Assets' in data.columns:
                ratios['Asset_Turnover'] = data['Revenue'].sum() / data['Total_Assets'].mean()
            
            # Growth ratios
            for col in ['Revenue', 'Net_Income', 'Total_Assets']:
                if col in data.columns and len(data) > 1:
                    growth = ((data[col].iloc[-1] - data[col].iloc[0]) / data[col].iloc[0]) * 100
                    ratios[f'{col}_Growth'] = growth
                    
        except Exception as e:
            ratios['calculation_error'] = str(e)
        
        return ratios

def generate_financial_sample_data():
    """Generate comprehensive financial sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate realistic financial data
    base_revenue = 50000
    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    growth_trend = 1 + 0.15 * np.arange(len(dates)) / len(dates)
    
    data = {
        'Date': dates,
        'Revenue': np.random.normal(base_revenue, 8000, len(dates)) * seasonal_factor * growth_trend,
        'Expenses': np.random.normal(base_revenue * 0.7, 5000, len(dates)) * seasonal_factor,
        'Gross_Profit': 0,  # Will be calculated
        'Net_Income': 0,    # Will be calculated
        'Cash_Flow': np.random.normal(10000, 3000, len(dates)),
        'Accounts_Receivable': np.random.normal(15000, 2000, len(dates)),
        'Accounts_Payable': np.random.normal(8000, 1500, len(dates)),
        'Inventory': np.random.normal(12000, 2500, len(dates)),
        'Current_Assets': np.random.normal(45000, 5000, len(dates)),
        'Current_Liabilities': np.random.normal(25000, 3000, len(dates)),
        'Total_Assets': np.random.normal(150000, 15000, len(dates)),
        'Department': np.random.choice(['Sales', 'Marketing', 'Operations', 'Admin'], len(dates)),
        'Transaction_Type': np.random.choice(['Income', 'Expense', 'Transfer'], len(dates)),
        'Customer_Segment': np.random.choice(['Enterprise', 'SMB', 'Individual'], len(dates))
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived financial metrics
    df['Revenue'] = np.maximum(df['Revenue'], 1000)  # Ensure positive values
    df['Expenses'] = np.maximum(df['Expenses'], 500)
    df['Gross_Profit'] = df['Revenue'] - df['Expenses'] * 0.6  # Assuming COGS is 60% of expenses
    df['Net_Income'] = df['Revenue'] - df['Expenses']
    df['Profit_Margin'] = (df['Net_Income'] / df['Revenue']) * 100
    df['Current_Ratio'] = df['Current_Assets'] / df['Current_Liabilities']
    
    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 AI-Powered Financial Accounting Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize Financial AI Agent
    financial_ai = FinancialAIAgent()
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # Data source selection
        data_source = st.selectbox(
            "Select Data Source",
            ["Financial Sample Data", "Upload Financial CSV", "Connect to Accounting API"]
        )
        
        # AI Features toggle
        st.header("� Financial AI Features")
        enable_insights = st.checkbox("Enable Financial AI Insights", value=True)
        enable_health_analysis = st.checkbox("Enable Financial Health Analysis", value=True)
        enable_anomaly_detection = st.checkbox("Enable Anomaly Detection", value=True)
        enable_forecasting = st.checkbox("Enable Financial Forecasting", value=True)
        
        # Analysis options
        st.header("📊 Analysis Options")
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Overview", "Profitability", "Cash Flow", "Balance Sheet", "Ratios"]
        )
        
        # Visualization options
        st.header("� Visualization")
        chart_type = st.selectbox(
            "Chart Type",
            ["Financial Dashboard", "Trend Analysis", "Ratio Analysis", "Anomaly Detection"]
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("� Financial Data Analysis")
        
        # Load data based on selection
        if data_source == "Financial Sample Data":
            df = generate_financial_sample_data()
            st.success("✅ Financial sample data loaded successfully!")
            st.info("📝 This sample includes: Revenue, Expenses, Cash Flow, Assets, Liabilities, and Financial Ratios")
        elif data_source == "Upload Financial CSV":
            uploaded_file = st.file_uploader("Choose a Financial CSV file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success("✅ Financial file uploaded successfully!")
            else:
                df = generate_financial_sample_data()
                st.info("Using financial sample data. Please upload a CSV file with columns like: Revenue, Expenses, Cash_Flow, Assets, Liabilities")
        else:
            df = generate_financial_sample_data()
            st.info("🔌 Accounting API connection not implemented. Using sample data.")
        
        # Display data overview
        st.subheader("📋 Financial Data Overview")
        
        # Show different views based on analysis type
        if analysis_type == "Overview":
            st.dataframe(df.head(10))
        elif analysis_type == "Profitability":
            profit_cols = [col for col in df.columns if any(term in col.lower() for term in ['revenue', 'profit', 'income', 'expense'])]
            if profit_cols:
                st.dataframe(df[profit_cols].head(10))
        elif analysis_type == "Cash Flow":
            cash_cols = [col for col in df.columns if any(term in col.lower() for term in ['cash', 'flow', 'receivable', 'payable'])]
            if cash_cols:
                st.dataframe(df[cash_cols].head(10))
        elif analysis_type == "Balance Sheet":
            balance_cols = [col for col in df.columns if any(term in col.lower() for term in ['asset', 'liability', 'inventory'])]
            if balance_cols:
                st.dataframe(df[balance_cols].head(10))
        
        # Financial metrics dashboard
        st.subheader("💰 Key Financial Metrics")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            if 'Revenue' in df.columns:
                total_revenue = df['Revenue'].sum()
                st.metric("Total Revenue", f"${total_revenue:,.0f}")
            else:
                st.metric("Total Records", len(df))
        
        with col_b:
            if 'Net_Income' in df.columns:
                total_profit = df['Net_Income'].sum()
                st.metric("Total Profit", f"${total_profit:,.0f}")
            elif 'Revenue' in df.columns and 'Expenses' in df.columns:
                total_profit = (df['Revenue'] - df['Expenses']).sum()
                st.metric("Total Profit", f"${total_profit:,.0f}")
        
        with col_c:
            if 'Profit_Margin' in df.columns:
                avg_margin = df['Profit_Margin'].mean()
                st.metric("Avg Profit Margin", f"{avg_margin:.1f}%")
            elif 'Current_Ratio' in df.columns:
                avg_ratio = df['Current_Ratio'].mean()
                st.metric("Avg Current Ratio", f"{avg_ratio:.2f}")
        
        with col_d:
            if 'Cash_Flow' in df.columns:
                total_cf = df['Cash_Flow'].sum()
                st.metric("Total Cash Flow", f"${total_cf:,.0f}")
    
    with col2:
        st.header("🤖 AI Financial Analysis")
        
        if enable_insights:
            with st.spinner("🧠 Generating financial insights..."):
                insights = financial_ai.generate_financial_insights(df)
                st.markdown(f'<div class="ai-response">{insights}</div>', unsafe_allow_html=True)
        
        if enable_health_analysis:
            st.subheader("🏥 Financial Health Analysis")
            with st.spinner("Analyzing financial health..."):
                health_analysis = financial_ai.analyze_financial_health(df)
                
                if "error" not in health_analysis:
                    # Display health status
                    health_color = {"Good": "🟢", "Moderate": "🟡", "Needs Attention": "🔴"}
                    st.markdown(f"**Overall Health:** {health_color.get(health_analysis['overall_health'], '⚪')} {health_analysis['overall_health']}")
                    
                    # Display key metrics
                    if health_analysis['key_metrics']:
                        st.write("**Key Metrics:**")
                        for metric, value in health_analysis['key_metrics'].items():
                            st.write(f"• {metric}: {value}")
                    
                    # Display recommendations
                    if health_analysis['recommendations']:
                        st.write("**💡 Recommendations:**")
                        for rec in health_analysis['recommendations']:
                            st.write(f"• {rec}")
        
        if enable_anomaly_detection:
            st.subheader("🔍 Anomaly Detection")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                anomaly_col = st.selectbox("Select column for anomaly detection", numeric_columns, key="anomaly_col")
                
                if st.button("�️ Detect Anomalies"):
                    with st.spinner("Detecting financial anomalies..."):
                        anomalies = financial_ai.detect_anomalies(df, anomaly_col)
                        
                        if "error" not in anomalies:
                            st.write(f"**Found {anomalies['total_anomalies']} anomalies ({anomalies['percentage']}% of data)**")
                            
                            if anomalies['anomalies']:
                                st.write("**🚨 Anomalies Detected:**")
                                for anomaly in anomalies['anomalies'][:5]:  # Show first 5
                                    st.write(f"• Value: ${anomaly['value']:,.2f} (Severity: {anomaly['severity']})")
        
        if enable_forecasting:
            st.subheader("🔮 Financial Forecasting")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                forecast_col = st.selectbox("Select metric to forecast", numeric_columns, key="forecast_col")
                forecast_periods = st.slider("Forecast periods", 3, 24, 12)
                
                if st.button("📈 Generate Forecast"):
                    with st.spinner("Generating financial forecast..."):
                        forecast = financial_ai.forecast_financial_metrics(df, forecast_col, forecast_periods)
                        
                        if "error" not in forecast:
                            st.success(f"Forecast Trend: {forecast['trend']}")
                            st.info(f"Model Accuracy: {forecast['model_score']:.1%}")
                            
                            # Show sample predictions
                            st.write("**📊 Sample Predictions:**")
                            for i, pred in enumerate(forecast['predictions'][:6], 1):
                                st.write(f"Period {i}: ${pred:,.2f}")
    
    # Enhanced Visualization section
    st.header("📊 Advanced Financial Visualizations")
    
    if len(df) > 0:
        if chart_type == "Financial Dashboard":
            # Create financial dashboard
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Revenue vs Expenses over time
                if 'Date' in df.columns and 'Revenue' in df.columns and 'Expenses' in df.columns:
                    fig_financial = go.Figure()
                    fig_financial.add_trace(go.Scatter(x=df['Date'], y=df['Revenue'], 
                                                     mode='lines', name='Revenue', line=dict(color='green')))
                    fig_financial.add_trace(go.Scatter(x=df['Date'], y=df['Expenses'], 
                                                     mode='lines', name='Expenses', line=dict(color='red')))
                    fig_financial.update_layout(title='Revenue vs Expenses Over Time', height=400)
                    st.plotly_chart(fig_financial, use_container_width=True)
            
            with viz_col2:
                # Profit margin analysis
                if 'Date' in df.columns and 'Profit_Margin' in df.columns:
                    fig_margin = px.line(df, x='Date', y='Profit_Margin', 
                                       title='Profit Margin Trend', color_discrete_sequence=['blue'])
                    fig_margin.update_layout(height=400)
                    st.plotly_chart(fig_margin, use_container_width=True)
        
        elif chart_type == "Ratio Analysis":
            # Financial ratios analysis
            ratios = financial_ai.calculate_financial_ratios(df)
            if ratios and 'calculation_error' not in ratios:
                ratio_df = pd.DataFrame(list(ratios.items()), columns=['Ratio', 'Value'])
                fig_ratios = px.bar(ratio_df, x='Ratio', y='Value', title='Financial Ratios Analysis')
                fig_ratios.update_layout(height=400)
                st.plotly_chart(fig_ratios, use_container_width=True)
        
        elif chart_type == "Trend Analysis":
            # Multi-metric trend analysis
            financial_cols = [col for col in df.columns if any(term in col.lower() 
                             for term in ['revenue', 'profit', 'cash', 'asset'])]
            if financial_cols and 'Date' in df.columns:
                selected_metrics = st.multiselect("Select metrics for trend analysis", 
                                                 financial_cols, default=financial_cols[:3])
                
                if selected_metrics:
                    fig_trends = go.Figure()
                    for metric in selected_metrics:
                        fig_trends.add_trace(go.Scatter(x=df['Date'], y=df[metric], 
                                                       mode='lines', name=metric))
                    fig_trends.update_layout(title='Multi-Metric Trend Analysis', height=400)
                    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("**🏦 AI-Powered Financial Accounting Dashboard** | Built with Streamlit, Advanced Analytics & Financial AI")
    
    # Add financial tips
    with st.expander("💡 Financial Analysis Tips"):
        st.markdown("""
        **Key Financial Metrics to Monitor:**
        - **Profit Margin**: Indicates how much profit you make per dollar of revenue
        - **Current Ratio**: Measures ability to pay short-term obligations
        - **Cash Flow**: Tracks actual cash movement in/out of business
        - **Revenue Growth**: Shows business expansion trends
        
        **AI Features:**
        - **Anomaly Detection**: Identifies unusual transactions or patterns
        - **Financial Forecasting**: Predicts future financial performance
        - **Health Analysis**: Comprehensive business health assessment
        - **Trend Analysis**: Long-term financial pattern recognition
        """)

if __name__ == "__main__":
    main()