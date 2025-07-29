"""
Configuration settings for AI-Powered Financial Accounting Dashboard
"""

# Financial Analysis Configuration
FINANCIAL_CONFIG = {
    # Risk Assessment Thresholds
    "risk_thresholds": {
        "current_ratio": {
            "low_risk": 2.0,
            "medium_risk": 1.5,
            "high_risk": 1.0
        },
        "debt_to_equity": {
            "low_risk": 0.3,
            "medium_risk": 0.6,
            "high_risk": 1.0
        },
        "profit_margin": {
            "excellent": 20.0,
            "good": 10.0,
            "acceptable": 5.0,
            "poor": 0.0
        }
    },
    
    # AI Model Parameters
    "ai_parameters": {
        "anomaly_detection": {
            "z_score_threshold": 2.5,
            "sensitivity": "medium"  # low, medium, high
        },
        "forecasting": {
            "default_periods": 12,
            "confidence_level": 0.95,
            "seasonal_adjustment": True
        }
    },
    
    # Financial Metrics Categories
    "metric_categories": {
        "profitability": [
            "revenue", "profit", "income", "margin", "earnings"
        ],
        "liquidity": [
            "cash", "current", "quick", "working_capital"
        ],
        "efficiency": [
            "turnover", "days", "cycle", "utilization"
        ],
        "leverage": [
            "debt", "equity", "leverage", "coverage"
        ],
        "market": [
            "price", "market", "value", "yield"
        ]
    },
    
    # Industry Benchmarks (example values)
    "industry_benchmarks": {
        "retail": {
            "current_ratio": 1.5,
            "profit_margin": 5.0,
            "inventory_turnover": 8.0
        },
        "manufacturing": {
            "current_ratio": 1.8,
            "profit_margin": 8.0,
            "asset_turnover": 1.2
        },
        "technology": {
            "current_ratio": 2.5,
            "profit_margin": 15.0,
            "roe": 20.0
        },
        "services": {
            "current_ratio": 1.3,
            "profit_margin": 12.0,
            "asset_turnover": 2.0
        }
    },
    
    # Alert Thresholds
    "alert_thresholds": {
        "cash_flow_negative_months": 3,
        "revenue_decline_percentage": -10.0,
        "expense_increase_percentage": 20.0,
        "anomaly_percentage": 5.0
    },
    
    # Chart and Visualization Settings
    "visualization": {
        "color_scheme": {
            "profit": "#2E8B57",      # Sea Green
            "loss": "#DC143C",        # Crimson
            "revenue": "#4169E1",     # Royal Blue
            "expenses": "#FF6347",    # Tomato
            "cash_flow": "#32CD32",   # Lime Green
            "assets": "#9370DB",      # Medium Purple
            "liabilities": "#FF69B4"  # Hot Pink
        },
        "chart_height": 400,
        "dashboard_layout": "wide"
    },
    
    # Data Validation Rules
    "validation_rules": {
        "required_columns": {
            "basic": ["Date", "Revenue", "Expenses"],
            "advanced": ["Date", "Revenue", "Expenses", "Cash_Flow", "Assets", "Liabilities"],
            "comprehensive": [
                "Date", "Revenue", "Expenses", "Cash_Flow", 
                "Current_Assets", "Current_Liabilities", "Total_Assets", 
                "Total_Liabilities", "Equity"
            ]
        },
        "data_types": {
            "Date": "datetime",
            "Revenue": "numeric",
            "Expenses": "numeric",
            "Cash_Flow": "numeric"
        }
    }
}

# AI Prompt Templates
AI_PROMPTS = {
    "financial_insight": """
    Analyze the following financial data and provide insights on:
    1. Overall financial health
    2. Key trends and patterns
    3. Areas of concern
    4. Opportunities for improvement
    5. Recommended actions
    
    Data summary: {data_summary}
    """,
    
    "risk_assessment": """
    Assess the financial risk profile based on:
    1. Liquidity ratios
    2. Profitability metrics
    3. Debt levels
    4. Cash flow patterns
    5. Market conditions
    
    Current metrics: {financial_metrics}
    """,
    
    "forecast_analysis": """
    Based on historical financial data, provide:
    1. Revenue forecasting
    2. Expense projections
    3. Cash flow predictions
    4. Growth opportunities
    5. Risk factors
    
    Historical trends: {trend_analysis}
    """
}

# Feature Flags
FEATURE_FLAGS = {
    "enable_ai_insights": True,
    "enable_anomaly_detection": True,
    "enable_forecasting": True,
    "enable_risk_analysis": True,
    "enable_benchmarking": True,
    "enable_real_time_alerts": True,
    "enable_export_reports": True,
    "enable_api_integration": False,  # Future feature
    "enable_multi_currency": False   # Future feature
}

# Application Settings
APP_SETTINGS = {
    "app_title": "AI-Powered Financial Accounting Dashboard",
    "app_icon": "🏦",
    "layout": "wide",
    "sidebar_state": "expanded",
    "theme": "light",  # light, dark, auto
    "refresh_interval": 300,  # seconds
    "max_file_size": 200,  # MB
    "supported_file_types": ["csv", "xlsx", "json"],
    "timezone": "UTC"
}
