"""
Financial Analysis Utilities for AI-Powered Accounting Dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any


class FinancialCalculator:
    """Advanced financial calculations and metrics"""
    
    @staticmethod
    def calculate_roi(initial_investment: float, final_value: float) -> float:
        """Calculate Return on Investment"""
        return ((final_value - initial_investment) / initial_investment) * 100
    
    @staticmethod
    def calculate_npv(cash_flows: List[float], discount_rate: float) -> float:
        """Calculate Net Present Value"""
        npv = 0
        for i, cash_flow in enumerate(cash_flows):
            npv += cash_flow / ((1 + discount_rate) ** i)
        return npv
    
    @staticmethod
    def calculate_irr(cash_flows: List[float], guess: float = 0.1) -> float:
        """Calculate Internal Rate of Return using Newton-Raphson method"""
        try:
            def npv_func(rate):
                return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
            
            def npv_derivative(rate):
                return sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows))
            
            rate = guess
            for _ in range(100):  # max iterations
                npv = npv_func(rate)
                if abs(npv) < 1e-6:
                    return rate * 100  # return as percentage
                derivative = npv_derivative(rate)
                if abs(derivative) < 1e-6:
                    break
                rate = rate - npv / derivative
            return rate * 100
        except:
            return None
    
    @staticmethod
    def calculate_break_even_point(fixed_costs: float, variable_cost_per_unit: float, 
                                 price_per_unit: float) -> float:
        """Calculate break-even point in units"""
        if price_per_unit <= variable_cost_per_unit:
            return float('inf')
        return fixed_costs / (price_per_unit - variable_cost_per_unit)
    
    @staticmethod
    def calculate_working_capital(current_assets: float, current_liabilities: float) -> float:
        """Calculate working capital"""
        return current_assets - current_liabilities
    
    @staticmethod
    def calculate_debt_to_equity(total_debt: float, total_equity: float) -> float:
        """Calculate debt-to-equity ratio"""
        if total_equity == 0:
            return float('inf')
        return total_debt / total_equity


class FinancialForecasting:
    """Advanced forecasting methods for financial data"""
    
    @staticmethod
    def moving_average_forecast(data: pd.Series, window: int = 3, periods: int = 12) -> List[float]:
        """Simple moving average forecast"""
        if len(data) < window:
            return [data.mean()] * periods
        
        last_values = data.tail(window).mean()
        return [last_values] * periods
    
    @staticmethod
    def exponential_smoothing_forecast(data: pd.Series, alpha: float = 0.3, 
                                     periods: int = 12) -> List[float]:
        """Exponential smoothing forecast"""
        if len(data) == 0:
            return [0] * periods
        
        # Initialize with first value
        smoothed = [data.iloc[0]]
        
        # Calculate exponentially smoothed values
        for i in range(1, len(data)):
            smoothed.append(alpha * data.iloc[i] + (1 - alpha) * smoothed[-1])
        
        # Forecast future values
        last_smoothed = smoothed[-1]
        return [last_smoothed] * periods
    
    @staticmethod
    def seasonal_decompose_forecast(data: pd.Series, periods: int = 12) -> List[float]:
        """Simple seasonal forecast based on historical patterns"""
        if len(data) < 12:
            return FinancialForecasting.moving_average_forecast(data, periods=periods)
        
        # Extract seasonal pattern (simple approach)
        seasonal_period = min(12, len(data) // 2)
        seasonal_pattern = []
        
        for i in range(seasonal_period):
            seasonal_values = [data[j] for j in range(i, len(data), seasonal_period)]
            seasonal_pattern.append(np.mean(seasonal_values))
        
        # Generate forecasts
        forecasts = []
        trend = (data.iloc[-1] - data.iloc[0]) / len(data)
        
        for i in range(periods):
            season_idx = i % len(seasonal_pattern)
            seasonal_component = seasonal_pattern[season_idx]
            trend_component = data.iloc[-1] + trend * (i + 1)
            forecasts.append(trend_component * (seasonal_component / np.mean(seasonal_pattern)))
        
        return forecasts


class FinancialRiskAnalysis:
    """Risk analysis and assessment tools"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        excess_returns = returns.mean() - risk_free_rate / 252  # daily risk-free rate
        return excess_returns / returns.std() * np.sqrt(252)  # annualized
    
    @staticmethod
    def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient"""
        if len(asset_returns) != len(market_returns) or market_returns.var() == 0:
            return 1.0
        return np.cov(asset_returns, market_returns)[0, 1] / market_returns.var()
    
    @staticmethod
    def assess_liquidity_risk(current_ratio: float, quick_ratio: float, 
                            cash_ratio: float) -> Dict[str, Any]:
        """Assess liquidity risk based on financial ratios"""
        risk_level = "Low"
        warnings = []
        
        if current_ratio < 1.0:
            risk_level = "High"
            warnings.append("Current ratio below 1.0 indicates potential liquidity issues")
        elif current_ratio < 1.5:
            risk_level = "Medium"
            warnings.append("Current ratio below 1.5 suggests tight liquidity")
        
        if quick_ratio < 0.8:
            if risk_level != "High":
                risk_level = "Medium"
            warnings.append("Quick ratio below 0.8 indicates potential short-term liquidity stress")
        
        if cash_ratio < 0.1:
            warnings.append("Very low cash ratio - limited immediate liquidity")
        
        return {
            "risk_level": risk_level,
            "warnings": warnings,
            "recommendations": FinancialRiskAnalysis._get_liquidity_recommendations(risk_level)
        }
    
    @staticmethod
    def _get_liquidity_recommendations(risk_level: str) -> List[str]:
        """Get liquidity risk recommendations"""
        if risk_level == "High":
            return [
                "Immediate action required to improve cash position",
                "Consider asset liquidation or emergency financing",
                "Review and extend payment terms with suppliers",
                "Accelerate collection of accounts receivable"
            ]
        elif risk_level == "Medium":
            return [
                "Monitor cash flow closely",
                "Consider establishing credit lines",
                "Optimize inventory management",
                "Review payment schedules"
            ]
        else:
            return [
                "Maintain current liquidity management practices",
                "Consider investment opportunities for excess cash",
                "Regular monitoring of liquidity ratios"
            ]


class FinancialReportGenerator:
    """Generate comprehensive financial reports"""
    
    @staticmethod
    def generate_executive_summary(financial_data: Dict[str, Any]) -> str:
        """Generate executive summary of financial performance"""
        summary = ["EXECUTIVE SUMMARY", "=" * 50]
        
        # Revenue analysis
        if 'total_revenue' in financial_data:
            summary.append(f"Total Revenue: ${financial_data['total_revenue']:,.2f}")
        
        if 'revenue_growth' in financial_data:
            growth = financial_data['revenue_growth']
            trend = "↗️" if growth > 0 else "↘️"
            summary.append(f"Revenue Growth: {growth:.1f}% {trend}")
        
        # Profitability
        if 'profit_margin' in financial_data:
            margin = financial_data['profit_margin']
            summary.append(f"Profit Margin: {margin:.1f}%")
        
        # Financial health
        if 'financial_health' in financial_data:
            summary.append(f"Financial Health: {financial_data['financial_health']}")
        
        return "\n".join(summary)
    
    @staticmethod
    def generate_ratio_analysis(ratios: Dict[str, float]) -> str:
        """Generate detailed ratio analysis report"""
        report = ["FINANCIAL RATIO ANALYSIS", "=" * 50]
        
        # Categorize ratios
        profitability_ratios = {k: v for k, v in ratios.items() 
                              if 'margin' in k.lower() or 'profit' in k.lower()}
        liquidity_ratios = {k: v for k, v in ratios.items() 
                           if 'current' in k.lower() or 'quick' in k.lower()}
        efficiency_ratios = {k: v for k, v in ratios.items() 
                            if 'turnover' in k.lower() or 'days' in k.lower()}
        
        # Add ratio categories to report
        if profitability_ratios:
            report.append("\nPROFITABILITY RATIOS:")
            for ratio, value in profitability_ratios.items():
                report.append(f"  {ratio}: {value:.2f}")
        
        if liquidity_ratios:
            report.append("\nLIQUIDITY RATIOS:")
            for ratio, value in liquidity_ratios.items():
                report.append(f"  {ratio}: {value:.2f}")
        
        if efficiency_ratios:
            report.append("\nEFFICIENCY RATIOS:")
            for ratio, value in efficiency_ratios.items():
                report.append(f"  {ratio}: {value:.2f}")
        
        return "\n".join(report)
