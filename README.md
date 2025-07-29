# AI-Powered Analytics Dashboard

A comprehensive Streamlit application that combines data analytics with AI-powered insights and predictions.

## Features

🤖 **AI-Powered Analytics**
- Automatic data insights generation
- Trend prediction and forecasting
- Statistical analysis

📊 **Interactive Visualizations**
- Line charts, bar charts, scatter plots
- Correlation heatmaps
- Real-time data visualization

📈 **Data Analysis**
- CSV file upload support
- Sample data generation
- Multiple data source options

🎛️ **User-Friendly Interface**
- Responsive design
- Interactive sidebar controls
- Real-time metrics display

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up environment variables in `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Features Overview

### Data Sources
- **Sample Data**: Built-in demo dataset with sales, revenue, and customer data
- **CSV Upload**: Upload your own CSV files for analysis
- **API Connection**: Framework for connecting to external APIs (customizable)

### AI Capabilities
- **Automated Insights**: Statistical analysis and data profiling
- **Trend Prediction**: Linear trend analysis with future value predictions
- **Data Summarization**: Automatic generation of key metrics and findings

### Visualizations
- **Time Series Charts**: Track trends over time
- **Category Analysis**: Compare performance across different categories
- **Correlation Analysis**: Understand relationships between variables
- **Custom Charts**: Multiple chart types for different analysis needs

## Customization

The app is designed to be easily customizable:

1. **Add New AI Features**: Extend the `AIAnalytics` class
2. **Custom Visualizations**: Add new chart types in the visualization section
3. **Data Sources**: Implement additional data connection methods
4. **Styling**: Modify the CSS in the app for custom appearance

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- Requests
- Python-dotenv

## License

MIT License - feel free to use and modify as needed.
