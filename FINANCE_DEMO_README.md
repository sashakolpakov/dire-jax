# Financial Market Microstructure Analysis with DIRE

This demo showcases how to use the PyTorch/PyKeOps DIRE implementation for analyzing financial tick data, revealing market microstructure patterns through dimensionality reduction.

## Overview

We provide two implementations that fetch real market data from Polygon.io and create embeddings that reveal hidden market patterns:

1. **Simple Version** (`finance_tick_embedding.py`) - Minute-bar data analysis
2. **Advanced Version** (`finance_tick_hierarchical.py`) - Tick-level data with hierarchical embedding

## Key Features

### Microstructure Features Extracted
- **Price Dynamics**: Returns, realized volatility, price ranges
- **Volume Patterns**: Total volume, trade size distribution, volume volatility
- **Market Microstructure**: 
  - Kyle's lambda (price impact coefficient)
  - Amihud illiquidity measure
  - Trade intensity (trades per second)
- **Regime Indicators**: Large trade ratios, volatility bursts

### What The Embeddings Reveal
1. **Time-based Structure**: Market open (9:30-10:00) and close (15:30-16:00) form distinct clusters
2. **Volatility Regimes**: High volatility periods cluster together
3. **Liquidity Patterns**: Lunch-time (12:00-13:00) shows different microstructure
4. **News Events**: Visible as sudden jumps in embedding space
5. **HFT vs Institutional Flow**: Different trading patterns separate in 2D

## Installation

```bash
# Install required packages
pip install polygon-api-client plotly scikit-learn

# Already installed from main setup:
# - torch
# - pykeops
```

## Running the Demos

### For Jupyter Notebook (Recommended for Headless Server)

Create a notebook and run:

```python
# Import and run the simple version
%run finance_tick_embedding.py

# Or run the advanced hierarchical version
%run finance_tick_hierarchical.py

# The embedding dataframe is returned, so you can analyze further:
import pandas as pd
import plotly.express as px

# After running, 'embedding' variable contains the results
if 'embedding' in locals():
    # Custom analysis
    print(f"Embedding shape: {embedding.shape}")
    
    # Create custom visualizations
    fig = px.scatter(
        embedding,
        x='x', 
        y='y',
        color='volume',
        animation_frame='hour',  # Animate by hour
        title='Market Evolution Throughout the Day'
    )
    fig.show()
```

### For Command Line with Saved Output

```python
# Modified version that saves plots as HTML
import plotly.express as px
import plotly.io as pio

# Run the analysis
result_df = analyzer.analyze_trading_session("SPY")

# Save plots as HTML files
fig = px.scatter(result_df, x='embed_x', y='embed_y', color='time_of_day')
pio.write_html(fig, file='spy_embedding_time.html')

# View on headless server by downloading the HTML files
```

## Example Analysis Pipeline

```python
from finance_tick_hierarchical import HierarchicalTickEmbedder
import pandas as pd
import numpy as np

# Initialize analyzer
analyzer = HierarchicalTickEmbedder()

# Analyze different tickers
tickers = ['SPY', 'AAPL', 'TSLA', 'QQQ']
results = {}

for ticker in tickers:
    print(f"Analyzing {ticker}...")
    result_df = analyzer.analyze_trading_session(
        ticker=ticker,
        max_trades=50000  # Limit for faster processing
    )
    results[ticker] = result_df

# Compare volatility patterns across stocks
for ticker, df in results.items():
    volatility_cluster = df.groupby('regime')['volatility'].mean()
    print(f"\n{ticker} Volatility by Regime:")
    print(volatility_cluster)

# Find anomalous time periods
for ticker, df in results.items():
    # Points far from center might be anomalous
    center_x, center_y = df['embed_x'].mean(), df['embed_y'].mean()
    df['distance_from_center'] = np.sqrt(
        (df['embed_x'] - center_x)**2 + 
        (df['embed_y'] - center_y)**2
    )
    
    anomalies = df.nlargest(10, 'distance_from_center')
    print(f"\n{ticker} Anomalous Periods:")
    print(anomalies[['time_of_day', 'volatility', 'volume', 'distance_from_center']])
```

## Interactive Notebook Workflow

```python
# Cell 1: Setup
from finance_tick_hierarchical import HierarchicalTickEmbedder
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

analyzer = HierarchicalTickEmbedder()

# Cell 2: Create Interactive Widget
ticker_widget = widgets.Dropdown(
    options=['SPY', 'AAPL', 'TSLA', 'QQQ', 'NVDA'],
    value='SPY',
    description='Ticker:'
)

color_widget = widgets.Dropdown(
    options=['time_of_day', 'volatility', 'volume', 'trade_intensity', 'regime'],
    value='time_of_day',
    description='Color by:'
)

def update_plot(ticker, color_by):
    # Fetch and analyze
    df = analyzer.analyze_trading_session(ticker, max_trades=50000)
    
    if df is not None:
        # Create plot
        fig = px.scatter(
            df, 
            x='embed_x', 
            y='embed_y',
            color=color_by,
            hover_data=['time_of_day', 'return', 'volatility', 'volume'],
            title=f'{ticker} Market Microstructure'
        )
        fig.show()
        
        # Summary stats
        print(f"\n{ticker} Statistics:")
        print(f"Total points: {len(df)}")
        print(f"Volatility range: {df['volatility'].min():.4f} - {df['volatility'].max():.4f}")
        print(f"Volume range: {df['volume'].min():.0f} - {df['volume'].max():.0f}")

# Cell 3: Run Interactive Analysis
interactive_plot = widgets.interactive(
    update_plot,
    ticker=ticker_widget,
    color_by=color_widget
)

display(interactive_plot)
```

## Understanding the Results

### Typical Patterns

1. **U-Shaped Volatility**: Higher volatility at market open and close
2. **Lunch Effect**: Reduced liquidity and different patterns 12:00-13:00
3. **Friday Afternoon**: Often shows distinct patterns (position squaring)
4. **News Clusters**: Sudden regime changes indicate news events

### Using for Trading Insights

```python
# Identify stable vs unstable periods
def classify_market_stability(embedding_df):
    # Calculate local density in embedding space
    from sklearn.neighbors import KernelDensity
    
    kde = KernelDensity(bandwidth=0.5)
    X = embedding_df[['embed_x', 'embed_y']].values
    kde.fit(X)
    
    # Score each point
    log_density = kde.score_samples(X)
    embedding_df['stability_score'] = np.exp(log_density)
    
    # High density = stable, Low density = unstable/transitional
    stable_periods = embedding_df[embedding_df['stability_score'] > 
                                  embedding_df['stability_score'].median()]
    
    print("Stable periods characteristics:")
    print(f"Avg volatility: {stable_periods['volatility'].mean():.4f}")
    print(f"Avg volume: {stable_periods['volume'].mean():.0f}")
    
    return embedding_df

# Run stability analysis
enhanced_df = classify_market_stability(result_df)
```

## Performance Notes

- **Minute bars**: Handles full day instantly (~390 points)
- **10-second bars**: Handles full day in seconds (~2,340 points)  
- **Tick data**: Can process 100K+ trades with hierarchical approach
- **Real-time capable**: Fast enough for live market monitoring

## Extending the Analysis

### Add Custom Features

```python
def add_custom_features(trades_df):
    # Add your domain-specific features
    trades_df['vwap_deviation'] = (trades_df['price'] - trades_df['vwap']) / trades_df['vwap']
    trades_df['momentum'] = trades_df['price'].pct_change(20)  # 20-period momentum
    # ... more custom features
    return trades_df
```

### Multi-Asset Analysis

```python
# Embed multiple assets in the same space
all_assets = pd.concat([
    analyzer.analyze_trading_session('SPY').assign(ticker='SPY'),
    analyzer.analyze_trading_session('TLT').assign(ticker='TLT'),  # Bonds
    analyzer.analyze_trading_session('GLD').assign(ticker='GLD'),  # Gold
])

# See cross-asset relationships
fig = px.scatter(
    all_assets,
    x='embed_x',
    y='embed_y', 
    color='ticker',
    title='Cross-Asset Market Regimes'
)
```

## API Key Note

The included API key is for demo purposes. For production use, get your own free key at https://polygon.io/

## Troubleshooting

1. **No data returned**: Markets are closed on weekends. The script automatically adjusts to the last trading day.
2. **Rate limits**: The free Polygon tier has rate limits. Add delays if hitting limits.
3. **Memory issues**: Reduce `max_trades` parameter or use larger time windows for aggregation.

## Next Steps

1. **Pattern Library**: Build a library of known patterns (news events, option expiries, etc.)
2. **Anomaly Detection**: Use embedding distance for real-time anomaly detection
3. **Regime Prediction**: Train models on embedding transitions to predict regime changes
4. **Cross-Asset Analysis**: Embed multiple assets together to find relationships
5. **Options Integration**: Add options flow data for fuller market picture

The speed of DIRE with PyKeOps makes this suitable for real-time trading systems!