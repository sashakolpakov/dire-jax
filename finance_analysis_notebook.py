#!/usr/bin/env python3
"""
Jupyter Notebook-friendly version of the finance analysis.
Run this in Jupyter with: %run finance_analysis_notebook.py
Or import it: from finance_analysis_notebook import *
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Set plotly to notebook mode for better display
pio.renderers.default = "notebook"

# Import our modules
from finance_tick_hierarchical import HierarchicalTickEmbedder
from dire_jax.dire_pytorch import DiRePyTorch

# Initialize global analyzer
analyzer = HierarchicalTickEmbedder()

def quick_analysis(ticker='SPY', max_trades=50000, save_plots=False):
    """
    Quick analysis function for notebooks.
    Returns the embedding dataframe and displays plots.
    """
    print(f"Analyzing {ticker}...")
    
    # Run analysis
    df = analyzer.analyze_trading_session(ticker, max_trades=max_trades)
    
    if df is None:
        print("No data available!")
        return None
    
    # Create subplot figure with multiple views
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Colored by Time of Day',
            'Colored by Volatility',
            'Colored by Volume',
            'Colored by Trade Intensity'
        )
    )
    
    # Add time-colored scatter
    fig.add_trace(
        go.Scatter(
            x=df['embed_x'],
            y=df['embed_y'],
            mode='markers',
            marker=dict(
                size=5,
                color=df['time_of_day'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Hour", x=0.45, y=0.85)
            ),
            text=df.index.strftime('%H:%M:%S'),
            hovertemplate='Time: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}',
            name='Time'
        ),
        row=1, col=1
    )
    
    # Add volatility-colored scatter
    fig.add_trace(
        go.Scatter(
            x=df['embed_x'],
            y=df['embed_y'],
            mode='markers',
            marker=dict(
                size=5,
                color=df['volatility'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Vol", x=1.0, y=0.85)
            ),
            text=df['volatility'].round(4),
            hovertemplate='Volatility: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add volume-colored scatter
    fig.add_trace(
        go.Scatter(
            x=df['embed_x'],
            y=df['embed_y'],
            mode='markers',
            marker=dict(
                size=5,
                color=np.log10(df['volume'] + 1),
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Log Vol", x=0.45, y=0.35)
            ),
            text=df['volume'].astype(int),
            hovertemplate='Volume: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add trade intensity-colored scatter
    fig.add_trace(
        go.Scatter(
            x=df['embed_x'],
            y=df['embed_y'],
            mode='markers',
            marker=dict(
                size=5,
                color=df['trade_intensity'],
                colorscale='Greens',
                showscale=True,
                colorbar=dict(title="Intensity", x=1.0, y=0.35)
            ),
            text=df['trade_intensity'].round(2),
            hovertemplate='Trade Intensity: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Market Microstructure Embedding Analysis",
        height=800,
        showlegend=False,
        hovermode='closest'
    )
    
    # Update axes
    for i in range(1, 5):
        row = (i-1) // 2 + 1
        col = (i-1) % 2 + 1
        fig.update_xaxes(title_text="DIRE Dim 1", row=row, col=col)
        fig.update_yaxes(title_text="DIRE Dim 2", row=row, col=col)
    
    # Show the plot
    fig.show()
    
    # Save if requested
    if save_plots:
        filename = f"{ticker}_embedding_analysis.html"
        fig.write_html(filename)
        print(f"Saved plot to {filename}")
    
    # Print summary statistics
    print(f"\n{ticker} Summary Statistics:")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"Total periods: {len(df)}")
    print(f"Avg return: {df['return'].mean():.6f}")
    print(f"Avg volatility: {df['volatility'].mean():.4f}")
    print(f"Total volume: {df['volume'].sum():,.0f}")
    print(f"Peak volatility time: {df.loc[df['volatility'].idxmax()].name}")
    print(f"Peak volume time: {df.loc[df['volume'].idxmax()].name}")
    
    return df

def compare_tickers(tickers=['SPY', 'QQQ', 'IWM'], max_trades=30000):
    """
    Compare multiple tickers in a single embedding space.
    """
    all_data = []
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        df = analyzer.analyze_trading_session(ticker, max_trades=max_trades)
        if df is not None:
            df['ticker'] = ticker
            all_data.append(df)
    
    if not all_data:
        print("No data available!")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create comparison plot
    fig = px.scatter(
        combined_df,
        x='embed_x',
        y='embed_y',
        color='ticker',
        hover_data=['time_of_day', 'volatility', 'volume'],
        title='Multi-Asset Market Microstructure Comparison',
        labels={'embed_x': 'DIRE Dimension 1', 'embed_y': 'DIRE Dimension 2'}
    )
    
    fig.update_layout(height=600, width=900)
    fig.show()
    
    # Compute correlation of embeddings
    print("\nCross-Asset Embedding Correlations:")
    for i, ticker1 in enumerate(tickers):
        for ticker2 in tickers[i+1:]:
            data1 = combined_df[combined_df['ticker'] == ticker1][['embed_x', 'embed_y']].values
            data2 = combined_df[combined_df['ticker'] == ticker2][['embed_x', 'embed_y']].values
            
            # Align by taking minimum length
            min_len = min(len(data1), len(data2))
            if min_len > 0:
                corr_x = np.corrcoef(data1[:min_len, 0], data2[:min_len, 0])[0, 1]
                corr_y = np.corrcoef(data1[:min_len, 1], data2[:min_len, 1])[0, 1]
                print(f"{ticker1}-{ticker2}: X-corr={corr_x:.3f}, Y-corr={corr_y:.3f}")
    
    return combined_df

def find_anomalies(df, threshold=2.5):
    """
    Find anomalous periods in the embedding.
    """
    # Calculate distance from center
    center_x = df['embed_x'].mean()
    center_y = df['embed_y'].mean()
    
    df['distance_from_center'] = np.sqrt(
        (df['embed_x'] - center_x)**2 + 
        (df['embed_y'] - center_y)**2
    )
    
    # Find outliers (beyond threshold standard deviations)
    mean_dist = df['distance_from_center'].mean()
    std_dist = df['distance_from_center'].std()
    threshold_dist = mean_dist + threshold * std_dist
    
    anomalies = df[df['distance_from_center'] > threshold_dist].copy()
    anomalies = anomalies.sort_values('distance_from_center', ascending=False)
    
    print(f"Found {len(anomalies)} anomalous periods (>{threshold} std from center):")
    print("\nTop 10 Anomalies:")
    
    for idx, row in anomalies.head(10).iterrows():
        time_str = row.name.strftime('%H:%M:%S') if hasattr(row.name, 'strftime') else str(row.name)
        print(f"  {time_str}: volatility={row['volatility']:.4f}, "
              f"volume={row['volume']:.0f}, distance={row['distance_from_center']:.2f}")
    
    # Visualize anomalies
    fig = go.Figure()
    
    # Normal points
    normal = df[df['distance_from_center'] <= threshold_dist]
    fig.add_trace(go.Scatter(
        x=normal['embed_x'],
        y=normal['embed_y'],
        mode='markers',
        marker=dict(size=4, color='blue', opacity=0.5),
        name='Normal',
        text=normal.index.strftime('%H:%M:%S') if hasattr(normal.index, 'strftime') else normal.index,
        hovertemplate='Time: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}'
    ))
    
    # Anomalous points
    fig.add_trace(go.Scatter(
        x=anomalies['embed_x'],
        y=anomalies['embed_y'],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Anomalous',
        text=anomalies.index.strftime('%H:%M:%S') if hasattr(anomalies.index, 'strftime') else anomalies.index,
        hovertemplate='ANOMALY<br>Time: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}'
    ))
    
    fig.update_layout(
        title='Market Anomaly Detection',
        xaxis_title='DIRE Dimension 1',
        yaxis_title='DIRE Dimension 2',
        height=600
    )
    
    fig.show()
    
    return anomalies

def create_animation(df, window_size=50):
    """
    Create an animated plot showing market evolution through the day.
    """
    # Add frame numbers for animation
    df['frame'] = np.arange(len(df)) // window_size
    
    # Create animated scatter plot
    fig = px.scatter(
        df,
        x='embed_x',
        y='embed_y',
        animation_frame='frame',
        color='volatility',
        size='volume',
        hover_data=['time_of_day', 'return'],
        title='Market Evolution Animation',
        labels={'embed_x': 'DIRE Dimension 1', 'embed_y': 'DIRE Dimension 2'},
        color_continuous_scale='Viridis',
        size_max=15
    )
    
    fig.update_layout(height=600, width=800)
    
    # Slow down animation
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 500
    
    fig.show()
    
    return fig

# Print instructions when imported
print("""
Financial Market Analysis with DIRE - Notebook Functions
========================================================

Available functions:

1. quick_analysis(ticker='SPY', max_trades=50000, save_plots=False)
   - Analyze a single ticker with multiple visualizations
   - Returns embedding dataframe

2. compare_tickers(tickers=['SPY', 'QQQ', 'IWM'], max_trades=30000)
   - Compare multiple assets in the same embedding space
   - Shows cross-asset relationships

3. find_anomalies(df, threshold=2.5)
   - Identify anomalous market periods
   - Visualizes outliers in embedding space

4. create_animation(df, window_size=50)
   - Animated visualization of market evolution
   - Shows how patterns change through the day

Example usage:
--------------
# Analyze SPY
df_spy = quick_analysis('SPY')

# Find anomalies
anomalies = find_anomalies(df_spy)

# Compare multiple assets
comparison = compare_tickers(['SPY', 'AAPL', 'TSLA'])

# Create animation
anim = create_animation(df_spy)
""")