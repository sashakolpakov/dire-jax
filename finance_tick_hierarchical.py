#!/usr/bin/env python3

"""
Advanced: Hierarchical DIRE for high-frequency tick data analysis.
Handles millions of ticks using the hierarchical approach.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from polygon import RESTClient
except ImportError:
    print("Installing polygon-api-client...")
    import subprocess
    subprocess.check_call(["pip", "install", "polygon-api-client"])
    from polygon import RESTClient

from dire_jax.dire_pytorch import DiRePyTorch

API_KEY = "BBRlVSAOiNqeJ77yjnveFBqZZTOFv2gN"


class HierarchicalTickEmbedder:
    """
    Multi-scale embedding of tick data using hierarchical DIRE.
    """
    
    def __init__(self, api_key=API_KEY):
        self.client = RESTClient(api_key)
        
        # Different embedders for different scales
        self.coarse_embedder = DiRePyTorch(
            n_components=2,
            max_iter_layout=16,  # Fewer iterations for coarse
            verbose=False
        )
        
        self.fine_embedder = DiRePyTorch(
            n_components=2,
            max_iter_layout=32,
            verbose=False
        )
    
    def fetch_trades_for_date(self, ticker: str, date_str: str, limit: int = None) -> pd.DataFrame:
        """
        Fetch all trades for a given date.
        """
        print(f"Fetching trades for {ticker} on {date_str}...")
        
        all_trades = []
        timestamp_gte = f"{date_str}T09:30:00Z"
        timestamp_lt = f"{date_str}T16:00:00Z"
        
        # Polygon API pagination
        trades_iter = self.client.list_trades(
            ticker=ticker,
            timestamp_gte=timestamp_gte,
            timestamp_lt=timestamp_lt,
            limit=limit if limit else 50000,
            order='asc'
        )
        
        for trade in trades_iter:
            all_trades.append({
                'timestamp': trade.participant_timestamp if hasattr(trade, 'participant_timestamp') else trade.sip_timestamp,
                'price': trade.price,
                'size': trade.size,
                'exchange': trade.exchange if hasattr(trade, 'exchange') else 0,
                'conditions': ','.join(map(str, trade.conditions)) if hasattr(trade, 'conditions') and trade.conditions else ''
            })
            
            if limit and len(all_trades) >= limit:
                break
        
        df = pd.DataFrame(all_trades)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            df = df.set_index('timestamp').sort_index()
            
            # Compute additional features
            df['log_price'] = np.log(df['price'])
            df['log_size'] = np.log(df['size'] + 1)
            df['price_change'] = df['price'].diff()
            df['time_diff'] = df.index.to_series().diff().dt.total_seconds()
            df['trade_intensity'] = 1 / (df['time_diff'] + 0.001)  # Trades per second
            
        print(f"Fetched {len(df)} trades")
        return df
    
    def compute_rolling_features(self, trades_df: pd.DataFrame, window: str = '10S') -> pd.DataFrame:
        """
        Compute rolling window features from tick data.
        Each row becomes a point to embed.
        """
        # Resample to fixed time windows
        agg_dict = {
            'price': ['first', 'last', 'mean', 'std', 'min', 'max'],
            'size': ['sum', 'mean', 'std', 'max', 'count'],
            'log_size': ['mean', 'std'],
            'trade_intensity': ['mean', 'max']
        }
        
        rolled = trades_df.resample(window).agg(agg_dict)
        rolled.columns = ['_'.join(col).strip() for col in rolled.columns]
        rolled = rolled.dropna()
        
        # Compute additional derived features
        features_df = pd.DataFrame(index=rolled.index)
        
        # Price features
        features_df['return'] = np.log(rolled['price_last'] / rolled['price_first'])
        features_df['volatility'] = rolled['price_std'] / rolled['price_mean']
        features_df['range'] = (rolled['price_max'] - rolled['price_min']) / rolled['price_mean']
        features_df['mid_price'] = (rolled['price_max'] + rolled['price_min']) / 2
        
        # Volume features
        features_df['volume'] = rolled['size_sum']
        features_df['avg_trade_size'] = rolled['size_mean']
        features_df['max_trade_size'] = rolled['size_max']
        features_df['trade_count'] = rolled['size_count']
        features_df['volume_volatility'] = rolled['size_std'] / (rolled['size_mean'] + 1e-8)
        
        # Microstructure features
        features_df['large_trade_ratio'] = rolled['size_max'] / (rolled['size_sum'] + 1e-8)
        features_df['trade_intensity'] = rolled['trade_intensity_mean']
        features_df['intensity_burst'] = rolled['trade_intensity_max']
        
        # Kyle's lambda proxy (price impact)
        features_df['kyle_lambda'] = np.abs(features_df['return']) / (np.log(features_df['volume'] + 1) + 1e-8)
        
        # Amihud illiquidity
        features_df['illiquidity'] = np.abs(features_df['return']) / (features_df['volume'] + 1)
        
        return features_df
    
    def hierarchical_embedding(self, features_df: pd.DataFrame, 
                              levels: List[int] = [100, 1000, 10000]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Perform hierarchical embedding at multiple scales.
        """
        n_total = len(features_df)
        features = features_df.values
        
        # Normalize features
        features_norm = (features - np.nanmean(features, axis=0)) / (np.nanstd(features, axis=0) + 1e-8)
        features_norm = np.nan_to_num(features_norm, 0)
        
        embeddings = []
        indices_per_level = []
        
        print(f"Hierarchical embedding of {n_total} points...")
        
        for level_idx, n_points in enumerate(levels):
            if n_points > n_total:
                n_points = n_total
            
            print(f"  Level {level_idx}: {n_points} points")
            
            # Smart sampling for this level
            if level_idx == 0:
                # First level: uniform sampling
                indices = np.linspace(0, n_total-1, n_points, dtype=int)
            else:
                # Subsequent levels: guided by previous embedding
                prev_embedding = embeddings[-1]
                prev_indices = indices_per_level[-1]
                
                # Find points far from existing samples for diversity
                indices = self.diversity_sampling(features_norm, prev_indices, n_points)
            
            # Get features for this level
            level_features = features_norm[indices]
            
            # Embed this level
            if n_points <= 50000:
                # Use PyKeOps for smaller levels
                embedding = self.coarse_embedder.fit_transform(level_features)
            else:
                # Use fine embedder for larger levels
                embedding = self.fine_embedder.fit_transform(level_features)
            
            embeddings.append(embedding)
            indices_per_level.append(indices)
        
        # Propagate to all points using the finest level
        final_embedding = self.propagate_embedding(
            features_norm, 
            indices_per_level[-1], 
            embeddings[-1]
        )
        
        return final_embedding, embeddings
    
    def diversity_sampling(self, features: np.ndarray, 
                          existing_indices: np.ndarray, 
                          n_target: int) -> np.ndarray:
        """
        Sample points that are diverse from existing samples.
        """
        n_total = len(features)
        n_existing = len(existing_indices)
        n_new = n_target - n_existing
        
        if n_new <= 0:
            return existing_indices[:n_target]
        
        # Compute distances to nearest existing point
        remaining_indices = np.setdiff1d(np.arange(n_total), existing_indices)
        
        if len(remaining_indices) <= n_new:
            return np.concatenate([existing_indices, remaining_indices])
        
        # Simple diversity: random sampling from remaining
        # (Could be improved with actual distance calculations)
        new_indices = np.random.choice(remaining_indices, n_new, replace=False)
        
        return np.concatenate([existing_indices, new_indices])
    
    def propagate_embedding(self, features: np.ndarray, 
                           anchor_indices: np.ndarray,
                           anchor_embedding: np.ndarray) -> np.ndarray:
        """
        Propagate embedding from anchors to all points.
        """
        n_total = len(features)
        final_embedding = np.zeros((n_total, 2))
        
        # Place anchors
        final_embedding[anchor_indices] = anchor_embedding
        
        # For non-anchor points, find nearest anchor and use its embedding
        # (with small noise to avoid overlap)
        non_anchor_mask = np.ones(n_total, dtype=bool)
        non_anchor_mask[anchor_indices] = False
        non_anchor_indices = np.where(non_anchor_mask)[0]
        
        if len(non_anchor_indices) > 0:
            # Simple nearest neighbor assignment
            anchor_features = features[anchor_indices]
            
            for idx in non_anchor_indices:
                # Find nearest anchor
                distances = np.sum((anchor_features - features[idx])**2, axis=1)
                nearest_anchor_idx = np.argmin(distances)
                
                # Use nearest anchor's embedding with small perturbation
                final_embedding[idx] = anchor_embedding[nearest_anchor_idx] + np.random.randn(2) * 0.01
        
        return final_embedding
    
    def analyze_trading_session(self, ticker: str = "SPY", 
                               date_str: str = None,
                               max_trades: int = None) -> pd.DataFrame:
        """
        Complete pipeline for analyzing a trading session.
        """
        if not date_str:
            # Use last trading day
            today = datetime.now()
            if today.weekday() == 5:  # Saturday
                date_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            elif today.weekday() == 6:  # Sunday
                date_str = (today - timedelta(days=2)).strftime("%Y-%m-%d")
            else:
                date_str = today.strftime("%Y-%m-%d")
        
        # Fetch trades
        trades_df = self.fetch_trades_for_date(ticker, date_str, limit=max_trades)
        
        if trades_df.empty:
            print("No trades found!")
            return None
        
        # Compute features at different time scales
        print("\nComputing multi-scale features...")
        
        # 10-second windows for fine detail
        features_10s = self.compute_rolling_features(trades_df, '10S')
        
        # 1-minute windows for medium scale
        features_1m = self.compute_rolling_features(trades_df, '60S')
        
        # 5-minute windows for coarse scale
        features_5m = self.compute_rolling_features(trades_df, '300S')
        
        print(f"Feature dimensions: 10s={len(features_10s)}, 1m={len(features_1m)}, 5m={len(features_5m)}")
        
        # Perform hierarchical embedding on finest scale
        if len(features_10s) > 100:
            # Determine hierarchy levels based on data size
            n_points = len(features_10s)
            if n_points < 1000:
                levels = [50, n_points]
            elif n_points < 10000:
                levels = [100, 500, n_points]
            else:
                levels = [100, 1000, 5000, n_points]
            
            embedding, level_embeddings = self.hierarchical_embedding(features_10s, levels)
            
            # Create result dataframe
            result_df = features_10s.copy()
            result_df['embed_x'] = embedding[:, 0]
            result_df['embed_y'] = embedding[:, 1]
            
            # Add time-based features for visualization
            result_df['hour'] = result_df.index.hour
            result_df['minute'] = result_df.index.minute
            result_df['seconds'] = result_df.index.second
            result_df['time_of_day'] = result_df['hour'] + result_df['minute']/60 + result_df['seconds']/3600
            
            return result_df
        else:
            print("Not enough data for embedding")
            return None
    
    def identify_market_regimes(self, embedding_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify market regimes using clustering on the embedding.
        """
        from sklearn.cluster import DBSCAN
        
        # Cluster the embedding
        X = embedding_df[['embed_x', 'embed_y']].values
        
        # DBSCAN for regime identification
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(X)
        embedding_df['regime'] = clustering.labels_
        
        # Identify regime characteristics
        regime_stats = []
        for regime in embedding_df['regime'].unique():
            if regime == -1:  # Noise points
                continue
            
            regime_data = embedding_df[embedding_df['regime'] == regime]
            
            stats = {
                'regime': regime,
                'count': len(regime_data),
                'avg_return': regime_data['return'].mean(),
                'avg_volatility': regime_data['volatility'].mean(),
                'avg_volume': regime_data['volume'].mean(),
                'avg_trade_intensity': regime_data['trade_intensity'].mean(),
                'time_range': f"{regime_data.index[0].strftime('%H:%M')} - {regime_data.index[-1].strftime('%H:%M')}"
            }
            regime_stats.append(stats)
        
        regime_df = pd.DataFrame(regime_stats)
        
        print("\nIdentified Market Regimes:")
        print(regime_df.to_string())
        
        return embedding_df


def main():
    """
    Demo of hierarchical tick data embedding.
    """
    print("="*70)
    print("Hierarchical Financial Tick Data Analysis with DIRE")
    print("="*70)
    
    analyzer = HierarchicalTickEmbedder()
    
    # Analyze a liquid stock/ETF
    ticker = "AAPL"  # Can also try "SPY", "TSLA", "QQQ"
    
    print(f"\nAnalyzing {ticker}...")
    result_df = analyzer.analyze_trading_session(
        ticker=ticker,
        max_trades=100000  # Limit for demo
    )
    
    if result_df is not None:
        print(f"\nEmbedding complete!")
        print(f"Data shape: {result_df.shape}")
        
        # Identify regimes
        result_df = analyzer.identify_market_regimes(result_df)
        
        # Visualize
        import plotly.express as px
        
        # Plot 1: Colored by time of day
        fig1 = px.scatter(
            result_df,
            x='embed_x',
            y='embed_y',
            color='time_of_day',
            hover_data=['return', 'volatility', 'volume', 'trade_intensity'],
            title=f'{ticker} Market Microstructure - Colored by Time',
            labels={'embed_x': 'DIRE Dimension 1', 'embed_y': 'DIRE Dimension 2'}
        )
        fig1.show()
        
        # Plot 2: Colored by regime
        fig2 = px.scatter(
            result_df,
            x='embed_x',
            y='embed_y',
            color='regime',
            hover_data=['time_of_day', 'return', 'volatility', 'volume'],
            title=f'{ticker} Market Regimes',
            labels={'embed_x': 'DIRE Dimension 1', 'embed_y': 'DIRE Dimension 2'}
        )
        fig2.show()
        
        # Plot 3: Colored by volatility
        fig3 = px.scatter(
            result_df,
            x='embed_x',
            y='embed_y',
            color='volatility',
            hover_data=['time_of_day', 'volume', 'trade_intensity'],
            title=f'{ticker} Volatility Clusters',
            labels={'embed_x': 'DIRE Dimension 1', 'embed_y': 'DIRE Dimension 2'},
            color_continuous_scale='Viridis'
        )
        fig3.show()
        
        print("\nVisualization complete! Look for:")
        print("1. Opening auction effects (9:30-10:00)")
        print("2. Lunch-time liquidity drop (12:00-13:00)")  
        print("3. Closing auction preparation (15:30-16:00)")
        print("4. News-driven volatility clusters")
        print("5. HFT vs institutional trading regimes")
        
        return result_df
    
    return None


if __name__ == "__main__":
    result = main()