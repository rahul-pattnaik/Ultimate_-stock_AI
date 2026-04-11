"""
Advanced Peer Comparison & AI-Powered Benchmarking
Features: Sector analysis, correlation matrices, clustering, relative strength
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComparisonMetric(Enum):
    """Key metrics for comparison"""
    PE_RATIO = "pe_ratio"
    PB_RATIO = "pb_ratio"
    PS_RATIO = "ps_ratio"
    ROE = "roe"
    ROA = "roa"
    DEBT_TO_EQUITY = "debt_to_equity"
    CURRENT_RATIO = "current_ratio"
    FCF_MARGIN = "fcf_margin"
    REVENUE_GROWTH = "revenue_growth"
    EPS_GROWTH = "eps_growth"
    DIVIDEND_YIELD = "dividend_yield"
    PRICE_MOMENTUM = "price_momentum"


@dataclass
class PeerMetrics:
    """Peer company metrics snapshot"""
    ticker: str
    company_name: str
    sector: str
    industry: str
    market_cap: float
    metrics: Dict[ComparisonMetric, float]
    growth_rate: float
    quality_score: float  # 0-100

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap,
            'quality_score': round(self.quality_score, 1),
            'growth_rate': round(self.growth_rate, 2)
        }


@dataclass
class ComparisonResult:
    """Peer comparison analysis result"""
    target_ticker: str
    peer_list: List[PeerMetrics]
    percentile_rank: Dict[str, float]  # Metric -> percentile (0-100)
    relative_strength: Dict[str, float]  # How strong target is vs peers
    recommendation: str
    analysis_summary: str


class PeerComparator:
    """Advanced peer comparison and analysis"""

    def __init__(self):
        self.peer_database: Dict[str, PeerMetrics] = {}
        self.sector_cache: Dict[str, List[str]] = {}

    def add_peer(self, peer_metrics: PeerMetrics) -> None:
        """Add peer to database"""
        self.peer_database[peer_metrics.ticker] = peer_metrics
        
        # Update sector cache
        sector = peer_metrics.sector
        if sector not in self.sector_cache:
            self.sector_cache[sector] = []
        if peer_metrics.ticker not in self.sector_cache[sector]:
            self.sector_cache[sector].append(peer_metrics.ticker)

    def get_sector_peers(self, sector: str) -> List[PeerMetrics]:
        """Get all peers in a sector"""
        if sector not in self.sector_cache:
            return []
        return [self.peer_database[ticker] for ticker in self.sector_cache[sector]]

    def get_industry_peers(self, industry: str) -> List[PeerMetrics]:
        """Get all peers in an industry"""
        return [p for p in self.peer_database.values() if p.industry == industry]

    def calculate_percentile(self, target_ticker: str, metric: ComparisonMetric) -> float:
        """Calculate percentile rank for a metric"""
        if target_ticker not in self.peer_database:
            return 0.0

        target_peer = self.peer_database[target_ticker]
        target_value = target_peer.metrics.get(metric)

        if target_value is None:
            return 0.0

        # Get all values for comparison
        peers = self.get_sector_peers(target_peer.sector)
        all_values = [p.metrics.get(metric) for p in peers if p.metrics.get(metric) is not None]

        if not all_values:
            return 50.0

        # Handle different metric directions
        if metric in [ComparisonMetric.DEBT_TO_EQUITY]:
            # Lower is better
            rank = sum(1 for v in all_values if v >= target_value) / len(all_values) * 100
        else:
            # Higher is better
            rank = sum(1 for v in all_values if v <= target_value) / len(all_values) * 100

        return round(rank, 1)

    def calculate_relative_strength(self, target_ticker: str) -> Dict[str, float]:
        """Calculate relative strength scores vs peers"""
        if target_ticker not in self.peer_database:
            return {}

        target_peer = self.peer_database[target_ticker]
        peers = self.get_sector_peers(target_peer.sector)

        relative_strength = {}

        for metric in ComparisonMetric:
            target_value = target_peer.metrics.get(metric)
            if target_value is None:
                continue

            peer_values = [p.metrics.get(metric) for p in peers if p.metrics.get(metric) is not None]
            if not peer_values:
                continue

            # Calculate relative strength (target vs avg peer)
            peer_avg = np.mean(peer_values)
            
            if metric in [ComparisonMetric.DEBT_TO_EQUITY]:
                # Lower is better
                strength = (peer_avg - target_value) / (abs(peer_avg) + 1e-10) * 100
            else:
                # Higher is better
                strength = (target_value - peer_avg) / (abs(peer_avg) + 1e-10) * 100

            relative_strength[metric.value] = round(strength, 2)

        return relative_strength

    def comprehensive_comparison(self, target_ticker: str, limit: int = 10) -> Optional[ComparisonResult]:
        """Generate comprehensive peer comparison"""
        if target_ticker not in self.peer_database:
            logger.warning(f"Target ticker {target_ticker} not found")
            return None

        target_peer = self.peer_database[target_ticker]
        peers = self.get_sector_peers(target_peer.sector)
        
        # Sort by market cap similarity
        peers.sort(key=lambda p: abs(p.market_cap - target_peer.market_cap))
        peer_list = [p for p in peers if p.ticker != target_ticker][:limit]

        # Calculate percentile ranks
        percentile_rank = {}
        for metric in ComparisonMetric:
            percentile_rank[metric.value] = self.calculate_percentile(target_ticker, metric)

        # Calculate relative strength
        relative_strength = self.calculate_relative_strength(target_ticker)

        # Generate recommendation
        avg_percentile = np.mean(list(percentile_rank.values()))
        if avg_percentile > 75:
            recommendation = "OUTPERFORMER"
        elif avg_percentile > 50:
            recommendation = "IN LINE"
        else:
            recommendation = "UNDERPERFORMER"

        analysis_summary = f"{target_ticker} ranks at {avg_percentile:.0f}th percentile vs sector peers"

        return ComparisonResult(
            target_ticker=target_ticker,
            peer_list=peer_list,
            percentile_rank=percentile_rank,
            relative_strength=relative_strength,
            recommendation=recommendation,
            analysis_summary=analysis_summary
        )


class ClusteringAnalyzer:
    """ML-based peer clustering and grouping"""

    @staticmethod
    def normalize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize metrics for clustering (z-score normalization)"""
        return (metrics_df - metrics_df.mean()) / (metrics_df.std() + 1e-10)

    @staticmethod
    def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate Euclidean distance"""
        return np.sqrt(np.sum((v1 - v2) ** 2))

    @staticmethod
    def simple_kmeans(data: np.ndarray, k: int = 3, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Simple K-means clustering"""
        n_samples, n_features = data.shape

        # Random initialization
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = data[indices]

        for iteration in range(max_iter):
            # Assign clusters
            distances = np.zeros((n_samples, k))
            for i in range(k):
                distances[:, i] = np.sqrt(np.sum((data - centroids[i]) ** 2, axis=1))
            
            clusters = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = clusters == i
                if mask.sum() > 0:
                    new_centroids[i] = data[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return clusters, centroids

    @staticmethod
    def correlation_matrix(peer_metrics_list: List[PeerMetrics]) -> pd.DataFrame:
        """Calculate correlation matrix of peer metrics"""
        data = []
        tickers = []

        for peer in peer_metrics_list:
            metric_values = [v for v in peer.metrics.values() if v is not None and not np.isnan(v)]
            data.append(metric_values)
            tickers.append(peer.ticker)

        if not data:
            return pd.DataFrame()

        # Ensure all rows have same length
        min_len = min(len(row) for row in data)
        data = [row[:min_len] for row in data]

        df = pd.DataFrame(data, index=tickers)
        return df.corr()

    @staticmethod
    def identify_clusters(peer_metrics_list: List[PeerMetrics], n_clusters: int = 3) -> Dict[int, List[str]]:
        """Identify natural clusters among peers"""
        if len(peer_metrics_list) < n_clusters:
            n_clusters = len(peer_metrics_list)

        # Prepare data
        data = []
        tickers = []
        for peer in peer_metrics_list:
            values = []
            for metric in ComparisonMetric:
                val = peer.metrics.get(metric, 0)
                values.append(val if val is not None and not np.isnan(val) else 0)
            
            if values:
                data.append(values)
                tickers.append(peer.ticker)

        if not data:
            return {}

        data = np.array(data)
        
        # Normalize
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-10)

        # Cluster
        clusters, _ = ClusteringAnalyzer.simple_kmeans(data, k=n_clusters)

        # Group by cluster
        result = {}
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            result[cluster_id] = [ticker for i, ticker in enumerate(tickers) if mask[i]]

        return result


class SectorAnalyzer:
    """Sector-level analysis and performance metrics"""

    def __init__(self):
        self.sector_peers: Dict[str, List[PeerMetrics]] = {}

    def add_sector_data(self, sector: str, peers: List[PeerMetrics]) -> None:
        """Add peers for a sector"""
        self.sector_peers[sector] = peers

    def calculate_sector_statistics(self, sector: str) -> Dict[str, float]:
        """Calculate sector-wide statistics"""
        if sector not in self.sector_peers:
            return {}

        peers = self.sector_peers[sector]
        stats = {}

        for metric in ComparisonMetric:
            values = [p.metrics.get(metric) for p in peers if p.metrics.get(metric) is not None]
            
            if values:
                stats[f'{metric.value}_mean'] = float(np.mean(values))
                stats[f'{metric.value}_median'] = float(np.median(values))
                stats[f'{metric.value}_std'] = float(np.std(values))

        return stats

    def sector_health_score(self, sector: str) -> float:
        """Calculate overall sector health score (0-100)"""
        if sector not in self.sector_peers:
            return 0.0

        peers = self.sector_peers[sector]
        scores = [p.quality_score for p in peers]

        return float(np.mean(scores)) if scores else 0.0

    def identify_leaders_laggards(self, sector: str, top_n: int = 5) -> Tuple[List[str], List[str]]:
        """Identify top performers and underperformers"""
        if sector not in self.sector_peers:
            return [], []

        peers = self.sector_peers[sector]
        peers_sorted = sorted(peers, key=lambda p: p.quality_score, reverse=True)

        leaders = [p.ticker for p in peers_sorted[:top_n]]
        laggards = [p.ticker for p in peers_sorted[-top_n:]]

        return leaders, laggards
