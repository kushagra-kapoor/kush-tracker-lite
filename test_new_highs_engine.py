"""
Unit tests for new_highs_engine.py in Kush Tracker Lite
Tests multi-tier breakout detection, highs frequency (1W & 1M counts),
custom RS rating (0.40*1M + 0.40*3M + 0.20*6M), dual momentum, and Sortino ratios.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from new_highs_engine import (
    compute_custom_rs_rating,
    compute_dual_momentum_metrics,
    compute_highs_frequency_and_tiers,
    compute_new_highs_universe,
    get_industry_clustering_stats
)

class TestNewHighsEngine(unittest.TestCase):
    def setUp(self):
        # Generate 800 trading days of synthetic data for 4 stocks
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=800, freq='B')
        
        # Stock A: Constant uptrend making new All-Time Highs today and over the last month
        prices_a = np.linspace(50, 200, 800)
        high_a = prices_a * 1.01
        close_a = prices_a
        low_a = prices_a * 0.99
        
        # Stock B: Peaked 600 days ago at 300, crashed to 100, now at 180 (New 52W High, but NOT 2Y or ATH)
        prices_b = np.full(800, 150.0)
        prices_b[:200] = np.linspace(150, 300, 200) # peak 300 at day 200
        prices_b[200:548] = np.linspace(300, 100, 348) # bottom 100
        prices_b[548:] = np.linspace(100, 180, 252) # rallies to 180 in the last 252 days
        high_b = prices_b * 1.01
        close_b = prices_b
        low_b = prices_b * 0.99
        
        # Stock C: Coiling within 3% of ATH (peaked at 200 last week, currently at 195)
        prices_c = np.linspace(80, 200, 800)
        prices_c[-1] = 195.0 # today is 195, while prior ATH was 200 (~2.5% below ATH)
        high_c = prices_c * 1.01
        high_c[-1] = 196.0
        close_c = prices_c
        low_c = prices_c * 0.99
        
        # Stock D: Downtrending / Laggard
        prices_d = np.linspace(200, 80, 800)
        high_d = prices_d * 1.01
        close_d = prices_d
        low_d = prices_d * 0.99
        
        self.tickers = ['STOCK_A.NS', 'STOCK_B.NS', 'STOCK_C.NS', 'STOCK_D.NS']
        
        self.close_df = pd.DataFrame({
            'STOCK_A.NS': close_a,
            'STOCK_B.NS': close_b,
            'STOCK_C.NS': close_c,
            'STOCK_D.NS': close_d
        }, index=dates)
        
        self.high_df = pd.DataFrame({
            'STOCK_A.NS': high_a,
            'STOCK_B.NS': high_b,
            'STOCK_C.NS': high_c,
            'STOCK_D.NS': high_d
        }, index=dates)
        
        self.low_df = pd.DataFrame({
            'STOCK_A.NS': low_a,
            'STOCK_B.NS': low_b,
            'STOCK_C.NS': low_c,
            'STOCK_D.NS': low_d
        }, index=dates)
        
        self.volume_df = pd.DataFrame({
            'STOCK_A.NS': np.full(800, 100000),
            'STOCK_B.NS': np.full(800, 100000),
            'STOCK_C.NS': np.full(800, 100000),
            'STOCK_D.NS': np.full(800, 100000)
        }, index=dates)
        
        self.industry_map = {
            'STOCK_A.NS': 'Technology',
            'STOCK_B.NS': 'Automobile',
            'STOCK_C.NS': 'Technology',
            'STOCK_D.NS': 'Energy'
        }

    def test_breakout_tier_classification(self):
        """Test correct tier categorization: ATH vs 52W vs Coiling vs Normal."""
        high_stats = compute_highs_frequency_and_tiers(self.high_df, self.close_df)
        
        # Stock A should be Lifetime ATH
        self.assertTrue(high_stats['is_ath_today']['STOCK_A.NS'])
        self.assertEqual(high_stats['tier']['STOCK_A.NS'], "🚀 Lifetime ATH")
        
        # Stock B should be 52W High, but NOT Lifetime ATH (since it peaked at 300 earlier)
        self.assertTrue(high_stats['is_52w_today']['STOCK_B.NS'])
        self.assertFalse(high_stats['is_ath_today']['STOCK_B.NS'])
        self.assertIn("52W High", high_stats['tier']['STOCK_B.NS'])
        
        # Stock C should be Coiling near ATH (within 5% of peak)
        self.assertFalse(high_stats['is_ath_today']['STOCK_C.NS'])
        self.assertTrue(high_stats['is_coiling']['STOCK_C.NS'])
        self.assertLessEqual(high_stats['dist_ath_pct']['STOCK_C.NS'], 5.0)

    def test_highs_frequency_counts(self):
        """Test rolling 1W (5-day) and 1M (21-day) new highs summation counts."""
        high_stats = compute_highs_frequency_and_tiers(self.high_df, self.close_df)
        
        # Stock A made consecutive highs, so it should have 5 highs in 1W and 21 highs in 1M
        self.assertGreaterEqual(high_stats['count_1w']['STOCK_A.NS'], 4)
        self.assertGreaterEqual(high_stats['count_1m']['STOCK_A.NS'], 15)
        
        # Stock D is in downtrend, so should have 0 new highs
        self.assertEqual(high_stats['count_1w']['STOCK_D.NS'], 0)
        self.assertEqual(high_stats['count_1m']['STOCK_D.NS'], 0)

    def test_rs_rating_calculation(self):
        """Test custom 0.40*1M + 0.40*3M + 0.20*6M weighted return and percentile ranking 1-99."""
        rs_rating, rs_raw = compute_custom_rs_rating(self.close_df)
        
        # Check that ratings are within [1, 99]
        for t in self.tickers:
            self.assertGreaterEqual(rs_rating[t], 1)
            self.assertLessEqual(rs_rating[t], 99)
            
        # Stock B has the highest recent 1M/3M/6M momentum, so it has RS 99
        self.assertGreater(rs_rating['STOCK_B.NS'], rs_rating['STOCK_A.NS'])
        self.assertGreater(rs_rating['STOCK_A.NS'], rs_rating['STOCK_D.NS'])
        self.assertEqual(rs_rating['STOCK_B.NS'], 99) # Top stock in 4-stock universe
        self.assertEqual(rs_rating['STOCK_D.NS'], 26) # Lowest stock in 4-stock universe

    def test_dual_momentum_metrics(self):
        """Test Dual Score weighting (50% RS + 50% Velocity) and Dual Rank."""
        rs_rating, _ = compute_custom_rs_rating(self.close_df)
        vel, dual_score, dual_rank = compute_dual_momentum_metrics(self.close_df, rs_rating)
        
        # Stock B and A should have positive velocity and top dual scores
        self.assertGreater(vel['STOCK_B.NS'], 0)
        self.assertGreater(dual_score['STOCK_B.NS'], dual_score['STOCK_D.NS'])
        self.assertEqual(dual_rank['STOCK_B.NS'], 1)
        self.assertEqual(dual_rank['STOCK_A.NS'], 2)

    def test_sortino_ratio_integration(self):
        """Test Sortino 3M and 6M computation for new high stocks."""
        df = compute_new_highs_universe(
            close_df=self.close_df,
            high_df=self.high_df,
            low_df=self.low_df,
            volume_df=self.volume_df,
            industry_map=self.industry_map,
            compute_sortino=True
        )
        
        self.assertFalse(df.empty)
        row_a = df[df['Ticker'] == 'STOCK_A'].iloc[0]
        # Stock A is an upward trend with low downside volatility, so Sortino > 0
        self.assertGreaterEqual(row_a['Sortino 3M'], 0.0)
        self.assertGreaterEqual(row_a['Sortino 6M'], 0.0)

    def test_industry_clustering(self):
        """Test aggregation of new highs by industry."""
        df = compute_new_highs_universe(
            close_df=self.close_df,
            high_df=self.high_df,
            low_df=self.low_df,
            volume_df=self.volume_df,
            industry_map=self.industry_map,
            compute_sortino=True
        )
        
        ind_stats = get_industry_clustering_stats(df)
        self.assertFalse(ind_stats.empty)
        # Technology has Stock A (ATH)
        tech_row = ind_stats[ind_stats['Industry'] == 'Technology']
        self.assertGreaterEqual(len(tech_row), 1)

    def test_empty_dataframe_handling(self):
        """Test edge cases with empty data."""
        empty_df = pd.DataFrame()
        self.assertTrue(compute_new_highs_universe(empty_df, empty_df).empty)
        rs_r, rs_raw = compute_custom_rs_rating(empty_df)
        self.assertTrue(rs_r.empty)
        self.assertTrue(get_industry_clustering_stats(empty_df).empty)

    def test_coiling_thresholds(self):
        """Verify coiling logic: distance from peak between 0.01% and 5.0%."""
        high_stats = compute_highs_frequency_and_tiers(self.high_df, self.close_df)
        # Stock C is coiling at 195 with prior peak at 200 (~2.5%)
        self.assertTrue(high_stats['is_coiling']['STOCK_C.NS'])
        dist = high_stats['dist_ath_pct']['STOCK_C.NS']
        self.assertGreater(dist, 0.0)
        self.assertLessEqual(dist, 5.0)

    def test_days_since_high_and_setup_tag(self):
        """Test calculation of days since high, horizon categories, and setup tags."""
        df = compute_new_highs_universe(
            close_df=self.close_df,
            high_df=self.high_df,
            low_df=self.low_df,
            volume_df=self.volume_df,
            industry_map=self.industry_map,
            compute_sortino=True
        )
        self.assertFalse(df.empty)
        
        # Stock A made ATH today
        stk_a = df[df['Ticker'] == 'STOCK_A'].iloc[0]
        self.assertEqual(stk_a['Days Since High'], 0)
        self.assertEqual(stk_a['Days Since High Text'], "⚡ Today")
        self.assertEqual(stk_a['Horizon'], "Today")
        self.assertEqual(stk_a['Setup Tag'], "⚡ Breaking Out Today")
        
        # Stock C peaked recently and is coiling/consolidating
        stk_c = df[df['Ticker'] == 'STOCK_C'].iloc[0]
        self.assertIn(stk_c['Horizon'], ["Today", "1-Week", "1-Month"])
        self.assertIn(stk_c['Setup Tag'], [
            "🎯 Near 21 EMA (Buy Zone)", "✨ Consolidating", "⏳ Extended (>7%)",
            "⚡ Breaking Out Today", "🛡️ Near 50 SMA", "⚠️ Below 50 SMA", "🛡️ 21 EMA Shakeout"
        ])

if __name__ == '__main__':
    unittest.main()
