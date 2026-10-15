"""
Comprehensive Verification Suite for Kush Tracker Lite Super-Trader Migration.
Validates:
1. Jim Roppel 21 EMA / 50 SMA Support Bounce Scanner
2. Jesse Livermore Sister Stock Tandem Confirmation Engine
3. Steve Cohen & Shake Pryzby Beach Ball Divergence & Timing Context
4. Dr. Martin Zweig Breadth Thrust (ZBT) & 9:1 Up-Volume Engine
5. Progressive Exposure Plan Zweig Super-Thrust and Early Thrust rules
6. 20 EMA Recapture & 7-Step Recovery Sequence
"""
import sys
import os
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure app root is in sys.path
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from intraday_ui import (
    detect_roppel_support_bounce,
    compute_sister_stock_tandems,
    detect_beach_ball_divergence,
    get_beach_ball_timing_state
)
from breadth_engine import compute_zweig_thrust_and_volume_signals
from macro_regime_engine import (
    get_progressive_exposure_plan,
    detect_20ema_recapture_and_sequence
)


def test_roppel_support_bounce():
    print("Testing Jim Roppel Support Bounce Scanner...")
    
    # 1. Valid 21 EMA Bounce: Price > 50 SMA > 200 SMA, low touches within -0.5% of 21 EMA, quiet volume, strong close
    res_21 = detect_roppel_support_bounce(
        current_price=105.0,
        today_low=99.8,
        sma_21=100.0,
        sma_50=92.0,
        sma_200=80.0,
        vol_expansion=0.85,
        close_range_pct=75.0,
        today_pct=1.5
    )
    assert res_21['is_bounce'] is True, f"Expected 21 EMA bounce True, got {res_21}"
    assert res_21['bounce_type'] == '21 EMA', f"Expected '21 EMA', got {res_21['bounce_type']}"
    assert res_21['bounce_dist_pct'] == -0.2, f"Expected -0.2%, got {res_21['bounce_dist_pct']}"

    # 2. Valid 50 SMA Bounce (Institutional Line in the Sand)
    res_50 = detect_roppel_support_bounce(
        current_price=93.0,
        today_low=89.5,
        sma_21=96.0,
        sma_50=90.0,
        sma_200=75.0,
        vol_expansion=0.95,
        close_range_pct=60.0,
        today_pct=0.5
    )
    assert res_50['is_bounce'] is True, f"Expected 50 SMA bounce True, got {res_50}"
    assert res_50['bounce_type'] == '50 SMA', f"Expected '50 SMA', got {res_50['bounce_type']}"

    # 3. Failed: Volume too heavy (institutional distribution dump, not quiet test)
    res_heavy_vol = detect_roppel_support_bounce(
        current_price=101.0,
        today_low=100.0,
        sma_21=100.0,
        sma_50=92.0,
        sma_200=80.0,
        vol_expansion=2.5,  # Heavy dump
        close_range_pct=70.0,
        today_pct=0.2
    )
    assert res_heavy_vol['is_bounce'] is False, "Expected False on heavy volume surge"

    # 4. Failed: Not in Stage 2 (below 200 SMA)
    res_bear = detect_roppel_support_bounce(
        current_price=65.0,
        today_low=64.8,
        sma_21=65.0,
        sma_50=70.0,
        sma_200=85.0,
        vol_expansion=0.6,
        close_range_pct=70.0,
        today_pct=0.5
    )
    assert res_bear['is_bounce'] is False, "Expected False in secular downtrend below 200 SMA"

    print("  ✅ All Jim Roppel Support Bounce tests passed!")


def test_sister_stock_tandems():
    print("Testing Jesse Livermore Sister Stock Tandem Engine...")
    
    # Create sample universe with Defense Industry (Tandem) and Metals (Divergence)
    data = [
        # Defense: HAL and BEL both strong
        {'Ticker': 'HAL👑', 'Industry': 'Defense', 'RS Rating': 96, 'Today %': 3.2, 'Dist from 52W High %': 2.1},
        {'Ticker': 'BEL', 'Industry': 'Defense', 'RS Rating': 91, 'Today %': 1.8, 'Dist from 52W High %': 3.4},
        {'Ticker': 'BDL', 'Industry': 'Defense', 'RS Rating': 84, 'Today %': 2.0, 'Dist from 52W High %': 5.0},
        
        # Metals: TATASTEEL up but JSWSTEEL dumping
        {'Ticker': 'TATASTEEL', 'Industry': 'Metals', 'RS Rating': 88, 'Today %': 2.5, 'Dist from 52W High %': 4.0},
        {'Ticker': 'JSWSTEEL', 'Industry': 'Metals', 'RS Rating': 82, 'Today %': -3.8, 'Dist from 52W High %': 28.0}, # Breaking down
        
        # Solitary: Only 1 stock in Space tech
        {'Ticker': 'DATA_PATTERNS', 'Industry': 'Space', 'RS Rating': 90, 'Today %': 4.0, 'Dist from 52W High %': 1.0}
    ]
    df = pd.DataFrame(data)
    df_tandem = compute_sister_stock_tandems(df)

    # Verify HAL (Confirmed sister BEL)
    hal_tag = df_tandem[df_tandem['Ticker'] == 'HAL👑']['Sister Stock Tandem'].iloc[0]
    assert '👯 Confirmed' in hal_tag, f"Expected HAL confirmed, got {hal_tag}"
    assert 'BEL' in hal_tag, f"Expected BEL mentioned in HAL tag, got {hal_tag}"

    # Verify BDL (Group Confirmed)
    bdl_tag = df_tandem[df_tandem['Ticker'] == 'BDL']['Sister Stock Tandem'].iloc[0]
    assert '🛡️ Group Confirmed' in bdl_tag, f"Expected BDL group confirmed, got {bdl_tag}"

    # Verify TATASTEEL (Divergence due to JSWSTEEL dumping)
    tata_tag = df_tandem[df_tandem['Ticker'] == 'TATASTEEL']['Sister Stock Tandem'].iloc[0]
    assert '⚠️ Divergence' in tata_tag, f"Expected TATASTEEL divergence warning, got {tata_tag}"

    # Verify DATA_PATTERNS (Solitary Leader)
    dp_tag = df_tandem[df_tandem['Ticker'] == 'DATA_PATTERNS']['Sister Stock Tandem'].iloc[0]
    assert '— Solitary Leader' in dp_tag, f"Expected solitary leader, got {dp_tag}"

    print("  ✅ All Jesse Livermore Sister Stock Tandem tests passed!")


def test_beach_ball_divergence():
    print("Testing Steve Cohen Beach Ball Divergence & Timing...")
    
    # Sample resilient leader holding 50 SMA near 52W high
    df = pd.DataFrame([{
        'Ticker': 'COFORGE👑',
        'Dist from 50 SMA %': 4.5,
        'Dist from 10 EMA %': 0.8,
        'Today %': 1.2,
        '1M Return': 14.5,
        '12D Trend Score': 9,
        'Dist from 52W High %': 3.2,
        'Volume Expansion': 1.1,
        'RS Rating': 94
    }])
    
    market_state_correction = {
        'green_light': False,
        'index_pct': -1.2,
        'recapture_sequence': {
            'secular_bull': True,
            'current_step_num': 4,
            'recapture_active': False
        }
    }
    
    df_bb = detect_beach_ball_divergence(df, index_state=market_state_correction, is_us=False)
    assert df_bb['Is Beach Ball'].iloc[0] is True or df_bb['Is Beach Ball'].iloc[0] == 1, "Expected Is Beach Ball True"
    assert df_bb['Beach Ball Score'].iloc[0] >= 70, f"Expected BB Score >= 70, got {df_bb['Beach Ball Score'].iloc[0]}"

    timing = get_beach_ball_timing_state(market_state_correction, is_us=False)
    assert timing['state'] == 'PRIME_STALK', f"Expected PRIME_STALK in correction, got {timing['state']}"
    assert 'PRIME' in timing['badge'], f"Expected PRIME badge, got {timing['badge']}"

    # Test ignition timing when 20 EMA is recaptured
    market_state_ignition = {
        'green_light': True,
        'index_pct': 1.5,
        'recapture_sequence': {
            'secular_bull': True,
            'current_step_num': 5,
            'recapture_active': True
        }
    }
    timing_ign = get_beach_ball_timing_state(market_state_ignition, is_us=False)
    assert timing_ign['state'] == 'IGNITION', f"Expected IGNITION, got {timing_ign['state']}"
    assert 'BUY IGNITION' in timing_ign['badge'], f"Expected BUY IGNITION, got {timing_ign['badge']}"

    print("  ✅ All Steve Cohen Beach Ball Divergence & Timing tests passed!")


def test_zweig_breadth_thrust_and_volume():
    print("Testing Dr. Martin Zweig Breadth Thrust & 9:1 Up-Volume Signals...")
    
    # Generate 30 days of synthetic advances/declines & up/down volume
    dates = pd.date_range(end='2026-10-15', periods=30, freq='B')
    
    # Create an oversold condition (first 10 days adv=100, dec=600 -> ratio ~ 0.14)
    # Followed by a massive thrust (next 10 days adv=650, dec=50 -> ratio ~ 0.92)
    adv = [120]*10 + [650]*10 + [400]*10
    dec = [580]*10 + [50]*10 + [300]*10
    
    # Set day 12 and day 20 as 9:1 up volume days (up_vol >= 90% of total)
    up_vol = [200.0]*11 + [1800.0] + [200.0]*7 + [1900.0] + [300.0]*10
    down_vol = [800.0]*11 + [100.0] + [800.0]*7 + [100.0] + [300.0]*10
    
    df = pd.DataFrame({
        'advances': adv,
        'declines': dec,
        'up_volume': up_vol,
        'down_volume': down_vol
    }, index=dates)
    
    df_calc = compute_zweig_thrust_and_volume_signals(df)
    
    # 1. Check ZBT 10-EMA calculation exists
    assert 'zbt_10_ema' in df_calc.columns, "Expected zbt_10_ema column"
    assert df_calc['zbt_10_ema'].iloc[0] <= 0.40, "Initial EMA should be low/oversold"
    assert df_calc['zbt_10_ema'].iloc[19] >= 0.615, "Thrust EMA should exceed 0.615"
    
    # 2. Check ZBT signal fired
    assert (df_calc['zbt_signal'] == 1).any(), "ZBT signal should have fired"
    assert df_calc['zbt_active'].iloc[-1] == 1, "ZBT active state should remain active"
    
    # 3. Check 9:1 up volume days
    assert df_calc['is_9_to_1_up_day'].sum() >= 2, f"Expected >= 2 9:1 days, got {df_calc['is_9_to_1_up_day'].sum()}"
    
    # 4. Check Double 9:1 Blast-Off
    assert df_calc['double_9_to_1_active'].iloc[-1] == 1, "Double 9:1 Blast-Off should be active"

    print("  ✅ All Dr. Martin Zweig Quantitative Breadth & Volume tests passed!")


def test_progressive_exposure_plan():
    print("Testing Progressive Exposure Plan with Zweig Tape integration...")
    
    # Scenario A: Double 9:1 Blast-Off Active in Confirmed Uptrend
    zweig_tape_blastoff = {'double_9_to_1_active': True, 'zbt_active': True}
    exp_blastoff = get_progressive_exposure_plan(
        regime_label="Confirmed Uptrend",
        dd_count=1,
        dd_cluster=False,
        ftd_detected=True,
        leadership_grade='A',
        zweig_tape=zweig_tape_blastoff
    )
    assert 'SUPER-THRUST' in exp_blastoff['deck_state'], f"Expected Super-Thrust, got {exp_blastoff['deck_state']}"
    assert '100%' in exp_blastoff['exposure_range'], f"Expected 100% allocation, got {exp_blastoff['exposure_range']}"

    # Scenario B: ZBT Fired during Market in Correction (Oversold blast-off test)
    zweig_tape_zbt_correction = {'double_9_to_1_active': False, 'zbt_active': True}
    exp_correction_zbt = get_progressive_exposure_plan(
        regime_label="Market in Correction",
        dd_count=7,
        dd_cluster=True,
        ftd_detected=False,
        leadership_grade='D',
        zweig_tape=zweig_tape_zbt_correction
    )
    assert 'EARLY THRUST' in exp_correction_zbt['deck_state'], f"Expected Early Thrust, got {exp_correction_zbt['deck_state']}"
    assert '20% – 30%' in exp_correction_zbt['exposure_range'], f"Expected 20-30%, got {exp_correction_zbt['exposure_range']}"

    # Scenario C: Standard Market in Correction without ZBT
    exp_pure_correction = get_progressive_exposure_plan(
        regime_label="Market in Correction",
        dd_count=6,
        dd_cluster=False,
        zweig_tape={'double_9_to_1_active': False, 'zbt_active': False}
    )
    assert 'FROZEN DECK' in exp_pure_correction['deck_state'], f"Expected Frozen Deck, got {exp_pure_correction['deck_state']}"
    assert '0% – 10%' in exp_pure_correction['exposure_range'], f"Expected 0-10%, got {exp_pure_correction['exposure_range']}"

    print("  ✅ All Progressive Exposure Plan tests passed!")


def test_20ema_recapture_sequence():
    print("Testing 20 EMA Recapture & 7-Step Recovery Sequence...")
    
    # Generate 50 sessions of synthetic price action testing 20 EMA
    dates = pd.date_range(end='2026-10-15', periods=50, freq='B')
    prices = np.linspace(100, 120, 50)
    df = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': [1000000] * 50
    }, index=dates)
    
    res = detect_20ema_recapture_and_sequence(df)
    assert 'secular_bull' in res, "Missing secular_bull"
    assert 'current_step_num' in res, "Missing current_step_num"
    assert 1 <= res['current_step_num'] <= 7, f"Invalid step {res['current_step_num']}"
    assert 'sequence_summary' in res, "Missing sequence_summary"
    
    print("  ✅ All 20 EMA Recapture & 7-Step Sequence tests passed!")


if __name__ == '__main__':
    print("==================================================================")
    print("⚡ RUNNING KUSH TRACKER LITE SUPER-TRADER MIGRATION TEST SUITE ⚡")
    print("==================================================================")
    test_roppel_support_bounce()
    test_sister_stock_tandems()
    test_beach_ball_divergence()
    test_zweig_breadth_thrust_and_volume()
    test_progressive_exposure_plan()
    test_20ema_recapture_sequence()
    print("==================================================================")
    print("🎉 ALL 6 SUPER-TRADER LITE TEST SUITES PASSED CLEANLY! (100% GREEN)")
    print("==================================================================")
