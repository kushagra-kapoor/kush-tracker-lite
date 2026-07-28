import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

pages = [
    "pages.intraday_monitor",
    "pages.intraday_monitor_us",
    "pages.true_market_leader",
    "pages.true_market_leader_us",
    "pages.market_regime",
    "pages.focus_list"
]

def run_audit():
    errors = 0
    for page in pages:
        try:
            print(f"Auditing {page}...")
            __import__(page)
            print(f"✅ {page} imported successfully!")
        except Exception as e:
            print(f"❌ Error in {page}:")
            traceback.print_exc()
            errors += 1
    
    if errors == 0:
        print("\n🎉 Full audit passed! All pages load without top-level errors.")
    else:
        print(f"\n⚠️ Audit failed with {errors} errors.")

if __name__ == "__main__":
    run_audit()
