import os
import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

# --- CUSTOM INDICATORS ---
def SMA(values, n):
    return pd.Series(values).rolling(n).mean()

def EMA(values, n):
    return pd.Series(values).ewm(span=n, adjust=False).mean()

def HIGHEST(values, n):
    """Returns the highest high of the last n days"""
    return pd.Series(values).rolling(n).max()

def MOMENTUM(values, n):
    """Calculates percentage return over n days"""
    close_series = pd.Series(values)
    return ((close_series - close_series.shift(n)) / close_series.shift(n)) * 100

# --- STRATEGY LOGIC ---
class QullamaggieBreakout(Strategy):
    # Strategy Parameters (optimizable)
    stop_loss_pct = 0.95  # 5% initial stop loss
    momentum_lookback = 64
    momentum_min = 30.0

    def init(self):
        # Calculate moving averages
        self.ema10 = self.I(EMA, self.data.Close, 10)
        self.ema21 = self.I(EMA, self.data.Close, 21)
        self.sma50 = self.I(SMA, self.data.Close, 50)
        
        # 3-Day Consolidation High (ORH)
        self.orh = self.I(HIGHEST, self.data.High, 3)
        
        # 3-Month Momentum
        self.mom = self.I(MOMENTUM, self.data.Close, self.momentum_lookback)

    def next(self):
        # 1. Exit Management: Trailing stop on a close below the 21 EMA
        if self.position:
            if self.data.Close[-1] < self.ema21[-1]:
                self.position.close()
            return

        # Wait for indicators to populate
        if np.isnan(self.sma50[-1]) or np.isnan(self.mom[-1]):
            return

        # 2. Setup Identification (Based on today's closing data)
        trend_aligned = (self.data.Close[-1] > self.ema10[-1] > self.ema21[-1] > self.sma50[-1])
        momentum_good = (self.mom[-1] >= self.momentum_min)

        # 3. Execution (Simulating intraday breakout for tomorrow)
        if trend_aligned and momentum_good:
            # The trigger is the highest high of the last 3 days
            entry_trigger = self.orh[-1]
            sl_price = entry_trigger * self.stop_loss_pct

            # Place a stop order. It only executes tomorrow if price hits the entry_trigger.
            self.buy(stop=entry_trigger, sl=sl_price)

# --- EXECUTION LOOP ---
def run_backtest():
    if not os.path.exists("tickers.txt"):
        tickers = ["MANALIPETC.NS", "DIXON.NS", "ZOMATO.NS"]
    else:
        with open("tickers.txt", "r") as f:
            tickers = [line.strip() + ".NS" for line in f.readlines() if line.strip()]

    results_list = []
    os.makedirs("html_charts", exist_ok=True)

    print(f"⚙️ Starting backtest on {len(tickers)} tickers for the last 5 years...")

    for t in tickers:
        try:
            print(f"Processing {t}...")
            df = yf.download(t, period="5y", interval="1d", progress=False)
            
            if hasattr(df.columns, 'levels'):
                df.columns = df.columns.get_level_values(0)
            
            df = df.dropna(subset=['Close', 'High', 'Low'])
            if len(df) < 100:
                continue

            # Initialize backtester (starting with ₹1,00,000, 100% equity per trade)
            bt = Backtest(df, QullamaggieBreakout, cash=100000, commission=0.002, exclusive_orders=True)
            stats = bt.run()
            
            # Generate interactive HTML chart for this specific stock
            chart_file = f"html_charts/{t.replace('.NS', '')}_backtest.html"
            bt.plot(filename=chart_file, open_browser=False)

            results_list.append({
                "Ticker": t,
                "Return [%]": round(stats['Return [%]'], 2),
                "Win Rate [%]": round(stats['Win Rate [%]'], 2),
                "Max Drawdown [%]": round(stats['Max. Drawdown [%]'], 2),
                "Total Trades": stats['# Trades']
            })

        except Exception as e:
            print(f"Error on {t}: {e}")

    # Compile and save master report
    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values(by="Return [%]", ascending=False)
        results_df.to_csv("backtest_summary.csv", index=False)
        print("\n✅ Backtest Complete!")
        print(results_df.head(10).to_string(index=False))

if __name__ == "__main__":
    run_backtest()
