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
    return pd.Series(values).rolling(n).max()

def MOMENTUM(values, n):
    close_series = pd.Series(values)
    return ((close_series - close_series.shift(n)) / close_series.shift(n)) * 100

# --- STRATEGY LOGIC ---
class QullamaggieBreakout(Strategy):
    stop_loss_pct = 0.95
    momentum_lookback = 64
    momentum_min = 30.0

    def init(self):
        self.ema10 = self.I(EMA, self.data.Close, 10)
        self.ema21 = self.I(EMA, self.data.Close, 21)
        self.sma50 = self.I(SMA, self.data.Close, 50)
        self.orh = self.I(HIGHEST, self.data.High, 3)
        self.mom = self.I(MOMENTUM, self.data.Close, self.momentum_lookback)

    def next(self):
        # Strictly prevent taking trades before 2023
        if self.data.index[-1].year < 2023:
            return

        # 1. Exit Management: Trailing stop on a close below the 21 EMA
        if self.position:
            if self.data.Close[-1] < self.ema21[-1]:
                self.position.close()
            return

        if np.isnan(self.sma50[-1]) or np.isnan(self.mom[-1]):
            return

        # 2. Setup Identification
        trend_aligned = (self.data.Close[-1] > self.ema10[-1] > self.ema21[-1] > self.sma50[-1])
        momentum_good = (self.mom[-1] >= self.momentum_min)

        # 3. Execution via Stop Orders for the next day
        if trend_aligned and momentum_good:
            entry_trigger = self.orh[-1]
            sl_price = entry_trigger * self.stop_loss_pct
            self.buy(stop=entry_trigger, sl=sl_price)

# --- EXECUTION LOOP ---
def run_backtest():
    if not os.path.exists("tickers.txt"):
        print("❌ Please add your 500-stock tickers.txt file to this directory.")
        return

    with open("tickers.txt", "r") as f:
        raw_tickers = [line.strip() for line in f.readlines() if line.strip()]
    
    # Ensure all tickers end with .NS for Yahoo Finance
    tickers = [t if t.endswith(".NS") else f"{t}.NS" for t in raw_tickers]

    results_list = []
    os.makedirs("html_charts", exist_ok=True)

    print(f"⚙️ Starting backtest on {len(tickers)} tickers (2023 to Present)...")

    for t in tickers:
        try:
            # Fetch from late 2022 to pre-load the 64-day momentum math for Jan 1, 2023
            df = yf.download(t, start="2022-09-01", progress=False)
            
            if hasattr(df.columns, 'levels'):
                df.columns = df.columns.get_level_values(0)
            
            df = df.dropna(subset=['Close', 'High', 'Low'])
            if len(df) < 100:
                continue

            bt = Backtest(df, QullamaggieBreakout, cash=100000, commission=0.002, exclusive_orders=True)
            stats = bt.run()
            
            # Only save data and charts for stocks that actually triggered a setup
            if stats['# Trades'] > 0:
                chart_file = f"html_charts/{t.replace('.NS', '')}_2023_backtest.html"
                bt.plot(filename=chart_file, open_browser=False)

                results_list.append({
                    "Ticker": t.replace('.NS', ''),
                    "Return [%]": round(stats['Return [%]'], 2),
                    "Win Rate [%]": round(stats['Win Rate [%]'], 2),
                    "Max Drawdown [%]": round(stats['Max. Drawdown [%]'], 2),
                    "Total Trades": stats['# Trades']
                })
                print(f"✔️ {t} tested ({stats['# Trades']} trades).")

        except Exception as e:
            continue

    if results_list:
        results_df = pd.DataFrame(results_list).sort_values(by="Return [%]", ascending=False)
        results_df.to_csv("backtest_2023_summary.csv", index=False)
        print("\n✅ 2023 Backtest Complete! Top Performers:")
        print(results_df.head(15).to_string(index=False))

if __name__ == "__main__":
    run_backtest()
