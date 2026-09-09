import os
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress intra-candle warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
        if self.data.index[-1].year < 2023:
            return

        # Trailing stop on 21 EMA close
        if self.position:
            if self.data.Close[-1] < self.ema21[-1]:
                self.position.close()
            return

        if np.isnan(self.sma50[-1]) or np.isnan(self.mom[-1]):
            return

        trend_aligned = (self.data.Close[-1] > self.ema10[-1] > self.ema21[-1] > self.sma50[-1])
        momentum_good = (self.mom[-1] >= self.momentum_min)

        if trend_aligned and momentum_good:
            entry_trigger = self.orh[-1]
            sl_price = entry_trigger * self.stop_loss_pct
            self.buy(stop=entry_trigger, sl=sl_price)

# --- WORKER THREAD ---
def process_ticker(t):
    try:
        df = yf.download(t, start="2022-09-01", progress=False)
        
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna(subset=['Close', 'High', 'Low'])
        if len(df) < 100:
            return None

        bt = Backtest(df, QullamaggieBreakout, cash=100000, commission=0.002, exclusive_orders=True)
        stats = bt.run()
        
        if stats['# Trades'] > 0:
            return {
                "Ticker": t.replace('.NS', ''),
                "Return [%]": round(stats['Return [%]'], 2),
                "Win Rate [%]": round(stats['Win Rate [%]'], 2),
                "Max Drawdown [%]": round(stats['Max. Drawdown [%]'], 2),
                "Total Trades": stats['# Trades'],
                "_bt_obj": bt
            }
        return None
    except Exception:
        return None

def generate_dashboard(results):
    rows_html = ""
    for r in results:
        ticker = r["Ticker"]
        has_chart = r.get("has_chart", False)
        link = f"<a href='{ticker}_2023_backtest.html' target='_blank'>📈 View Chart</a>" if has_chart else "<span style='color:#555;'>No Chart</span>"
        color = "#3fb950" if r["Return [%]"] > 0 else "#f85149"

        rows_html += f"""
        <tr>
            <td><strong>{ticker}</strong></td>
            <td style='color: {color}; font-weight: bold;'>{r['Return [%]']:+.2f}%</td>
            <td>{r['Win Rate [%]']:.1f}%</td>
            <td style='color: #f85149;'>{r['Max Drawdown [%]']:.2f}%</td>
            <td>{r['Total Trades']}</td>
            <td>{link}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Qullamaggie Backtest Results (2023 - Present)</title>
    <style>
        body {{ background-color: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 24px; }}
        .container {{ max-width: 1000px; margin: auto; }}
        h1 {{ color: #58a6ff; font-size: 1.5rem; }}
        p {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #30363d; font-size: 0.9rem; }}
        th {{ background: #21262d; color: #f0f6fc; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.5px; }}
        tr:hover {{ background: #1f242c; }}
        a {{ color: #58a6ff; text-decoration: none; font-weight: 600; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Qullamaggie Backtest Dashboard</h1>
        <p>Testing period: 2023 to Present | Sorted by Highest Return</p>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Total Return</th>
                    <th>Win Rate</th>
                    <th>Max Drawdown</th>
                    <th>Trades</th>
                    <th>Interactive Plot</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
    """
    with open("html_charts/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

# --- EXECUTION LOOP ---
def run_backtest():
    if not os.path.exists("tickers.txt"):
        print("❌ tickers.txt missing.")
        return

    with open("tickers.txt", "r") as f:
        raw_tickers = [line.strip() for line in f.readlines() if line.strip()]
    
    tickers = [t if t.endswith(".NS") else f"{t}.NS" for t in raw_tickers]

    results_list = []
    os.makedirs("html_charts", exist_ok=True)

    print(f"⚡ Running backtest on {len(tickers)} tickers...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results_list.append(res)
                print(f"✔️ {res['Ticker']} processed ({res['Total Trades']} trades).")

    if results_list:
        results_list = sorted(results_list, key=lambda x: x["Return [%]"], reverse=True)
        
        # Plot interactive charts for top 15 performers
        for i, res in enumerate(results_list):
            bt = res.pop("_bt_obj", None) 
            if i < 15 and bt:
                chart_file = f"html_charts/{res['Ticker']}_2023_backtest.html"
                try:
                    bt.plot(filename=chart_file, open_browser=False)
                    res["has_chart"] = True
                except Exception:
                    res["has_chart"] = False
            else:
                res["has_chart"] = False

        # Build dashboard and write CSV
        generate_dashboard(results_list)
        
        results_df = pd.DataFrame(results_list).drop(columns=["has_chart"], errors="ignore")
        results_df.to_csv("backtest_2023_summary.csv", index=False)
        print("\n✅ Backtest & Dashboard compilation complete!")

if __name__ == "__main__":
    run_backtest()
