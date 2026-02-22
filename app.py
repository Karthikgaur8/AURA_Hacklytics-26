import yfinance as yf
import torch
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import gradio as gr
from chronos import BaseChronosPipeline
from google import genai
from google.genai import types

# 1. Initialize APIs
client = genai.Client(api_key="INSERT_YOUR_API_KEY") 

print("Loading Chronos-Bolt into VRAM...")
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cuda", 
    dtype=torch.bfloat16,
)

def compute_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculates Delta and Vega via closed-form Black-Scholes."""
    if T <= 0 or sigma <= 0: return 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100 
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    return delta, vega


# --- TAB 1: BATCHED SCREENER ---
# --- TAB 1: BATCHED SCREENER ---
def run_institutional_screener():
    # Dynamically load the S&P 100 universe
    with open('sp100.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
        
    print(f"Batch downloading {len(tickers)} assets...")
    df = yf.download(tickers, period="6mo", group_by="ticker", progress=False)
    
    # DYNAMIC RISK-FREE RATE: Pull 13-week Treasury Yield (^IRX)
    try:
        rf_data = yf.download("^IRX", period="5d", progress=False)
        r = rf_data['Close'].dropna().iloc[-1].item() / 100.0
    except Exception:
        r = 0.043 # Fallback if treasury data fails
        
    valid_tickers = []
    price_list = []
    
    for t in tickers:
        try:
            prices = df[t]['Close'].dropna().values
            if len(prices) >= 60:
                price_list.append(prices[-60:])
                valid_tickers.append(t)
        except Exception:
            continue
            
    # Batched H200 Inference
    context = torch.tensor(np.array(price_list), dtype=torch.float32).cuda()
    quantiles, _ = pipeline.predict_quantiles(
        context, prediction_length=15, quantile_levels=[0.1, 0.5, 0.9]
    )
    
    results = []
    
    for i, ticker in enumerate(valid_tickers):
        try:
            S = price_list[i][-1]
            low_80 = quantiles[i, :, 0].cpu().numpy()
            high_80 = quantiles[i, :, 2].cpu().numpy()
            
            # Log Return Volatility Extraction
            log_q90 = np.log(high_80[-1] / S)
            log_q10 = np.log(low_80[-1] / S)
            predicted_vol = ((log_q90 - log_q10) / 2.56) * np.sqrt(252/15) 
            
            stock = yf.Ticker(ticker)
            expirations = stock.options
            if not expirations: continue
            
            # 15-day Expiry Alignment
            target_dte = 21 
            closest_expiry = None
            min_diff = 999
            for exp in expirations:
                dte = (pd.to_datetime(exp) - pd.Timestamp.today()).days
                if dte > 7 and abs(dte - target_dte) < min_diff:
                    min_diff = abs(dte - target_dte)
                    closest_expiry = exp
                    
            if not closest_expiry: continue
            T = max((pd.to_datetime(closest_expiry) - pd.Timestamp.today()).days / 365.0, 0.001)
            
            opt_chain = stock.option_chain(closest_expiry)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # STRADDLE MID-IV FIX: Pull both ATM Call and ATM Put
            atm_call = calls.iloc[(calls['strike'] - S).abs().argsort()[:1]].iloc[0]
            atm_put = puts.iloc[(puts['strike'] - S).abs().argsort()[:1]].iloc[0]
            
            # LIQUIDITY FILTER: Ensure adequate Open Interest
            call_oi = atm_call.get('openInterest', 0)
            if pd.isna(call_oi) or call_oi < 500:
                continue

            # Average the IV for the true straddle pricing
            market_iv = (atm_call['impliedVolatility'] + atm_put['impliedVolatility']) / 2.0
            
            # BID-ASK SPREAD AWARENESS (Dynamic penalty)
            call_ask = atm_call.get('ask', 0.01)
            call_bid = atm_call.get('bid', 0.01)
            bid_ask_pct = (call_ask - call_bid) / (call_ask + 0.001) # Avoid division by zero
            dynamic_cost_penalty = max(0.02, bid_ask_pct) # Base 2% or wider if illiquid

            delta, vega = compute_greeks(S, atm_call['strike'], T, r, market_iv, 'call')
            
            raw_spread = predicted_vol - market_iv
            adjusted_spread = raw_spread - dynamic_cost_penalty if raw_spread > 0 else raw_spread + dynamic_cost_penalty
            
            # VEGA-WEIGHTED EDGE
            vega_edge = adjusted_spread * vega

            results.append({
                "Ticker": ticker,
                "Expiry": closest_expiry,
                "Chronos_Vol": round(predicted_vol * 100, 2),
                "Straddle_IV": round(market_iv * 100, 2),
                "Adj_Spread": round(adjusted_spread * 100, 2),
                "Vega": round(vega, 3),
                "Vega_Edge": round(vega_edge, 3),
                "OI": int(call_oi),
                "Delta": round(delta, 3)
            })
        except Exception:
            continue
            
    results.sort(key=lambda x: abs(x["Vega_Edge"]), reverse=True)
    df_results = pd.DataFrame(results).head(15) 
    
    prompt = f"Act as an Institutional Portfolio Manager. Analyze this Vega-weighted variance data: {df_results.to_string()}\nSelect the single most optimal asset based on the highest absolute Vega_Edge, ensuring adequate liquidity. Map its data strictly to the JSON schema."
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "Target_Asset": {"type": "STRING"},
            "Target_Expiry": {"type": "STRING"},
            "Action": {"type": "STRING"},
            "Vega_Edge": {"type": "NUMBER"},
            "Delta_Hedge_Ratio": {"type": "NUMBER"},
            "require_human_execution_confirmation": {"type": "BOOLEAN"}
        },
        "required": ["Target_Asset", "Target_Expiry", "Action", "Vega_Edge", "Delta_Hedge_Ratio", "require_human_execution_confirmation"]
    }
    
    try:
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=response_schema, temperature=0.0)
        )
        trade_ticket = response.text
    except Exception as e:
        trade_ticket = f'{{"Error": "{str(e)}"}}'
        
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_results['Ticker'], df_results['Vega_Edge'], color=['tab:purple' if s > 0 else 'tab:orange' for s in df_results['Vega_Edge']])
    ax.set_title("Vega-Weighted Straddle Edge (Top 15 Liquid Chains)", fontweight="bold")
    ax.set_ylabel("Expected Edge (Spread × Vega)")
    ax.axhline(0, color='black', linewidth=1.2)
    plt.tight_layout()
    
    return fig, df_results, trade_ticket

# --- TAB 2: DEEP DIVE CONE ---
def generate_vol_cone(ticker):
    try:
        stock_data = yf.download(ticker, period="6mo", progress=False)
        close_prices = stock_data['Close'].values.flatten()
        context = torch.tensor(close_prices).unsqueeze(0).cuda()
        
        quantiles, _ = pipeline.predict_quantiles(context, prediction_length=15, quantile_levels=[0.1, 0.5, 0.9])
        low_80 = quantiles[0, :, 0].cpu().numpy()
        median = quantiles[0, :, 1].cpu().numpy()
        high_80 = quantiles[0, :, 2].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        time_hist = np.arange(len(close_prices[-45:]))
        time_pred = np.arange(len(time_hist), len(time_hist) + 15)
        
        ax.plot(time_hist, close_prices[-45:], color="black", label="Historical (T-45 to T-0)")
        ax.plot(time_pred, median, color="tab:blue", linestyle="--", label="Chronos Median Forecast")
        ax.fill_between(time_pred, low_80, high_80, color="red", alpha=0.15, label="80% Predictive Volatility Cone")
        
        ax.set_title(f"Zero-Shot Variance Distribution: {ticker.upper()}", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception:
        return None

# --- TAB 3: CALIBRATION BACKTEST ---
def run_calibration_backtest(ticker):
    try:
        stock_data = yf.download(ticker, period="6mo", progress=False)
        close_prices = stock_data['Close'].values.flatten()
        
        context_data = close_prices[:-15]
        actual_holdout = close_prices[-15:]
        
        context = torch.tensor(context_data).unsqueeze(0).cuda()
        quantiles, _ = pipeline.predict_quantiles(context, prediction_length=15, quantile_levels=[0.1, 0.5, 0.9])
        
        low_80 = quantiles[0, :, 0].cpu().numpy()
        median = quantiles[0, :, 1].cpu().numpy()
        high_80 = quantiles[0, :, 2].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        time_hist = np.arange(len(context_data[-45:]))
        time_pred = np.arange(len(time_hist), len(time_hist) + 15)
        
        ax.plot(time_hist, context_data[-45:], color="black", label="Historical Context (T-60 to T-15)")
        ax.plot(time_pred, actual_holdout, color="red", marker="o", linewidth=2, label="Actual Realized Price (T-15 to T-0)")
        ax.plot(time_pred, median, color="tab:blue", linestyle="--", label="Chronos T-15 Median Forecast")
        ax.fill_between(time_pred, low_80, high_80, color="tab:blue", alpha=0.2, label="Chronos T-15 80% CI")
        
        ax.set_title(f"Walk-Forward Out-of-Sample Calibration: {ticker.upper()}", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception:
        return None

# --- UI LAYOUT ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# ⚡ AURA: Institutional Variance Engine")
    
    with gr.Tab("Market Screener (Batched GPU Inference)"):
        gr.Markdown("Batched zero-shot inference across 100 mega-cap equities to identify transaction-cost-adjusted variance mispricing.")
        submit_btn = gr.Button("Initialize Batched H200 Market Scan", variant="primary")
        with gr.Row():
            plot_output = gr.Plot(scale=2)
        with gr.Row():
            raw_data_output = gr.Dataframe(label="L1 Risk Metrics & Greeks", scale=2)
            thesis_output = gr.Code(label="Gemini 3.1 Pro: JSON Algorithmic Trade Ticket", language="json", scale=1)
        submit_btn.click(fn=run_institutional_screener, inputs=[], outputs=[plot_output, raw_data_output, thesis_output])
        
    with gr.Tab("Asset Deep Dive"):
        gr.Markdown("Generate an isolated 15-day predictive volatility cone for a specific target asset.")
        with gr.Row():
            cone_ticker = gr.Textbox(label="Enter Ticker (e.g., TSLA)")
            cone_btn = gr.Button("Generate Cone")
        cone_plot = gr.Plot()
        cone_btn.click(fn=generate_vol_cone, inputs=cone_ticker, outputs=cone_plot)
        
    with gr.Tab("Model Calibration (Backtest)"):
        gr.Markdown("Walk-forward backtest isolating the last 15 days of trading to prove out-of-sample forecast calibration.")
        with gr.Row():
            calib_ticker = gr.Textbox(label="Enter Ticker for Backtest (e.g., AAPL)")
            calib_btn = gr.Button("Run Calibration Matrix")
        calib_plot = gr.Plot()
        calib_btn.click(fn=run_calibration_backtest, inputs=calib_ticker, outputs=calib_plot)

if __name__ == "__main__":
    demo.launch(share=True)