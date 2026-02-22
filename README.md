# ⚡ AURA: Institutional Variance Engine

**AURA** is an institutional-grade variance mispricing engine that shifts the focus from fragile directional price prediction to the mathematical extraction of volatility alpha. 

By utilizing **Amazon’s Chronos-Bolt** zero-shot foundation model on an **NVIDIA H200** cluster, the system executes batched tensor inference across the S&P 100 to identify spreads between forecasted variance and live market Implied Volatility (IV).

## 🚀 Core Features
- **Batched GPU Inference:** Stacks 100+ tickers into a single PyTorch tensor for simultaneous parallel forecasting.
- **Vega-Weighted Alpha:** Ranks opportunities by actual dollar-payout potential ($Edge = Spread_{adj} \times \nu$) rather than raw percentage spreads.
- **Microstructure Realism:** Includes dynamic risk-free rate scaling, 21-day expiry alignment, liquidity gating (OI > 500), and bid-ask spread penalties.
- **Agentic Routing:** Uses **Gemini 3.1 Pro** in a deterministic, zero-temperature mode to generate machine-readable JSON trade tickets.

## 📊 Mathematical Foundation
We extract annualized volatility ($\sigma$) from the foundation model's 10th and 90th percentile quantiles using a log-return standard deviation proxy:

$$\sigma_{predicted} = \sqrt{\frac{252}{15}} \times \frac{\ln(q_{90} / q_{10})}{2.56}$$

Greeks are calculated via a closed-form Black-Scholes implementation:

$$d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}$$
$$\nu = S \cdot \phi(d_1) \cdot \sqrt{T}$$

## 🛠️ Tech Stack
- **Model:** `amazon/chronos-bolt-base`
- **Agent:** `google/gemini-3.1-pro-preview`
- **Compute:** NVIDIA H200 (141GB VRAM)
- **Frontend:** Gradio (Institutional Monochrome)

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/AURA.git](https://github.com/yourusername/AURA.git)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the engine:
   ```bash
   python app.py
   ```
## ⚖️ Disclaimer
This project is an educational proxy for variance trading. Options Greeks are approximated via Black-Scholes (European-style). 
Real-world U.S. equities use American-style options. Use for live trading at your own risk.
