# ETF Portfolio Signals

Daily, non-leveraged top-100 US-listed ETF screener.

- Universe sourced from [ETFdb top 100 by AUM](https://etfdb.com/compare/market-cap/)
- Leveraged/inverse ETFs are filtered out
- Each ETF is scored on risk-adjusted momentum, trend (distance from 20d/50d MAs), RSI and volatility
- Output is four-bucket directional label: `Major Up`, `Mildly Up`, `Mildly Down`, `Major Down`

## Run

```bash
cd etf_signals
python3 -m pip install -r requirements.txt
python3 generate_signals.py
```

Outputs:
- `output/latest_signals.json`
- `output/latest_signals.html`

These are directional probability-style signals, not explicit buy/sell recommendations.
