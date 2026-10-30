"""
FX pricing margin analysis.

Question: when a money-transfer provider quotes a customer a USD -> foreign
currency rate, how large a markup over the mid-market rate does it need so
that it still earns a target margin after the market moves against it during
the settlement window?

Data: daily ECB reference rates via the Frankfurter API (free, no API key).

Usage:
    python fx_pricing_analysis.py --start 2023-01-01 --end 2025-10-01 \
        --currencies EUR GBP CAD MXN INR --target-margin 0.008 --competitor-markup 0.015
    python fx_pricing_analysis.py --demo      # synthetic data, for testing only
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

API_URLS = [
    "https://api.frankfurter.dev/v1/{start}..{end}?base=USD&symbols={symbols}",
    "https://api.frankfurter.app/{start}..{end}?from=USD&to={symbols}",
]
TRADING_DAYS = 252
COLORS = ["#2a6fdb", "#e0703a", "#2a9d78", "#c2477d", "#7b61c9", "#8a8a8a"]


def parse_args():
    p = argparse.ArgumentParser(description="FX markup and margin analysis")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-10-01")
    p.add_argument("--currencies", nargs="+", default=["EUR", "GBP", "CAD", "MXN", "INR"])
    p.add_argument("--target-margin", type=float, default=0.008,
                   help="margin the provider wants to keep after FX moves (0.008 = 0.8%%)")
    p.add_argument("--competitor-markup", type=float, default=0.015,
                   help="markup a typical bank charges over mid-market (0.015 = 1.5%%)")
    p.add_argument("--holding-days", type=int, default=2,
                   help="business days between quoting the customer and covering the trade")
    p.add_argument("--confidence", type=float, default=0.95,
                   help="share of trades that must not lose money after the FX move")
    p.add_argument("--train-end", default=None,
                   help="fit markups on dates up to this one and test on later dates (e.g. 2024-12-31)")
    p.add_argument("--output", default="output")
    p.add_argument("--demo", action="store_true", help="use synthetic rates (never for real results)")
    return p.parse_args()


def fetch_rates(start, end, currencies):
    """Return a DataFrame of foreign-currency units per 1 USD, one row per business day."""
    symbols = ",".join(currencies)
    errors = []
    for template in API_URLS:
        url = template.format(start=start, end=end, symbols=symbols)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fx-pricing-analysis"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.load(resp)
            rates = pd.DataFrame.from_dict(payload["rates"], orient="index")
            rates.index = pd.to_datetime(rates.index)
            rates = rates.sort_index()[currencies].astype(float)
            print(f"Fetched {len(rates)} days of rates from {url.split('/')[2]}")
            return rates
        except Exception as exc:  # try the next mirror
            errors.append(f"{url}: {exc}")
    sys.exit("Could not fetch FX rates:\n  " + "\n  ".join(errors))


def demo_rates(start, end, currencies):
    """Synthetic random-walk rates. Only for checking that the script runs."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start, end)
    spot = {"EUR": 0.92, "GBP": 0.79, "CAD": 1.36, "MXN": 18.0, "INR": 83.0}
    vol = {"EUR": 0.07, "GBP": 0.08, "CAD": 0.06, "MXN": 0.13, "INR": 0.04}
    data = {}
    for c in currencies:
        shocks = rng.normal(0, vol.get(c, 0.08) / np.sqrt(TRADING_DAYS), len(dates))
        data[c] = spot.get(c, 1.0) * np.exp(np.cumsum(shocks))
    return pd.DataFrame(data, index=dates)


def realized_margin(markup, log_move):
    """
    Margin kept on a transfer quoted at mid_t * (1 - markup).

    The provider promises the customer mid_t * (1 - markup) units per USD, then
    buys those units h days later at mid_{t+h}. log_move = ln(mid_{t+h} / mid_t);
    a negative move (USD weakened) makes the units more expensive.
    """
    return 1 - (1 - markup) * np.exp(-log_move)


def recommend_markup(moves, target, confidence):
    """Return (adverse quantile move, recommended markup) for a set of h-day log moves."""
    # adverse move that is only exceeded on (1 - confidence) share of days
    q = np.quantile(moves, 1 - confidence)
    # markup that breaks even at that move: realized_margin(m, q) == 0
    break_even = 1 - np.exp(q)
    # earn at least the target, and lose money on at most (1 - confidence) of trades
    return q, max(target, break_even)


def out_of_sample(rates, target, competitor, h, confidence, train_end):
    """Fit markups on dates up to train_end, then backtest them on later dates only."""
    fwd_move = (np.log(rates).shift(-h) - np.log(rates)).dropna()
    # drop the last h training quotes: their outcome falls inside the test period
    train = fwd_move.loc[:train_end].iloc[:-h]
    test = fwd_move.loc[fwd_move.index > pd.Timestamp(train_end)]
    if train.empty or test.empty:
        sys.exit(f"--train-end {train_end} leaves no training or no test data")

    rows = []
    for c in rates.columns:
        _, markup = recommend_markup(train[c].values, target, confidence)
        in_sample = realized_margin(markup, train[c].values)
        ours = realized_margin(markup, test[c].values)
        theirs = realized_margin(competitor, test[c].values)
        rows.append({
            "currency": c,
            "train_markup": markup,
            "train_pct_days_loss": (in_sample < 0).mean(),
            "test_mean_margin": ours.mean(),
            "test_pct_days_loss": (ours < 0).mean(),
            "test_pct_days_below_target": (ours < target).mean(),
            "test_competitor_mean_margin": theirs.mean(),
        })
    table = pd.DataFrame(rows).set_index("currency")
    return table, (train.index[0], train.index[-1]), (test.index[0], test.index[-1])


def analyze(rates, target, competitor, h, confidence):
    log_ret = np.log(rates).diff().dropna()
    # h-day forward move for a quote made on day t
    fwd_move = (np.log(rates).shift(-h) - np.log(rates)).dropna()

    rows = []
    for c in rates.columns:
        moves = fwd_move[c].values
        q, markup = recommend_markup(moves, target, confidence)
        ours = realized_margin(markup, moves)
        theirs = realized_margin(competitor, moves)
        rows.append({
            "currency": c,
            "annualized_vol": log_ret[c].std() * np.sqrt(TRADING_DAYS),
            f"worst_{h}d_move_{int(confidence*100)}pct": q,
            "fx_risk_buffer": markup - target,
            "recommended_markup": markup,
            "competitor_markup": competitor,
            "customer_saving_per_1000usd": 1000 * (competitor - markup),
            "backtest_mean_margin": ours.mean(),
            "backtest_pct_days_below_target": (ours < target).mean(),
            "backtest_pct_days_loss": (ours < 0).mean(),
            "competitor_mean_margin": theirs.mean(),
        })
    table = pd.DataFrame(rows).set_index("currency")
    rolling_vol = log_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
    return table, rolling_vol


def plot_markups(table, target, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cur = table.index
    x = np.arange(len(cur))
    ax.bar(x, [target * 100] * len(cur), color=COLORS[0], label=f"Target margin ({target:.1%})")
    ax.bar(x, table["fx_risk_buffer"] * 100,
           bottom=target * 100, color=COLORS[1], label="Extra FX risk buffer")
    ax.axhline(table["competitor_markup"].iloc[0] * 100, color="#333", ls="--", lw=1.2,
               label=f"Competitor markup ({table['competitor_markup'].iloc[0]:.1%})")
    for i, v in enumerate(table["recommended_markup"] * 100):
        ax.text(i, v + 0.03, f"{v:.2f}%", ha="center", fontsize=9)
    ax.set_xticks(x, cur)
    ax.set_ylabel("Markup over mid-market (%)")
    ax.set_title("Recommended markup by currency")
    ax.set_ylim(0, max(table["competitor_markup"].iloc[0], table["recommended_markup"].max()) * 100 * 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(out / "markup_by_currency.png", dpi=150)
    plt.close(fig)


def plot_volatility(rolling_vol, out):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, c in enumerate(rolling_vol.columns):
        ax.plot(rolling_vol.index, rolling_vol[c] * 100, color=COLORS[i % len(COLORS)], lw=1.3, label=c)
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_title("21-day rolling volatility vs USD")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=len(rolling_vol.columns), fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "rolling_volatility.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    currencies = [c.upper() for c in args.currencies]
    out = Path(args.output)
    out.mkdir(exist_ok=True)

    if args.demo:
        print("DEMO MODE: synthetic rates. Do not use these numbers anywhere.")
        rates = demo_rates(args.start, args.end, currencies)
    else:
        rates = fetch_rates(args.start, args.end, currencies)
        rates = rates.loc[args.start:args.end]
        rates.to_csv(out / "fx_rates.csv")

    table, rolling_vol = analyze(rates, args.target_margin, args.competitor_markup,
                                 args.holding_days, args.confidence)
    table.round(6).to_csv(out / "pricing_table.csv")
    plot_markups(table, args.target_margin, out)
    plot_volatility(rolling_vol, out)

    pd.set_option("display.width", 200)
    print(f"\nPeriod {rates.index[0].date()} to {rates.index[-1].date()}, "
          f"{args.holding_days}-day holding, {args.confidence:.0%} confidence")
    show = table[["annualized_vol", "recommended_markup", "customer_saving_per_1000usd",
                  "backtest_mean_margin", "backtest_pct_days_loss", "competitor_mean_margin"]]
    print(show.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote {out / 'pricing_table.csv'}, markup_by_currency.png, rolling_volatility.png")

    if args.train_end:
        oos, train_span, test_span = out_of_sample(rates, args.target_margin, args.competitor_markup,
                                                   args.holding_days, args.confidence, args.train_end)
        oos.round(6).to_csv(out / "out_of_sample.csv")
        print(f"\nOut-of-sample: fit {train_span[0].date()} to {train_span[1].date()}, "
              f"test {test_span[0].date()} to {test_span[1].date()}")
        print(oos.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"\nWrote {out / 'out_of_sample.csv'}")


if __name__ == "__main__":
    main()
