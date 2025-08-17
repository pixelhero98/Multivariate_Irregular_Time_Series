# --- CRYPTO DISCOVERY (add to find_tickers_3.py) ---
# --- CRYPTO DISCOVERY (patched) ---
import json, os, math, time
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from find_tickers import available_in_period, build_universe_by_ratios

# Expanded + case-insensitive stablecoin set (lowercase)
STABLE_SYMBOLS = {
    "usdt","usdc","busd","usdd","tusd","dai","fdusd","eusd","ustc",
    "pyusd","eurt","euroc","usde","usdn","gusd","usdp","lusd","susd","usdx","usdr"
}

def _fetch_coins_page(vs: str, page: int, per_page: int = 250):
    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        f"?vs_currency={vs}&order=market_cap_desc&per_page={per_page}&page={page}&sparkline=false"
    )
    # simple retry/backoff
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []

def candidates_crypto_top(symbol_quote: str = "USD", top_n: int = 200, exclude_stablecoins: bool = True):
    """
    Returns a list of Yahoo-style crypto tickers like ['BTC-USD','ETH-USD',...].
    - Paginates past 250 results.
    - Ranks by the requested quote (vs_currency=<symbol_quote>).
    - Skips common stablecoins when requested.
    """
    vs = symbol_quote.lower()
    per_page = 250
    pages = max(1, math.ceil(top_n / per_page))

    syms = []
    for page in range(1, pages + 1):
        data = _fetch_coins_page(vs, page, per_page=per_page)
        if not data:
            break
        for row in data:
            sym = str(row.get("symbol", "")).upper()
            if not sym:
                continue
            if exclude_stablecoins and sym.lower() in STABLE_SYMBOLS:
                continue
            syms.append(f"{sym}-{symbol_quote.upper()}")
        if len(syms) >= top_n:
            break

    # Respect top_n and de-dupe while preserving order
    syms = syms[:top_n]
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def build_crypto_universe(
    top_n=200,
    symbol_quote="USD",
    # modeling / split params (reuses your existing selector)
    start="2017-01-01", end="2025-06-30",
    train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
    rank_start="2020-01-01", rank_end="2022-12-31",
    K=120, H=10, buffer_days=50,
    presence_mode="dynamic",
    min_windows_train=64, min_windows_val=6, min_windows_test=32,
    min_price=0.0,                 # crypto can be <$1, don't screen out
    out_dir="./ldt"
):
    os.makedirs(out_dir, exist_ok=True)
    raw = candidates_crypto_top(symbol_quote=symbol_quote, top_n=top_n)
    # Filter by yfinance availability in the period
    ok, _missing = available_in_period(raw, start=start, end=end, batch=100)

    # Reuse your exact ratio/coverage/liquidity logic
    sel, stats = build_universe_by_ratios(
        tickers=ok,
        start=start, end=end,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        rank_start=rank_start, rank_end=rank_end,
        K=K, H=H, buffer_days=buffer_days,
        presence_mode=presence_mode,
        min_windows_train=min_windows_train,
        min_windows_val=min_windows_val,
        min_windows_test=min_windows_test,
        topN=min(len(ok), top_n),
        min_price=min_price,
        out_csv=os.path.join(out_dir, f"CRYPTO_Dynamic{min(len(ok), top_n)}_ratio.csv")
    )

    # Also write full "all OK" list
    with open(os.path.join(out_dir, "CRYPTO_top.txt"), "w") as f:
        f.writelines([t + "\n" for t in sel])
    with open(os.path.join(out_dir, "CRYPTO_all.txt"), "w") as f:
        f.writelines([t + "\n" for t in ok])

    print(f"[CRYPTO] OK candidates: {len(ok)}   selected top: {len(sel)}")
    print(f"Wrote: CRYPTO_top.txt, CRYPTO_all.txt in {os.path.abspath(out_dir)}")
    return sel, stats


# ------------------------ CLI entry ------------------------
if __name__ == "__main__":
    # Example defaults; adjust as needed.
    params = dict(
        top_n=500,
        start="2017-01-01", end="2025-06-30",
        train_ratio=0.55, val_ratio=0.05, test_ratio=0.4,
        rank_start="2020-01-01", rank_end="2022-12-31",
        K=150, H=40,
        presence_mode="dynamic",
        min_price=0.0, out_dir="./ldt"
    )
    build_crypto_universe(**params)
