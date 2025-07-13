import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# 1. DATASET  (feature_names removed – only close_index is required)
# ---------------------------------------------------------------------
class CrossSectionDataset(Dataset):
    """Sliding-window, multi-asset dataset.

    Parameters
    ----------
    data : ndarray (T, N, F)
        Daily feature cube (time × asset × feature).
    dates : array-like (T,)
        Matching datetime index.
    window : int
        Look-back length W.
    close_index : int
        Column index of the *close* price in axis-2.
    produce_cls / produce_reg : bool
        Toggle creation of classification / regression targets.
    """
    def __init__(self,
                 data: np.ndarray,
                 dates: np.ndarray,
                 window: int,
                 close_index: int,
                 produce_cls: bool = True,
                 produce_reg: bool = True):

        T, N, F = data.shape
        S = T - window - 1
        self.data        = data
        self.dates_all   = np.asarray(dates)
        self.close_index = close_index
        self.produce_cls = produce_cls
        self.produce_reg = produce_reg
        self.N, self.F, self.W = N, F, window

        self.X = np.zeros((S, N, window * F), dtype=np.float32)
        self.y_cls = np.zeros((S, N), dtype=np.int64)   if produce_cls else None
        self.y_reg = np.zeros((S, N), dtype=np.float32) if produce_reg else None
        self.dates = []

        for t in range(window, T - 1):
            i = t - window
            past = data[t-window:t]  # (W,N,F)
            self.X[i] = past.transpose(1, 0, 2).reshape(N, window * F)

            today_close = data[t, :, close_index]
            next_close  = data[t+1, :, close_index]
            if produce_cls:
                self.y_cls[i] = (next_close > today_close).astype(np.int64)
            if produce_reg:
                self.y_reg[i] = next_close
            self.dates.append(pd.to_datetime(dates[t+1]))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        feats = torch.from_numpy(self.X[idx])
        cls   = torch.from_numpy(self.y_cls[idx]) if self.produce_cls else torch.empty(0)
        reg   = torch.from_numpy(self.y_reg[idx]) if self.produce_reg else torch.empty(0)
        return feats, cls, reg

    def split(self, ratio: float = 0.8):
        S = int(len(self) * ratio)
        self.dates_test = self.dates[S:]
        return (self.X[:S],  self.y_cls[:S] if self.produce_cls else None, self.y_reg[:S] if self.produce_reg else None,
                self.X[S:], self.y_cls[S:] if self.produce_cls else None, self.y_reg[S:] if self.produce_reg else None)

# ---------------------------------------------------------------------
# 2. BACKTESTER – simplified for direct compatibility (no reshape)
# ---------------------------------------------------------------------
class Backtester:
    """Evaluate a pre-trained model and compute back-test metrics."""

    def __init__(self, dataset: CrossSectionDataset, split_ratio: float = .45):
        (self.X_tr, self.yc_tr, self.yr_tr,
         self.X_te, self.yc_te, self.yr_te) = dataset.split(split_ratio)
        self.dates_test  = np.asarray(dataset.dates_test)
        self.data        = dataset.data
        self.dates_all   = dataset.dates_all
        self.close_idx   = dataset.close_index
        self.N, self.F, self.W = dataset.N, dataset.F, dataset.W
        self.num_test    = self.X_te.shape[0]

    def run(self, model: nn.Module, thresh: float = 0.5):
        X_test_np = self.X_te.copy()  # already (S, N, W*F)
        y_r   = model.predict_reg(X_test_np)              # (S,N)
        y_prob= model.predict_clf(X_test_np)              # (S,N)
        y_sig = (y_prob > thresh).astype(int)

        ml = {}
        if self.yr_te is not None:
            ml['mse'] = mean_squared_error(self.yr_te.flatten(), y_r.flatten())
            ml['mae'] = mean_absolute_error(self.yr_te.flatten(), y_r.flatten())
        if self.yc_te is not None:
            ml['acc'] = accuracy_score(self.yc_te.flatten(), y_sig.flatten())

        rets = np.zeros_like(y_r)
        p_t  = np.zeros_like(y_r)
        for i, dt in enumerate(self.dates_test):
            idx = np.where(self.dates_all == np.datetime64(dt))[0][0]
            p_t[i]  = self.data[idx-1, :, self.close_idx]
            nxt     = self.data[idx,   :, self.close_idx]
            rets[i] = nxt / p_t[i] - 1

        pnl_c  = (y_sig * rets).mean(axis=1)
        pred_r = (y_r - p_t) / p_t
        pnl_rb = ((pred_r > 0).astype(int) * rets).mean(axis=1)
        abs_sum= np.abs(pred_r).sum(axis=1, keepdims=True)
        weights= np.divide(pred_r, abs_sum, where=abs_sum>0)
        pnl_rc = (weights * rets).sum(axis=1)

        def perf(pnl):
            eq = np.cumprod(1+pnl)
            cum = eq[-1]-1
            std = pnl.std()
            sharpe = pnl.mean()/std*np.sqrt(252) if std>0 else np.nan
            dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
            return cum, sharpe, dd.min()

        bt = pd.DataFrame([perf(pnl_c), perf(pnl_rb), perf(pnl_rc)],
                          index=['classification','regression-binary','regression-continuous'],
                          columns=['cumulative_return','sharpe','max_drawdown'])
        return ml, bt

# ---------------------------------------------------------------------
# 3. HELPER – convert yfinance DataFrame → (T, N, F) cube
# ---------------------------------------------------------------------

def build_feature_cube(df: pd.DataFrame, tickers: list[str], feature_order: list[str]) -> np.ndarray:
    """Convert a yfinance OHLCV DataFrame to a 3‑D numpy cube.

    Parameters
    ----------
    df : DataFrame
        Multi‑index columns: level‑0 = feature (e.g. 'Open'), level‑1 = ticker.
    tickers : list[str]
        Tickers in the order you want along axis‑1 (N).
    feature_order : list[str]
        Features in the order you want along axis‑2 (F).  The index of 'Close'
        in this list is the close_index you supply to CrossSectionDataset.

    Returns
    -------
    cube : ndarray (T, N, F)
    """
    T = len(df)
    N = len(tickers)
    F = len(feature_order)
    cube = np.empty((T, N, F), dtype=np.float32)
    for f_idx, feat in enumerate(feature_order):
        wide = df[feat][tickers].ffill().values  # (T,N)
        cube[:, :, f_idx] = wide
    return cube



import yfinance as yf
from torch.utils.data import DataLoader

def prepare_data_and_backtester(tickers, start, end, features, close_feature, window, split_ratio=0.45, batch_size=64):
    """
    Downloads market data, creates dataset and backtester, and returns DataLoader.

    Parameters:
    - tickers: list[str] - stock symbols
    - start, end: str - date range
    - features: list[str] - features to use (e.g., ['Open', 'High', 'Low', 'Close'])
    - close_feature: str - the feature to use as the close price (e.g., 'Close')
    - window: int - historical window length
    - split_ratio: float - train/test split ratio
    - batch_size: int - DataLoader batch size

    Returns:
    - ds: CrossSectionDataset instance
    - bt: Backtester instance
    - train_loader: DataLoader for training data
    """

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
    cube = build_feature_cube(raw, tickers=tickers, feature_order=features)

    close_index = features.index(close_feature)
    ds = CrossSectionDataset(
        data=cube,
        dates=raw.index.to_numpy(),
        window=window,
        close_index=close_index,
        produce_cls=True,
        produce_reg=True
    )

    bt = Backtester(ds, split_ratio=split_ratio)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return ds, bt, train_loader
