import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def build_feature_cube(df: pd.DataFrame, features: list[str], tickers: list[str]) -> np.ndarray:
    """
    Build a 3D NumPy array of shape (T, N, F) from a DataFrame.

    Steps:
    1. Select columns: subset df by features and tickers.
    2. Forward-fill (ffill) then backward-fill (bfill) missing values to ensure no NaNs.
    3. Extract underlying values array of shape (T, N, F).
    """
    # Slice DataFrame to get only the required features for each ticker
    wide = df[features][tickers]
    # Fill any missing data both forward and backward
    wide_filled = wide.ffill().bfill().values
    # Compute dimensions: T = time steps, N = tickers, F = features
    T, N, F = wide_filled.shape[0], len(tickers), len(features)
    # Reshape to (T, N, F) and return
    return wide_filled.reshape(T, N, F)


class CrossSectionDataset(Dataset):
    """
    PyTorch Dataset for cross-sectional modeling.

    Each sample corresponds to one prediction date:
      - Input X[i]: a tensor of shape (N, F, window) containing lookback features.
      - Label y[i]: raw price, return, or binary up/down, per ticker.
      - Date dates[i]: the prediction date for indexing/evaluation.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tickers: list[str],
        features: list[str],
        close_feature: str,
        window: int = 10,
        use_returns: bool = False,
        classification: bool = False,
        threshold: float = 0.0,
    ):
        super().__init__()
        # Validate that the target column exists in features
        if close_feature not in features:
            raise ValueError(f"close_feature '{close_feature}' must be in features list")
        # Classification mode requires computing returns first
        if classification and not use_returns:
            raise ValueError("Classification requires use_returns=True to compute return labels.")

        # Store mode flags
        self.use_returns = use_returns
        self.classification = classification
        self.threshold = threshold

        # Build full feature cube of shape (T, N, F)
        X_all = build_feature_cube(df, features, tickers)
        dates = df.index.to_numpy()
        T = X_all.shape[0]
        # Number of samples: drop the first 'window' days + the last day (target uses t+1)
        S = T - window - 1

        # Initialize storage for inputs and labels
        # X: (samples, tickers, features, window)
        self.X = np.zeros((S, len(tickers), len(features), window), dtype=np.float32)
        # y: int64 for classification, float32 for regression
        label_dtype = np.int64 if classification else np.float32
        self.y = np.zeros((S, len(tickers)), dtype=label_dtype)
        self.dates = []  # store the date for each sample

        # Determine which feature index is the 'close' price
        idx_close = features.index(close_feature)

        # Slide over time to populate X and y
        for t in range(window, T - 1):
            i = t - window  # sample index
            # Extract the previous `window` days of features for all tickers
            block = X_all[t-window:t, :, :]  # shape (window, N, F)
            # Rearrange to (N, F, window) for PyTorch conv/net compatibility
            self.X[i] = np.transpose(block, (1, 2, 0))

            # Next-day closing price for each ticker
            next_price = X_all[t + 1, :, idx_close]

            if use_returns:
                # Compute simple return: (P_{t+1} - P_t) / P_t
                today_price = X_all[t, :, idx_close]
                returns = (next_price - today_price) / today_price
                if classification:
                    # Binary classification: up/down based on threshold
                    self.y[i] = (returns > self.threshold).astype(np.int64)
                else:
                    # Regression on simple returns
                    self.y[i] = returns
            else:
                # Regression on raw next-day prices
                self.y[i] = next_price

            # Record the target date corresponding to next_price
            self.dates.append(pd.to_datetime(dates[t + 1]))

    def __len__(self) -> int:
        # Number of samples in the dataset
        return len(self.y)

    def __getitem__(self, idx: int):
        # Return a single sample: (features, label, date)
        x = torch.from_numpy(self.X[idx])  # tensor shape (N, F, window)
        y = torch.from_numpy(self.y[idx])  # tensor shape (N,)
        date = self.dates[idx]             # pandas.Timestamp
        return x, y, date


class Backtester:
    """
    Utility for evaluating predictions against true series.

    Attributes
    ----------
    dates_all: np.ndarray of all prediction dates (matching y).  
    prices_all: np.ndarray of true labels (prices or returns) aligned to df index.
    date_to_index: mapping from np.datetime64(date) -> integer index.
    """
    def __init__(self, dates_all: list[pd.Timestamp], prices_all: np.ndarray):
        # Store arrays for fast indexing
        self.dates_all = np.array(dates_all)
        self.prices_all = prices_all
        # Build dict for O(1) date lookups
        self.date_to_index = {np.datetime64(dt): i for i, dt in enumerate(self.dates_all)}

    def _index_of(self, date: pd.Timestamp) -> int:
        # Convert to np.datetime64 for dict key lookup
        key = np.datetime64(date)
        if key not in self.date_to_index:
            raise KeyError(f"Date {date} not found in index")
        return self.date_to_index[key]

    def run_backtest(self, pred: np.ndarray, dates: list[pd.Timestamp]) -> dict:
        """
        Compare predictions to true values and compute performance metrics.

        pred: array of predicted labels (shape: S×N or flattened)
        dates: list of target dates aligning each pred to actual price/return

        Returns a dict with:
          - equity: cumulative P&L series
          - return: final P&L
          - sharpe: annualized Sharpe ratio
          - max_drawdown: maximum peak-to-trough drawdown
        """
        # Map each prediction date to the index in `prices_all`
        idxs = [self._index_of(d) for d in dates]
        preds = pred.flatten()
        trues = self.prices_all[idxs].flatten()

        # Compute cumulative P&L: sum of (pred - true)
        eq = np.cumsum(preds - trues)
        rtn = eq[-1]
        # Period returns are daily P&L increments
        returns = np.diff(eq, prepend=0)
        # Annualized Sharpe: mean/std * sqrt(252)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)

        # Peak-to-trough drawdown
        peak = np.maximum.accumulate(eq)
        drawdowns = (peak - eq) / peak
        max_dd = np.max(drawdowns)

        return {"equity": eq, "return": rtn, "sharpe": sharpe, "max_drawdown": max_dd}


def prepare_data_and_backtester(
    df: pd.DataFrame,
    tickers: list[str],
    features: list[str],
    close_feature: str,
    window: int = 10,
    split_ratio: float = 0.8,
    batch_size: int = 32,
    task: str = 'regression',  # 'regression' or 'classification'
    use_returns: bool = False,  # only used in regression mode
    threshold: float = 0.0,      # classification cutoff on returns
):
    """
    Top-level function to instantiate dataset, dataloaders, and backtester.

    Parameters
    ----------
    df: DataFrame indexed by date, with columns for each ticker and feature.
    tickers: list of ticker symbols to include.
    features: list of feature names that appear in df columns.
    close_feature: feature name used as price target.
    window: number of lookback days per sample.
    split_ratio: fraction of samples for training set.
    batch_size: samples per batch in DataLoader.
    task: 'regression' (price/return) or 'classification' (binary up/down).
    use_returns: if True, regress on simple returns; otherwise regress on raw price.
    threshold: for classification, the return cutoff for labeling up/down.

    Returns
    -------
    ds: CrossSectionDataset instance.
    bt: Backtester instance for performance evaluation.
    train_loader: DataLoader for training split.
    test_loader: DataLoader for testing split.
    """
    # Determine classification mode
    classification = (task == 'classification')
    # Force returns if classifying
    if classification:
        use_returns = True

    # Create dataset
    ds = CrossSectionDataset(
        df,
        tickers,
        features,
        close_feature,
        window,
        use_returns=use_returns,
        classification=classification,
        threshold=threshold,
    )
    # Split indices for train/test
    n = len(ds)
    split_idx = int(split_ratio * n)
    train_idx = list(range(split_idx))
    test_idx = list(range(split_idx, n))

    # Instantiate PyTorch DataLoaders
    train_loader = DataLoader(
        torch.utils.data.Subset(ds, train_idx),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(ds, test_idx),
        batch_size=batch_size,
        shuffle=False
    )

    # Prepare backtester using true close prices
    prices_all = np.array(df[close_feature])
    bt = Backtester(ds.dates, prices_all)

    return ds, bt, train_loader, test_loader
