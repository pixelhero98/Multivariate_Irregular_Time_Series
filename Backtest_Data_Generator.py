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
    wide = df[features][tickers]
    wide_filled = wide.ffill().bfill().values
    T, N, F = wide_filled.shape[0], len(tickers), len(features)
    return wide_filled.reshape(T, N, F)


class CrossSectionDataset(Dataset):
    """
    PyTorch Dataset for cross-sectional modeling.

    Each sample corresponds to one prediction date:
      - Input X[i]: a tensor of shape (N, F, window) containing lookback features.
      - Label y[i]: log-return, or binary classification on log-return, per ticker.
      - Date dates[i]: the prediction date for indexing/evaluation.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tickers: list[str],
        features: list[str],
        close_feature: str,
        window: int = 10,
        use_log_returns: bool = True,
        classification: bool = False,
        threshold: float = 0.0,
    ):
        super().__init__()
        if close_feature not in features:
            raise ValueError(f"close_feature '{close_feature}' must be in features list")
        if classification and not use_log_returns:
            raise ValueError("Classification requires use_log_returns=True to compute return labels.")

        self.use_log_returns = use_log_returns
        self.classification = classification
        self.threshold = threshold

        X_all = build_feature_cube(df, features, tickers)
        dates = df.index.to_numpy()
        T = X_all.shape[0]
        # Ensure sufficient data length for windowed samples
        if T <= window + 1:
            raise ValueError(f"Not enough data: need at least {window+2} rows, got {T}")
        S = T - window - 1

        self.X = np.zeros((S, len(tickers), len(features), window), dtype=np.float32)
        label_dtype = np.int64 if classification else np.float32
        self.y = np.zeros((S, len(tickers)), dtype=label_dtype)
        self.dates = []

        idx_close = features.index(close_feature)

        for t in range(window, T - 1):
            i = t - window
            block = X_all[t-window:t, :, :]
            self.X[i] = np.transpose(block, (1, 2, 0))

            next_price = X_all[t + 1, :, idx_close]
            if use_log_returns:
                today_price = X_all[t, :, idx_close]
                # log-return: ln(P_{t+1}) - ln(P_t)
                log_returns = np.log(next_price) - np.log(today_price)
                if classification:
                    # binary label: up (1) if log-return > threshold, else 0
                    self.y[i] = (log_returns > self.threshold).astype(np.int64)
                else:
                    # regression on log-returns
                    self.y[i] = log_returns
            else:
                # regression on raw next-day prices
                self.y[i] = next_price

            self.dates.append(pd.to_datetime(dates[t + 1]))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.y[idx])
        date = self.dates[idx]
        return x, y, date


class Backtester:
    """
    Utility for evaluating predictions against true series.
    """
    def __init__(self, dates_all: list[pd.Timestamp], prices_all: np.ndarray):
        self.dates_all = np.array(dates_all)
        self.prices_all = prices_all
        self.date_to_index = {np.datetime64(dt): i for i, dt in enumerate(self.dates_all)}

    def _index_of(self, date: pd.Timestamp) -> int:
        key = np.datetime64(date)
        if key not in self.date_to_index:
            raise KeyError(f"Date {date} not found in index")
        return self.date_to_index[key]

    def run_backtest(self, pred: np.ndarray, dates: list[pd.Timestamp]) -> dict:
        idxs = [self._index_of(d) for d in dates]
        preds = pred.flatten()
        trues = self.prices_all[idxs].flatten()

        eq = np.cumsum(preds - trues)
        rtn = eq[-1]
        returns = np.diff(eq, prepend=0)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)

        peak = np.maximum.accumulate(eq)
        drawdowns = (peak - eq) / peak
        max_dd = np.max(drawdowns)

        return {"equity": eq, "return": rtn, "sharpe": sharpe, "max_drawdown": max_dd}


def compute_metrics(
    preds: np.ndarray,
    trues: np.ndarray,
    task: str = 'regression',
    threshold: float = 0.5,
) -> dict:
    """
    Compute deep-learning metrics on predictions and true labels.

    Parameters
    ----------
    preds: predicted values or probabilities/logits (shape: S×N).
    trues: true labels (shape: S×N).
    task: 'regression' or 'classification'.
    threshold: cutoff for binary classification when task='classification'.

    Returns
    -------
    dict containing:
      - For regression: 'mse', 'mae'.
      - For classification: 'accuracy', 'f1'.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
    results = {}
    preds_flat = preds.flatten()
    trues_flat = trues.flatten()

    if task == 'regression':
        results['mse'] = mean_squared_error(trues_flat, preds_flat)
        results['mae'] = mean_absolute_error(trues_flat, preds_flat)
    elif task == 'classification':
        pred_labels = (preds_flat > threshold).astype(int)
        true_labels = trues_flat.astype(int)
        results['accuracy'] = accuracy_score(true_labels, pred_labels)
        results['f1'] = f1_score(true_labels, pred_labels)
    else:
        raise ValueError("`task` must be 'regression' or 'classification'.")

    return results


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    task: str = 'regression',
    threshold: float = 0.5,
    device: torch.device | str | None = None,
) -> dict:
    """
    Evaluate a PyTorch model on a DataLoader and compute metrics.

    Parameters
    ----------
    model: PyTorch model to evaluate.     
    loader: DataLoader yielding (x, y, date).
    task: 'regression' or 'classification'.
    threshold: cutoff for binary classification.
    device: torch device (e.g., 'cuda' or 'cpu').

    Returns
    -------
    metrics dict as returned by compute_metrics.
    """
    import numpy as np
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            if device:
                x = x.to(device)
            preds = model(x)
            # If model returns logits for classification, apply sigmoid
            preds_np = preds.cpu().numpy()
            trues_np = y.cpu().numpy()
            all_preds.append(preds_np)
            all_trues.append(trues_np)
    preds_array = np.vstack(all_preds)
    trues_array = np.vstack(all_trues)
    return compute_metrics(preds_array, trues_array, task=task, threshold=threshold)


def prepare_data_and_backtester(
    df: pd.DataFrame,
    tickers: list[str],
    features: list[str],
    close_feature: str,
    window: int = 10,
    split_ratio: float = 0.8,
    batch_size: int = 32,
    task: str = 'regression',  # 'regression' or 'classification'
    use_log_returns: bool = True,  # default to log-return regression
    threshold: float = 0.0,      # classification cutoff on log-return
):
    classification = (task == 'classification')
    if classification:
        use_log_returns = True

    ds = CrossSectionDataset(
        df,
        tickers,
        features,
        close_feature,
        window,
        use_log_returns=use_log_returns,
        classification=classification,
        threshold=threshold,
    )
    n = len(ds)
    split_idx = int(split_ratio * n)
    train_idx = list(range(split_idx))
    test_idx = list(range(split_idx, n))

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

    prices_all = np.array(df[close_feature])
    bt = Backtester(ds.dates, prices_all)

    return ds, bt, train_loader, test_loader

# === Example: Training and Evaluation Workflow ===
#
# 1) Prepare data and backtester:
# ds, bt, train_loader, test_loader = \
#     prepare_data_and_backtester(
#         df, tickers, features, close_feature,
#         window=10,
#         split_ratio=0.8,
#         batch_size=32,
#         task='regression',  # or 'classification'
#         use_log_returns=True,
#         threshold=0.0
#     )
#
# 2) Training loop (simplified):
# model = YourModel(...)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# for epoch in range(epochs):
#     model.train()
#     for x_batch, y_batch, _ in train_loader:
#         preds = model(x_batch)
#         # choose appropriate loss:
#         # regression: loss_fn = nn.MSELoss(); loss = loss_fn(preds, y_batch)
#         # classification: loss_fn = nn.BCEWithLogitsLoss(); loss = loss_fn(preds, y_batch.float())
#         loss = loss_fn(preds, y_batch)
#         optimizer.zero_grad(); loss.backward(); optimizer.step()
#
#     # 3) Evaluate ML metrics on test set
#     dl_metrics = evaluate_model(model, test_loader, task='regression', threshold=0.0, device='cpu')
#     print(f"Epoch {epoch}: DL metrics ->", dl_metrics)
#
# 4) Backtest test-set predictions
# preds_all = []
# dates_all = []
# with torch.no_grad():
#     for x_batch, _, dates in test_loader:
#         preds = model(x_batch).cpu().numpy()
#         preds_all.append(preds)
#         dates_all.extend(dates)
# preds_array = np.vstack(preds_all)
# bt_results = bt.run_backtest(preds_array, dates_all)
# print("Backtest results:", bt_results)
