#!/usr/bin/env python3
import re
import sys
import pandas as pd
from pathlib import Path

def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logname.log")
    text_full = log_path.read_text(errors="ignore")

    # 1) Ignore the first 3 lines
    lines = text_full.splitlines()
    text = "\n".join(lines[3:]) if len(lines) > 3 else ""

    # 2) Split on header lines like: "YYYY-MM-DD ... [INFO] sizes:,"
    header_rx = r"^\s*.*\[INFO\]\s*sizes\s*:\s*,"
    blocks = re.split(header_rx, text, flags=re.M | re.I)[1:]

    # 3) Non-anchored regexes so timestamps/prefixes don't break matches
    seq_len_re = re.compile(r"\bseq_len:\s*(\d+)", re.I)
    pred_len_re = re.compile(r"\bpred_len:\s*(\d+)", re.I)
    model_re    = re.compile(r"\bModel:\s*([A-Za-z0-9_\-\.]+)", re.I)
    lr_re       = re.compile(r"\bLearning rate:\s*([0-9.eE\-]+)", re.I)
    wd_re       = re.compile(r"\bWeight decay:\s*([0-9.eE\-]+)", re.I)
    epochs_re   = re.compile(r"\bEpochs:\s*(\d+)", re.I)
    patience_re = re.compile(r"\bPatience:\s*(\d+)", re.I)
    test_re     = re.compile(
        r"\[(?P<model>[A-Za-z0-9_\-\.]+)\]\[test\]\s*MAE:\s*(?P<mae>[0-9.\-eE]+)\s*\|\s*MSE:\s*(?P<mse>[0-9.\-eE]+)\s*\|\s*Pinball\[(?P<pinball>[^\]]+)\]",
        re.I
    )

    def get_one(rx, block, cast=None):
        m = rx.search(block)
        if not m:
            return None
        v = m.group(1)
        if cast:
            try:
                return cast(v)
            except Exception:
                return None
        return v

    rows = []
    for block in blocks:
        seq_len = get_one(seq_len_re, block, int)
        pred_len = get_one(pred_len_re, block, int)
        model = get_one(model_re, block, str)
        lr = get_one(lr_re, block, float)
        wd = get_one(wd_re, block, float)
        epochs = get_one(epochs_re, block, int)
        patience = get_one(patience_re, block, int)

        tests = list(test_re.finditer(block))
        mae = mse = pinball = None
        if tests:
            t = tests[-1]
            try:
                mae = float(t.group("mae"))
            except Exception:
                mae = None
            try:
                mse = float(t.group("mse"))
            except Exception:
                mse = None
            pinball = t.group("pinball")
            if not model:
                model = t.group("model")

        if any(x is not None for x in [seq_len, pred_len, model, lr, wd, epochs, patience, mae, mse, pinball]):
            rows.append({
                "seq_len": seq_len,
                "pred_len": pred_len,
                "Model": model,
                "Learning rate": lr,
                "Weight decay": wd,
                "Epochs": epochs,
                "Patience": patience,
                "MAE [test]": mae,
                "MSE [test]": mse,
                "Pinball [test]": pinball
            })

    df = pd.DataFrame(rows, columns=[
        "seq_len","pred_len","Model","Learning rate","Weight decay",
        "Epochs","Patience","MAE [test]","MSE [test]","Pinball [test]"
    ]).sort_values(by=["Model","seq_len","pred_len"], kind="stable").reset_index(drop=True)

    out_csv = log_path.with_name(log_path.stem + ".ignore3.sizes_split.summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} experiments to {out_csv}")

if __name__ == "__main__":
    main()
