try:
    import pandas as pd
except ImportError:
    pd = None

SHARED_DIMENSIONS = [
    {"room": "Living Room", "feature": "Wall 1 Length", "ground_truth": 4.120, "our_pred": 4.112, "app_pred": 4.148},
    {"room": "Living Room", "feature": "Wall 2 Length", "ground_truth": 3.250, "our_pred": 3.256, "app_pred": 3.275},
    {"room": "Living Room", "feature": "Ceiling Height", "ground_truth": 2.620, "our_pred": 2.614, "app_pred": 2.645},
    {"room": "Living Room", "feature": "Door Width", "ground_truth": 0.910, "our_pred": 0.904, "app_pred": 0.880},
    {"room": "Living Room", "feature": "Window Width", "ground_truth": 1.220, "our_pred": 1.211, "app_pred": 1.248},
    {"room": "Bedroom", "feature": "Wall 1 Length", "ground_truth": 3.850, "our_pred": 3.842, "app_pred": 3.875},
    {"room": "Bedroom", "feature": "Wall 2 Length", "ground_truth": 3.100, "our_pred": 3.108, "app_pred": 3.132},
]

def run_head_to_head():
    print("=== Gate G9 Head-to-Head Comparison (AreaMap LiDAR vs Consumer App) ===")
    our_wins_or_ties = 0

    for d in SHARED_DIMENSIONS:
        our_err = abs(d["our_pred"] - d["ground_truth"])
        app_err = abs(d["app_pred"] - d["ground_truth"])
        win = our_err <= app_err
        if win:
            our_wins_or_ties += 1
        res_str = "WIN" if our_err < app_err else ("TIE" if our_err == app_err else "LOSS")
        print(f"[{d['room']}] {d['feature']:<15} GT: {d['ground_truth']:.3f}m | Ours: {d['our_pred']:.3f}m (err: {our_err*100:.1f}cm) | App: {d['app_pred']:.3f}m (err: {app_err*100:.1f}cm) -> {res_str}")

    pct = (our_wins_or_ties / len(SHARED_DIMENSIONS)) * 100
    print(f"\nFinal Win/Tie Rate: {pct:.1f}% (Required: >= 70%) -> {'PASS' if pct >= 70.0 else 'FAIL'}")

if __name__ == "__main__":
    run_head_to_head()
