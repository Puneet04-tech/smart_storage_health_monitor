from pathlib import Path
import numpy as np
import pandas as pd

SMART_COLUMNS = [
    "smart_5","smart_187","smart_197","smart_198","smart_188","smart_194",
    "smart_9","smart_12","smart_171","smart_172","smart_177","smart_182",
    "smart_241","smart_1","smart_3","smart_4","smart_7","smart_10",
    "smart_193","smart_199","smart_240",
]

def generate(path: Path, samples: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    vendors = ["Seagate", "Western Digital", "Samsung", "Toshiba"]
    device_types = ["HDD", "SSD"]

    data = {
        "drive_id": [f"demo_drive_{i}" for i in range(samples)],
        "vendor": rng.choice(vendors, size=samples),
        "device_type": rng.choice(device_types, size=samples, p=[0.6, 0.4]),
        "recorded_at": pd.date_range(end=pd.Timestamp.now(), periods=samples, freq="h"),
    }

    # SMART-like random values
    data.update({
        "smart_5": rng.integers(0, 80, size=samples),
        "smart_187": rng.integers(0, 120, size=samples),
        "smart_197": rng.integers(0, 40, size=samples),
        "smart_198": rng.integers(0, 10, size=samples),
        "smart_188": rng.integers(0, 50, size=samples),
        "smart_194": rng.uniform(30, 70, size=samples),
        "smart_9": rng.integers(1000, 20000, size=samples),
        "smart_12": rng.integers(100, 1000, size=samples),
        "smart_171": rng.integers(0, 80, size=samples),
        "smart_172": rng.integers(0, 60, size=samples),
        "smart_177": rng.random(size=samples) * 100,
        "smart_182": rng.uniform(10, 100, size=samples),
        "smart_241": rng.integers(0, 200, size=samples),
        "smart_1": rng.integers(0, 100, size=samples),
        "smart_3": rng.uniform(1000, 5000, size=samples),
        "smart_4": rng.integers(100, 5000, size=samples),
        "smart_7": rng.integers(0, 50, size=samples),
        "smart_10": rng.integers(0, 30, size=samples),
        "smart_193": rng.integers(1000, 100000, size=samples),
        "smart_199": rng.integers(0, 100, size=samples),
        "smart_240": rng.integers(0, 10000, size=samples),
    })

    df = pd.DataFrame(data)

    # simple scoring -> label and rul_days
    score = (
        df["smart_5"] / 80 + df["smart_187"] / 120 + df["smart_171"] / 80 +
        df["smart_172"] / 60 + df["smart_1"] / 100 + np.abs(df["smart_194"] - 45) / 25
    )
    df["score"] = score / 6
    df["label"] = pd.cut(df["score"], bins=[-1, 0.35, 0.65, 10], labels=["healthy", "warning", "fail"]).astype(str)
    df["rul_days"] = np.where(df["label"] == "fail", rng.integers(1, 10, size=samples), rng.integers(30, 120, size=samples))

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote demo CSV to {path}")

if __name__ == '__main__':
    generate(Path("examples/demo_smart.csv"), samples=20)
