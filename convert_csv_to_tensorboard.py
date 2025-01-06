import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# metrics.csv のパスを指定
csv_path = "lightning_logs/version_1/metrics.csv"

# TensorBoard ログの保存先を指定
log_dir = "mylog/version_1/tensorboard_logs"

# CSV を読み込む
df = pd.read_csv(csv_path)

# TensorBoard SummaryWriter を初期化
writer = SummaryWriter(log_dir)

# 各メトリックを TensorBoard に書き込む
for _, row in df.iterrows():
    for col_name, value in row.items():
        if col_name == "step":  # "step" をグローバルステップとして扱う
            step = value
        elif col_name != "epoch" and not pd.isna(value):  # "epoch" は無視
            writer.add_scalar(col_name, value, step)

# SummaryWriter を閉じる
writer.close()

print(f"TensorBoard logs saved to: {log_dir}")