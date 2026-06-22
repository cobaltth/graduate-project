import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

# 1. 명령행 인자(Argument) 설정
parser = argparse.ArgumentParser(
    description="Plot training metrics starting from a specific epoch."
)
parser.add_argument(
    "--start",
    type=int,
    default=0,
    help="시작할 epoch 번호를 지정합니다. (기본값: 0)",
)
args = parser.parse_args()

# 2. 파일 상대 경로 설정 (현재 스크립트 위치 기준)
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "..", "train", "train.csv")

# 3. CSV 데이터 로드
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(
        f"Error: '{csv_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요."
    )
    exit()

# 4. 지정한 start epoch 이후의 데이터만 필터링
df = df[df["epoch"] >= args.start]

# 만약 필터링 후 데이터가 비어있다면 경고 후 종료
if df.empty:
    print(
        f"Warning: Epoch {args.start} 이후의 데이터가 csv 파일에 존재하지 않습니다."
    )
    exit()

# 5. Accuracy를 Error(1 - Accuracy)로 변환
df["train_error"] = 1 - df["train_acc"]
df["test_error"] = 1 - df["test_acc"]

# 6. 그래프 그리기 (Loss와 Error를 서브플롯으로 표현)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Left Plot: Loss ---
ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="blue")
ax1.plot(df["epoch"], df["test_loss"], label="Test Loss", color="red")
ax1.set_title(f"Training & Test Loss (From Epoch {args.start})")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)
ax1.legend()

# --- Right Plot: Error ---
ax2.plot(df["epoch"], df["train_error"], label="Train Error", color="blue")
ax2.plot(df["epoch"], df["test_error"], label="Test Error", color="red")
ax2.set_title(f"Training & Test Error (From Epoch {args.start})")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Error Rate")
ax2.grid(True)
ax2.legend()

plt.tight_layout()

# 7. 그래프를 'train' 폴더 안에 저장
output_plot_path = os.path.join(current_dir, "..", "train", "learning_curves.png")
plt.savefig(output_plot_path, dpi=300)
print(f"그래프가 성공적으로 저장되었습니다: {output_plot_path}")

plt.show()