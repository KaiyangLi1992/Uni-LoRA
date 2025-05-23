import os
import csv
from tensorboard.backend.event_processing import event_accumulator

log_dir = "/home/kli16/Uni-LoRA_project_one/NLU_normalized/NLU/output"  # 改成你的 logdir
# target_tag = "eval/accuracy"
target_tag = "eval/pearson"

output_file = "accuracy_export.csv"
with open(output_file, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Run", "Step", "Value"])

    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "tfevents" in file:
                path = os.path.join(root, file)
                ea = event_accumulator.EventAccumulator(path)
                try:
                    ea.Reload()
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue

                if target_tag in ea.Tags().get("scalars", []):
                    run_name = os.path.relpath(root, log_dir)
                    for scalar in ea.Scalars(target_tag):
                        writer.writerow([run_name, scalar.step, scalar.value])


import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("accuracy_export.csv")

# 按 Run 分组，找出每个 Run 中最大的 Value
max_values = df.groupby("Run")["Value"].max().reset_index()

# 按照 Value 降序排列（可选）
max_values = max_values.sort_values(by="Value", ascending=False)

# 打印结果
print(max_values)

# 可选：保存为新的 CSV
max_values.to_csv("accuracy_max_by_run.csv", index=False)


import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("accuracy_max_by_run.csv")

def extract_scheme(run_string):
    parts = run_string.split('_')
    # 假设方案名称包含前五个下划线分隔的部分
    return '_'.join(parts[:5])

# 应用该函数创建一个新的 'Scheme' 列
df['Scheme'] = df['Run'].apply(extract_scheme)

# 按照 'Scheme' 分组，并计算平均值、中位数和标准差
grouped_stats = df.groupby('Scheme')['Value'].agg(['mean', 'median', 'std']).reset_index()

# 打印结果
print(grouped_stats)


df = grouped_stats

def get_group_key(scheme):
    parts = scheme.split('_')
    # 提取前两个下划线分隔的部分作为分组键
    return '_'.join(parts[:2])

# 应用该函数创建一个新的 'Group' 列
df['Group'] = df['Scheme'].apply(get_group_key)

# 按照 'Group' 分组，并找到每个组中 'median' 值最大的行
best_rows = df.loc[df.groupby('Group')['median'].idxmax()]

# 打印结果
print(best_rows)



