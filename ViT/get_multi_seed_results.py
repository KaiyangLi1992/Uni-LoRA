import os
import ast
import statistics
from collections import defaultdict

folder_path = "logs"  # 

# : key = seed, value = list of accuracies
grouped_results = defaultdict(list)

for filename in os.listdir(folder_path):
    if not filename.endswith(".log"):
        continue
    file_path = os.path.join(folder_path, filename)
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            continue
        last_line = lines[-1].strip()
        try:
            data = ast.literal_eval(last_line)
            acc = data.get("eval_accuracy", None)
            if acc is None:
                acc = data.get("eval_test_accuracy", None)
            if acc is not None:
                #  seed ， base_cifar100_seed1_b0.01_h1.0.log -> base_cifar100_b0.01_h1.0.log
                prefix = filename.replace("_seed1", "_seed")\
                                 .replace("_seed2", "_seed")\
                                 .replace("_seed3", "_seed")\
                                 .replace("_seed4", "_seed")\
                                 .replace("_seed5", "_seed")
                grouped_results[prefix].append(acc)
        except Exception as e:
            print(f" {filename} : {e}")

# 
for prefix, accs in grouped_results.items():
    mean_acc = statistics.mean(accs)
    std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print(f"{prefix}: mean={mean_acc:.4f}, std={std_acc:.4f}, n={len(accs)}")