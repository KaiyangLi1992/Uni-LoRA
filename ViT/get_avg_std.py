import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_scalar_from_event_file(file_path, tag="eval/accuracy"):
    """
     event  tag  scalar 
    """
    try:
        ea = EventAccumulator(file_path)
        ea.Reload()
        if tag not in ea.Tags()["scalars"]:
            return None
        scalar_events = ea.Scalars(tag)
        if not scalar_events:
            return None
        return scalar_events[-1].value  # 
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

def collect_accuracies(root_dir, tag="eval/accuracy", keyword="cifar100"):
    """
     root_dir  keyword ， tag 
    """
    results = []
    for subdir, _, files in os.walk(root_dir):
        if keyword not in subdir:
            continue
        for file in files:
            if file.startswith("events.out.tfevents"):
                file_path = os.path.join(subdir, file)
                val = get_scalar_from_event_file(file_path, tag)
                if val is not None:
                    results.append(val)
                    print(f"✅ {file_path}: {val:.5f}")
    return results

if __name__ == "__main__":
    log_dir = "./output/"  #  TensorBoard 
    accuracies = collect_accuracies(log_dir)
    accuracies = accuracies[1::2] 
    print(accuracies)

    if accuracies:
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        print(f"\n📊 Final eval/accuracy for 'cifar100':")
        print(f"Mean: {mean:.5f}")
        print(f"Std:  {std:.5f}")
    else:
        print("⚠️ No valid accuracy values found.")