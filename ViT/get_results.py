import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT_DIR = "./"  # 
THRESHOLD = 0.87
matched_dirs = []

def find_event_dir(base_dir):
    """event"""
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                return root
    return None

all_dirs = [
    os.path.join(ROOT_DIR, d)
    for d in os.listdir(ROOT_DIR)
    if os.path.isdir(os.path.join(ROOT_DIR, d)) and "food101-run" in d
]

for d in all_dirs:
    event_dir = find_event_dir(d)
    if event_dir is None:
        print(f"🚫 {d} -> no event file found")
        continue

    try:
        event_acc = EventAccumulator(event_dir)
        event_acc.Reload()
        tags = event_acc.Tags().get("scalars", [])

        if "eval/accuracy" in tags:
            events = event_acc.Scalars("eval/accuracy")
            max_val = max(e.value for e in events)
            if max_val > THRESHOLD:
                print(f"✅ {d} -> max eval/accuracy: {max_val:.4f}")
                matched_dirs.append(d)
            else:
                print(f"❌ {d} -> max eval/accuracy: {max_val:.4f}")
        else:
            print(f"⚠️  {d} -> no 'eval/accuracy' found in events")
    except Exception as e:
        print(f"‼️ Error processing {event_dir}: {e}")

# 
with open("high_accuracy_dirs.txt", "w") as f:
    for d in matched_dirs:
        f.write(d + "\n")

print("\n🎯 Exported matched directories to high_accuracy_dirs.txt")