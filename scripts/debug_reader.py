# debug_reader.py

with open("scripts/train_and_audit.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, ln in enumerate(lines, 1):
    if 100 <= i <= 120:  # adjust this range if needed
        print(i, repr(ln))
