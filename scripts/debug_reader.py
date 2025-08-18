import sys

# Default range
start, end = 100, 120

# If user provides arguments, override defaults
if len(sys.argv) == 3:
    start = int(sys.argv[1])
    end = int(sys.argv[2])

with open("scripts/train_and_audit.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"\n--- Showing lines {start} to {end} from train_and_audit.py ---\n")

for i, ln in enumerate(lines, 1):
    if start <= i <= end:
        # Print line number and actual code (no quotes, no \n)
        print(f"{i:4d}: {ln}", end="")
# debug_reader.py
