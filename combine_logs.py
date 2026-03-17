import os
import re

log_dir = "C:\\Users\\along\\OneDrive\\Documents\\GitHub\\AI-Portfolio"
output_file = os.path.join(log_dir, "combined_training_log.txt")

# Get all log files
log_files = [f for f in os.listdir(log_dir) if f.startswith("training_log_") and f.endswith(".txt")]

# Extract timestamps and sort
logs_with_time = []
time_pattern = re.compile(r"training_log_(\d{8}_\d{6})\.txt")
for log_file in log_files:
    match = time_pattern.match(log_file)
    if match:
        timestamp = match.group(1)
        logs_with_time.append((timestamp, log_file))

logs_with_time.sort()  # Sort by timestamp

# Combine sorted logs
with open(output_file, "w") as outfile:
    for timestamp, log_file in logs_with_time:
        file_path = os.path.join(log_dir, log_file)
        with open(file_path, "r") as infile:
            outfile.write(f"\n--- Log from {log_file} ---\n")
            outfile.write(infile.read())

print(f"Combined logs saved to {output_file}")