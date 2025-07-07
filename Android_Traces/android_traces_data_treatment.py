import json
from datetime import datetime
from collections import defaultdict

# === CONFIG ===
INPUT_FILE = "state_traces.json"
OUTPUT_FILE = "aggregated_durations.json"

# Event pairs to compute durations for
paired_events = {
    "screen_on": "screen_off",
    "battery_charged_on": "battery_charged_off",
    "screen_unlock": "screen_lock"
}

# === Parse one message_str ===
def parse_event_string(message_str):
    grouped = defaultdict(list)
    lines = message_str.strip().splitlines()

    for line in lines:
        try:
            # Remove extra whitespace
            parts = line.strip().split()
            # print(parts)
            # break

            if len(parts) < 3:
                continue  # Not enough parts to form timestamp + event

            # Reconstruct timestamp and event
            timestamp_str = f"{parts[0]} {parts[1]}"
            event = parts[2]

            # Skip battery percentages or unknowns
            if "%" in event or event.lower() == "unknown":
                continue

            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            grouped[event].append(dt)
        except Exception as e:
            continue
    return grouped


# === Compute durations ===
def compute_durations(grouped):
    result = {}

    for start_event, end_event in paired_events.items():
        starts = grouped.get(start_event, [])
        ends = grouped.get(end_event, [])

        # Merge all timestamps with a tag
        timeline = [(dt, "start") for dt in starts] + [(dt, "end") for dt in ends]
        timeline.sort()  # sort by datetime

        stack = []
        total = 0.0

        for dt, kind in timeline:
            if kind == "start":
                stack.append(dt)
            elif kind == "end" and stack:
                start_time = stack.pop(0)  # FIFO: match earliest unmatched start
                if dt > start_time:
                    total += (dt - start_time).total_seconds()

        result[f"{start_event}_duration"] = round(total, 2)

    return result




# === MAIN PROCESSING ===
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

output = []
for guid, content in data.items():
    message_str = content.get("messages", "")
    # print(message_str)
    # break
    grouped_events = parse_event_string(message_str)

    print(f"\nGUID: {guid}")
    for event, timestamps in grouped_events.items():
        print(f"{event}: {len(timestamps)} events")


    durations = compute_durations(grouped_events)

    result = {
        "guid": content.get("guid", guid),
        "model": content.get("model", "unknown"),
        **durations
    }
    output.append(result)

# === SAVE ===
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved summary to {OUTPUT_FILE}")
