import json
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm  # Import the tqdm library

# Function to convert string to datetime object
def str_to_time(time_str):
    return datetime.strptime(time_str, "%Y%m%d%H%M%S")

# Function to process video data
def process_video_data(input_path):
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Initialize a dictionary to store time information for each face_id
    state_time = {}

    start_time = datetime.strptime(data[0]["time"], "%Y%m%d%H%M%S")

    # Start processing from the first time >= count_time
    try:
        start_index = next(i for i, entry in enumerate(data) if str_to_time(entry["time"]) - start_time >= timedelta(seconds=count_time))
    except StopIteration:
        print(f"Data is too short to count state for {count_time} seconds")
        return

    # Process data from start_index to the end of the data
    for i in tqdm(range(start_index, len(data)), desc="Processing data"):  # Use tqdm to display a progress bar
        current_entry = data[i]
        current_time = str_to_time(current_entry["time"])
        previous_time = str_to_time(data[i - 1]["time"])
        time_delta = current_time - previous_time

        for track in current_entry["tracks"]:
            face_id = track["face_id"]
            if face_id != "unknown":
                if face_id not in state_time:
                    state_time[face_id] = {"awake": 0, "sleeping": 0}

                # Check the state over the previous count_time seconds
                count_awake = 0
                count_sleeping = 0
                count_untracked = 0

                for j in range(i, -1, -1):
                    entry = data[j]
                    if current_time - str_to_time(entry["time"]) > timedelta(seconds=count_time):
                        break

                    for t in entry["tracks"]:
                        if t["face_id"] == face_id:
                            if t["state"] == "awake":
                                count_awake += 1
                            elif t["state"] == "sleeping":
                                count_sleeping += 1
                            elif t["state"] == "untracked":
                                count_untracked += 1

                total_count = count_awake + count_sleeping + count_untracked

                if total_count == 0:
                    continue

                awake_ratio = count_awake / total_count
                sleeping_ratio = count_sleeping / total_count

                if awake_ratio >= 0.2:
                    state_time[face_id]["awake"] += time_delta.total_seconds()
                elif sleeping_ratio >= 0.7:
                    state_time[face_id]["sleeping"] += time_delta.total_seconds()

    # Convert the state_time dictionary to the desired output format
    output = sorted(
        [
            {
                "id": face_id,
                "state_time": state_time[face_id]
            }
            for face_id, _ in state_time.items()
        ],
        key=lambda x: x['id']
    )

    return output

# Time interval to count the state
count_time = 60  # seconds

# Load the JSON data from the file
input_dir = Path("data/processed")
data_name = Path("20250306185000_20250306185500.json")
input_path = input_dir / data_name

# Process the data and save the result
result = process_video_data(input_path)

if result is not None:
    output_dir = Path("data/time")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / data_name

    with open(output_path, 'w') as file:
        json.dump(result, file, indent=4)

    print(f"Time data saved to {output_path}")
else:
    print("Failed to save data")
