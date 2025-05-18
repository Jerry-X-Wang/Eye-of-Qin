import json
from pathlib import Path

def process_video_data(data):
    # Initialize a dictionary to store time information for each face_id
    state_time = {}

    # Start processing from the first timestamp >= count_time
    start_index = next(i for i, entry in enumerate(data) if entry["timestamp"] >= count_time)
    
    for i in range(start_index, len(data)):
        current_entry = data[i]
        current_timestamp = current_entry["timestamp"]
        previous_timestamp = data[i-1]["timestamp"]
        time_diff = current_timestamp - previous_timestamp
        
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
                    if current_timestamp - entry["timestamp"] > count_time:
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
                    state_time[face_id]["awake"] += time_diff
                elif sleeping_ratio >= 0.6:
                    state_time[face_id]["sleeping"] += time_diff

    # Convert the face_times dictionary to the desired output format
    output = [
        {
            "id": face_id, 
            "state_time": state_time[face_id]
        }
        for face_id in state_time.items()
    ]

    return output

count_time = 60  # seconds. The time interval to count the state.

# Load the JSON data from the file
input_dir = Path("data/processed")
data_name = Path("test_video_2.mp4.json")
input_path = input_dir / data_name

with open(input_path, 'r') as file:
    data = json.load(file)

result = process_video_data(data)

output_dir = Path("data/time")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / data_name

with open(output_path, 'w') as file:
    json.dump(result, file, indent=4)

print(f"Time data saved to {output_path}")
