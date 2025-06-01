import json
from pathlib import Path
from tqdm import tqdm

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    
    # Calculate the boundaries of the intersection
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    
    # Calculate the area of the intersection
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Calculate the area of bbox1 and bbox2
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    
    # Calculate the area of the union
    union_area = area1 + area2 - inter_area
    
    # Calculate IoU
    if union_area == 0:
        return 0
    return inter_area / union_area

def process_data(input_path):
    # Read the original data
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Define an IoU threshold to determine if positions are similar
    IOU_THRESHOLD = 0.5

    # Traverse each track in each timestamp, add progress bar
    for i, entry in tqdm(enumerate(data), total=len(data), desc="Processing tracks"):
        tracks = entry["tracks"]
        
        for track in tracks:
            if track["face_id"] != "unknown":
                bbox = track["bbox"]
                face_id = track["face_id"]
                
                # Cover face_id forwards
                for j in range(i - 1, -1, -1):
                    prev_entry = data[j]
                    prev_tracks = prev_entry["tracks"]
                    max_iou = 0
                    max_iou_track = None
                    face_exist = False
                    for prev_track in prev_tracks:
                        if prev_track["face_id"] == face_id:
                            face_exist = True
                            break
                        iou = calculate_iou(bbox, prev_track["bbox"])
                        if iou >= IOU_THRESHOLD and iou > max_iou:
                            max_iou = iou
                            max_iou_track = prev_track
                    if face_exist:
                        break
                    if max_iou_track and max_iou_track["face_id"] == "unknown":
                        max_iou_track["face_id"] = face_id
                        # No need to break the loop here, as the index is moving backwards and the replacement is forward
                
                # Cover face_id backwards
                for j in range(i + 1, len(data)):
                    next_entry = data[j]
                    next_tracks = next_entry["tracks"]
                    max_iou = 0
                    max_iou_track = None
                    face_exist = False
                    for next_track in next_tracks:
                        if next_track["face_id"] == face_id:
                            face_exist = True
                            break
                        iou = calculate_iou(bbox, next_track["bbox"])
                        if iou >= IOU_THRESHOLD and iou > max_iou:
                            max_iou = iou
                            max_iou_track = next_track
                    if face_exist:
                        break
                    if max_iou_track and max_iou_track["face_id"] == "unknown":
                        max_iou_track["face_id"] = face_id
                        break  # Break the loop to avoid repeated coverings

    return data


if __name__ == "__main__":
    input_dir = Path("data/raw")
    data_name = Path("20250306185000_20250306185500.json")
    input_path = input_dir/data_name
    processed_data = process_data(input_path)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir/data_name

    print(f"Processed data saved to {output_path}")

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
