import json
from pathlib import Path

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    
    # 计算交集的边界
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    
    # 计算交集面积
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # 计算bbox1和bbox2的面积
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    
    # 计算并集面积
    union_area = area1 + area2 - inter_area
    
    # 计算IoU
    if union_area == 0:
        return 0
    return inter_area / union_area

def process_data(input_path):
    # 读取原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 定义一个IoU阈值来判断位置是否相近
    IOU_THRESHOLD = 0.5

    # 遍历每个时间刻的每个track
    for i, entry in enumerate(data):
        timestamp = entry["timestamp"]
        frame = entry["frame"]
        tracks = entry["tracks"]
        
        for track in tracks:
            if track["face_id"] != "unknown":
                track_id = track["track_id"]
                bbox = track["bbox"]
                face_id = track["face_id"]
                
                # 往前覆盖face_id
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
                        # 此处不需跳出循环，因为循环的索引是向后的而这里的替换是向前的
                
                # 往后覆盖face_id
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
                        break  # 跳出循环，避免重复覆盖

    return data

input_dir = Path("data/raw")
data_name = Path("test_video_2.mp4.json")
input_path = input_dir / data_name
processed_data = process_data(input_path)

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / data_name

print(f"Processed data saved to {output_path}")

with open(output_path, "w", encoding='utf-8') as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)
