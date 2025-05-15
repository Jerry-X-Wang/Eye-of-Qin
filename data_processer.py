import json
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path

def process_data(input_path):
    # 读取原始数据
    with open(input_path) as f:
        data = json.load(f)
    
    # 转换为更方便处理的结构
    all_points = []
    for frame in data:
        for track in frame["tracks"]:
            all_points.append({
                "timestamp": frame["timestamp"],
                "track_id": track["track_id"],
                "position": track["position"],
                "original_state": track["state"],
                "identity": track["identity"]
            })
    
    # 需求1：合并位置相近的ID
    # 使用DBSCAN聚类算法
    positions = np.array([[p["position"]["x"], p["position"]["y"]] for p in all_points])
    clustering = DBSCAN(eps=50, min_samples=2).fit(positions)  # 50像素为邻域半径
    
    # 创建ID映射表
    id_map = {}
    new_id = 1
    for label, point in zip(clustering.labels_, all_points):
        if label == -1:  # 噪声点保持原ID
            id_map[point["track_id"]] = f"ID-{new_id}"
            new_id +=1
        else:
            if label not in id_map:
                id_map[label] = f"CLS-{new_id}"
                new_id +=1
            point["processed_id"] = id_map[label]
    
    # 更新所有点的ID
    for p in all_points:
        p["processed_id"] = id_map.get(p["track_id"], p["track_id"])
    
    # 需求2：身份传播
    identity_records = defaultdict(lambda: {"last_identity": "unknown", "history": []})
    
    # 按时间排序
    sorted_points = sorted(all_points, key=lambda x: x["timestamp"])
    
    for p in sorted_points:
        record = identity_records[p["processed_id"]]
        if p["identity"] != "unknown":
            record["last_identity"] = p["identity"]
            record["history"].append(p["identity"])
        else:
            # 使用最近的非unknown身份
            if len(record["history"]) > 0:
                p["processed_identity"] = record["last_identity"]
            else:
                p["processed_identity"] = "unknown"
        
        # 保留最近5个识别结果
        if len(record["history"]) > 5:
            record["history"].pop(0)
    
    # 需求3：状态覆盖
    time_window = 60  # 60秒窗口
    state_records = defaultdict(list)
    
    for p in sorted_points:
        # 获取时间窗口内的记录
        window_start = p["timestamp"] - time_window
        window_records = [
            r for r in state_records[p["processed_id"]]
            if r["timestamp"] >= window_start
        ]
        
        # 统计状态比例
        total = len(window_records)
        sleeping_count = len([r for r in window_records 
                            if r["original_state"] in ("sleeping", "untracked")])
        
        # 添加当前记录到历史
        state_records[p["processed_id"]].append({
            "timestamp": p["timestamp"],
            "original_state": p["original_state"]
        })
        
        # 应用覆盖规则
        if total > 0 and (sleeping_count / total) >= 0.8:
            p["processed_state"] = "sleeping"
        else:
            p["processed_state"] = p["original_state"]
    
    # 重组数据结构
    processed_data = []
    current_frame = None
    
    for p in sorted_points:
        if not current_frame or current_frame["timestamp"] != p["timestamp"]:
            current_frame = {
                "timestamp": p["timestamp"],
                "processed_tracks": []
            }
            processed_data.append(current_frame)
        
        current_frame["processed_tracks"].append({
            "original_id": p["track_id"],
            "processed_id": p["processed_id"],
            "position": p["position"],
            "original_state": p["original_state"],
            "processed_state": p["processed_state"],
            "processed_identity": p.get("processed_identity", "unknown")
        })
    
    return processed_data

# 使用
input_dir = Path("data/raw")
data_name = Path("video_0042_0_10_20250307080100_20250307081139.json")
processed = process_data(input_dir / data_name)

# 保存结果
output_dir = Path("data/processed")
output_path = output_dir / f"{data_name}.json"

with open(output_path, "w") as f:
    json.dump(processed, f, indent=2, ensure_ascii=False)
