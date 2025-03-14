import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# 初始化YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 更改为 yolov5s

tracker = DeepSort(max_age=10)  # 增加 max_age
cap = cv2.VideoCapture("test_video-class_time.mp4")

# 性能优化参数
target_fps = 1  # 目标帧率
frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / target_fps)  # 计算跳帧间隔
frame_count = 0

# 窗口显示固定宽度
display_width = 1150  # 设置固定显示宽度

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_interval != 0:  # 跳帧处理
        continue

    # YOLOv5检测
    results = model(frame) 
    detections = results.xyxy[0].cpu().numpy()
    
    boxes = []
    confidences = []
    for detection in detections:
        if detection[4] >= 0 and detection[5] == 0:  # 置信度阈值
            x1, y1, x2, y2, confidence, class_id = detection
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(confidence)

    # 使用NMS消除重叠框
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.0, 0.5)  # 调整NMS参数
    filtered_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            filtered_detections.append(([x, y, w, h], confidences[i], None))

    # 更新跟踪器
    tracks = tracker.update_tracks(filtered_detections, frame=frame)
    
    # 绘制结果
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        
        cv2.rectangle(frame, 
                     (int(ltrb[0]), int(ltrb[1])),
                     (int(ltrb[2]), int(ltrb[3])),
                     (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", 
                   (int(ltrb[0]), int(ltrb[1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (0, 255, 0), 2)
    
    # 调整显示尺寸（保持宽高比）
    h, w = frame.shape[:2]
    display_height = int(display_width * (h / w))  # 根据宽度计算高度
    display_frame = cv2.resize(frame, (display_width, display_height))
    
    cv2.imshow("Tracking", display_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
