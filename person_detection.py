import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# 初始化YOLO模型
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers().flatten()]

tracker = DeepSort(max_age=600)
cap = cv2.VideoCapture("test_video-class_time.mp4")

# 性能优化参数
target_fps = 1  # 目标处理帧率
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

    # YOLO检测
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.4:  # 降低置信度阈值
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # 使用NMS消除重叠框
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # 调整NMS参数
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            detections.append(([x, y, w, h], confidences[i], None))

    # 更新跟踪器
    tracks = tracker.update_tracks(detections, frame=frame)
    
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
