import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 加载YOLOv8的默认预训练模型
model = YOLO('yolov8n.pt')

# 初始化面部检测模型
face_detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",  # 需要自行下载模型文件
    "",
    (320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

# 打开视频文件
cap = cv2.VideoCapture("test_video.mp4")

# 获取视频原始尺寸
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置窗口宽度
window_width = 1100

# 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 帧间隔设置
time_interval = 1  # second
frame_interval = int(fps * time_interval)
frame_number = 0

# 初始化追踪器
max_age_second = 10
max_age = int(max_age_second / time_interval)
tracker = DeepSort(max_age=max_age, n_init=2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number % frame_interval == 0:
        # YOLO人物检测
        results = model.predict(frame, conf=0.15, iou=0.2, classes=[0], imgsz=1024)
        
        # 处理检测结果
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = confidences[i]
            class_id = class_ids[i]
            if class_id == 0:
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, str(class_id)))

        # 更新追踪器
        tracks = tracker.update_tracks(detections, frame=frame)
        annotated_frame = results[0].plot(line_width=2)  # 绘制检测结果

        # 处理每个追踪目标
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # 边界检查
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_width, x2)
            y2 = min(original_height, y2)
            
            # 姿态检测
            state = "Sleeping"
            if x2 - x1 > 0 and y2 - y1 > 0:  # 有效区域检查
                roi = frame[y1:y2, x1:x2]
                h, w = roi.shape[:2]
                
                if h > 0 and w > 0:  # 防止空图像
                    face_detector.setInputSize((w, h))
                    _, faces = face_detector.detect(roi)
                    if faces is not None and len(faces) > 0:  # face detected
                        state = "Awake"

            
            if track.time_since_update != 0:
                state = "Untracked"
                
            if state == "Awake":
                colour = (0, 255, 0)  # blue, green, red
            elif state == "Sleeping":
                colour = (0, 255, 255)  # blue, green, red
            elif state == "Untracked":
                colour = (0, 170, 170)  # blue, green, red
            else:
                ValueError("Invalid state")
                
            # 绘制追踪信息和姿态
            cv2.putText(
                annotated_frame, 
                f"ID {track_id} {state}",
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,  # font size
                colour,
                5,  # font thickness
            )

        # 缩放窗口
        scale = window_width / annotated_frame.shape[1]
        width = int(annotated_frame.shape[1] * scale)
        height = int(annotated_frame.shape[0] * scale)
        resized_frame = cv2.resize(annotated_frame, (width, height), interpolation=cv2.INTER_AREA)
        
        cv2.imshow('Frame', resized_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
            break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()
