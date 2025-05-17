import cv2, json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path

# 是否显示监控画面
monitor_on = True

model = YOLO('yolov8n.pt')

# 初始化人脸检测器
face_detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

# 初始化人脸识别模型
face_recognizer = cv2.FaceRecognizerSF.create(
    "face_recognition_sface_2021dec.onnx",  # 需下载模型文件
    ""
)

# 构建人脸特征数据库
known_faces = []
faces_dir = Path("faces")
for img_path in faces_dir.iterdir():
    if img_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
        name = img_path.stem
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 人脸检测
        face_detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = face_detector.detect(img)
        
        if faces is not None and len(faces) > 0:
            # 提取第一个人脸特征
            face = faces[0]
            aligned_face = face_recognizer.alignCrop(img, face)
            feature = face_recognizer.feature(aligned_face)
            known_faces.append({'name': name, 'feature': feature})
            print(f"Face registered: {name}")

# 打开视频文件
video_dir = Path("videos")
video_name = Path("test_video_2.mp4")
input_path = video_dir / video_name
cap = cv2.VideoCapture(input_path)

# 获取视频原始尺寸
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置窗口宽度
window_width = 1100

# 获取视频数据
fps = cap.get(cv2.CAP_PROP_FPS)
total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 帧间隔设置
time_interval = 1  # second
frame_interval = int(fps * time_interval)
frame_number = 0

# 初始化追踪器
max_age_second = 10
max_age = int(max_age_second / time_interval)
tracker = DeepSort(max_age=max_age, n_init=2)
tracking_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number % frame_interval == 0:
        print(f"Frame {frame_number} / {total_frame_count}", f"{frame_number/total_frame_count*100:.2f}%")
        # YOLO人物检测
        results = model.predict(frame, conf=0.15, iou=0.2, classes=[0], imgsz=1024)
        
        # 处理检测结果
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
        detections = []
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            conf = confidences[i]
            class_id = class_ids[i]
            if class_id == 0:
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, str(class_id)))

        # 更新追踪器
        tracks = tracker.update_tracks(detections, frame=frame)
        annotated_frame = results[0].plot(line_width=2)  # 绘制检测结果

        frame_entry = {
            "timestamp": frame_number / fps,
            "frame": frame_number,
            "tracks": []
        }

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
            state = "sleeping"
            if x2 - x1 > 0 and y2 - y1 > 0:  # 有效区域检查
                roi = frame[y1:y2, x1:x2]
                h, w = roi.shape[:2]
                
                if h > 0 and w > 0:  # 防止空图像
                    face_detector.setInputSize((w, h))
                    _, faces = face_detector.detect(roi)
                    if faces is not None and len(faces) > 0:  # face detected
                        state = "awake"
            
            if track.time_since_update != 0:
                state = "untracked"

            # 人脸识别处理
            face_id = "unknown"
            if x2 - x1 > 0 and y2 - y1 > 0:
                roi = frame[y1:y2, x1:x2]
                h, w = roi.shape[:2]
                
                if h > 0 and w > 0:
                    face_detector.setInputSize((w, h))
                    _, faces = face_detector.detect(roi)
                    
                    if faces is not None and len(faces) > 0:
                        face = faces[0]
                        try:
                            # 人脸对齐和特征提取
                            aligned_face = face_recognizer.alignCrop(roi, face)
                            current_feature = face_recognizer.feature(aligned_face)
                            
                            # 特征比对
                            best_match = None
                            highest_score = 0
                            for known in known_faces:
                                score = face_recognizer.match(
                                    current_feature, 
                                    known['feature'],
                                    cv2.FaceRecognizerSF_FR_COSINE
                                )
                                if score > highest_score:
                                    highest_score = score
                                    best_match = known['name']
                            
                            # 相似度阈值判断
                            face_id = best_match if highest_score >= 0.4 else "unknown"
                        except Exception as e:
                            pass

            track_info = {
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "face_id": face_id,
                "state": state,
            }
            frame_entry["tracks"].append(track_info)

            if monitor_on:
                # set colour
                if state == "awake":
                    colour = (0, 255, 0)  # blue, green, red
                elif state == "sleeping":
                    colour = (0, 255, 255)  # blue, green, red
                elif state == "untracked":
                    colour = (200, 200, 200)  # blue, green, red
                else:
                    ValueError("Invalid state")

                # 显示文本
                cv2.putText(
                    annotated_frame, 
                    f"ID {track_id} {face_id} {state}",
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    colour,
                    5,
                )

        # save data
        tracking_data.append(frame_entry)

        if monitor_on:
            # 缩放窗口
            scale = window_width / annotated_frame.shape[1]
            width = int(annotated_frame.shape[1] * scale)
            height = int(annotated_frame.shape[0] * scale)
            resized_frame = cv2.resize(annotated_frame, (width, height), interpolation=cv2.INTER_AREA)
            # show monitor window
            cv2.imshow('Frame', resized_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
                break

    frame_number += 1

print("Done")

# save data file
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{video_name}.json"

with open(output_path, 'w') as f:
    json.dump(tracking_data, f, indent=2, ensure_ascii=False)

print(f"Data saved to {output_path}")

cap.release()
cv2.destroyAllWindows()
