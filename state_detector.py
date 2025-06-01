import cv2, json, ctypes
from datetime import datetime, timedelta
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path

# running flag
running = True


def parse_video_time(video_name):
    """解析视频文件名中的时间信息"""
    parts = video_name.stem.split("_")
    start_str = parts[-2]
    end_str = parts[-1]
    
    start_time = datetime.strptime(start_str, "%Y%m%d%H%M%S")
    end_time = datetime.strptime(end_str, "%Y%m%d%H%M%S")
    return start_time, end_time


def process_videos(start_time: datetime, end_time: datetime, monitor_on=True):
    """处理指定时间范围内的多个视频"""
    # 帧间隔设置
    time_interval = 1  # second

    # 初始化模型和追踪器
    model = YOLO("yolov8n.pt")
    max_age_second = 10
    max_age = int(max_age_second / time_interval)
    tracker = DeepSort(max_age=max_age, n_init=2)  # 全局追踪器
    state_data = []
    
    # 构建人脸数据库
    face_detector, face_recognizer, known_faces = init_face_system()

    # 获取符合条件的视频文件
    video_dir = Path("videos")
    video_files = []
    for video_path in video_dir.glob("video_*.mp4"):
        try:
            video_start, video_end = parse_video_time(video_path)
            # 判断时间范围是否有交集
            if (video_end > start_time) and (video_start < end_time):
                video_files.append((video_start, video_end, video_path))
        except:
            continue

    # 按时间顺序排序视频
    video_files.sort(key=lambda x: x[0])

    # 初始化窗口尺寸
    window_width = None

    # 遍历处理每个视频
    for video_start, video_end, video_path in video_files:
        if not running:
            break

        # 计算实际需要处理的时间段
        clip_start = max(video_start, start_time)
        clip_end = min(video_end, end_time)

        print(f"Processing {video_path.name} [{clip_start} - {clip_end}]\n")
        
        # 获取视频数据
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frame_count = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_interval = round(fps * time_interval)
        
        # 计算起始偏移帧数
        start_offset = max(0, round((clip_start - video_start).total_seconds() * fps))
        end_offset = min(total_frame_count, round((clip_end - video_start).total_seconds() * fps))
        
        # 设置帧读取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_offset)
        
        # 视频处理主循环
        frame_number = start_offset
        while running and cap.isOpened() and frame_number <= end_offset:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                # 计算绝对时间戳
                current_time = video_start + timedelta(seconds=frame_number/fps)
                print(f"Current: {current_time}")
                if current_time > clip_end:
                    break

                process_frame(
                    frame=frame,
                    frame_number=frame_number,
                    current_time=current_time,
                    model=model,
                    tracker=tracker,
                    face_detector=face_detector,
                    face_recognizer=face_recognizer,
                    known_faces=known_faces,
                    state_data=state_data,
                    monitor_on=monitor_on,
                    window_width=window_width
                )

                print(f"{(current_time - start_time) / (end_time - start_time) * 100:.2f}%")
                print()
            
            frame_number += 1

        cap.release()

    print("100%\nDone!")
    # 保存最终数据
    data_name = f"{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}.json"
    save_data(state_data, data_name)
    cv2.destroyAllWindows()


def init_face_system():
    """初始化人脸识别系统"""
    face_detector = cv2.FaceDetectorYN.create(
        "face_detection_yunet_2023mar.onnx", "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )
    
    face_recognizer = cv2.FaceRecognizerSF.create(
        "face_recognition_sface_2021dec.onnx", ""
    )

    known_faces = []
    faces_dir = Path("faces")
    for img_path in faces_dir.iterdir():
        if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            name = img_path.stem
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            face_detector.setInputSize((img.shape[1], img.shape[0]))
            _, faces = face_detector.detect(img)
            
            if faces is not None and len(faces) > 0:
                aligned_face = face_recognizer.alignCrop(img, faces[0])
                feature = face_recognizer.feature(aligned_face)
                known_faces.append({
                    "name": name,
                    "feature": feature
                })
                print(f"Face registered: {name}")
    
    return face_detector, face_recognizer, known_faces


def process_frame(**kwargs):
    """处理单个帧的核心逻辑"""
    process_start_time = datetime.now()

    # 解包参数
    frame = kwargs["frame"]
    current_time = kwargs["current_time"]
    tracker = kwargs["tracker"]
    
    # YOLO检测
    results = kwargs["model"].predict(frame, conf=0.15, iou=0.2, classes=[0], imgsz=1024, verbose=False)
    
    # 处理检测结果
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    
    # 构建检测结果
    detections = []
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        conf = confidences[i]
        if class_ids[i] == 0:  # 仅处理人物类别
            detections.append(([x1, y1, x2-x1, y2-y1], conf, "person"))
    
    # 更新追踪器
    tracks = tracker.update_tracks(detections, frame=frame)
    annotated_frame = results[0].plot(line_width=2)

    frame_entry = {
        "time": current_time.strftime("%Y%m%d%H%M%S"),
        "tracks": []
    }

    # 处理每个追踪目标
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        # 获取追踪信息
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        # 状态检测和人脸识别逻辑
        state = detect_state(frame, track, kwargs["face_detector"])
        face_id = recognize_face(frame, track, kwargs["face_detector"], 
                                kwargs["face_recognizer"], kwargs["known_faces"])
        
        # save track data
        track_data = {
            "track_id": track_id,
            "bbox": [x1, y1, x2, y2],
            "face_id": face_id.split("_")[0],  # remove suffix "_glasses"
            "state": state,
        }
        frame_entry["tracks"].append(track_data)
        
        # 可视化显示
        if kwargs["monitor_on"]:
            draw_annotation(annotated_frame, track_id, face_id, state, x1, y1)

    # save frame data
    kwargs["state_data"].append(frame_entry)

    # 显示监控画面
    if kwargs["monitor_on"]:
        display_frame(annotated_frame, kwargs["window_width"])

    process_end_time = datetime.now()
    print(f"Time elasped: {(process_end_time - process_start_time).total_seconds():.3f} s")


def detect_state(frame, track, face_detector):
    """检测目标状态"""
    ltrb = track.to_ltrb()
    x1, y1, x2, y2 = map(int, ltrb)
    
    state = "sleeping"
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        
        if h > 0 and w > 0:
            face_detector.setInputSize((w, h))
            _, faces = face_detector.detect(roi)
            if faces is not None and len(faces) > 0:
                state = "awake"
    
    if track.time_since_update != 0:
        state = "untracked"
    
    return state


def recognize_face(frame, track, detector, recognizer, known_faces):
    """人脸识别"""
    ltrb = track.to_ltrb()
    x1, y1, x2, y2 = map(int, ltrb)
    
    face_id = "unknown"
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        
        if h > 0 and w > 0:
            detector.setInputSize((w, h))
            _, faces = detector.detect(roi)
            
            if faces is not None and len(faces) > 0:
                try:
                    aligned = recognizer.alignCrop(roi, faces[0])
                    feature = recognizer.feature(aligned)
                    
                    best_score = 0.4
                    best_name = "unknown"
                    for known in known_faces:
                        score = recognizer.match(feature, known["feature"], 
                                               cv2.FaceRecognizerSF_FR_COSINE)
                        if score > best_score:
                            best_score = score
                            best_name = known["name"]
                    
                    face_id = best_name.split("_")[0]
                except:
                    pass
    return face_id


def draw_annotation(frame, track_id, face_id, state, x1, y1):
    """绘制标注信息"""
    colour_map = {
        "awake": (0, 255, 0),
        "sleeping": (0, 255, 255),
        "untracked": (200, 200, 200)
    }
    colour = colour_map.get(state, (200, 200, 200))
    
    cv2.putText(
        frame, f"ID {track_id} {face_id}",
        (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 5
    )


def display_frame(frame, window_width):
    global running
    """显示监控画面"""
    if window_width is None:
        if frame.shape[1] >= frame.shape[0]:
            window_width = int(0.9 * ctypes.windll.user32.GetSystemMetrics(0))
        else:
            window_height = int(0.9 * ctypes.windll.user32.GetSystemMetrics(1))
            window_width = int(frame.shape[1] * window_height / frame.shape[0])
    
    scale = window_width / frame.shape[1]
    resized_frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
    cv2.imshow("Monitor", resized_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
        running = False


def save_data(data, filename):
    """保存结果数据"""
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir/filename, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {output_dir/filename}")


# 使用示例
if __name__ == "__main__":
    start_time = datetime(2025, 3, 6, 18, 45, 11)
    end_time = datetime(2025, 3, 6, 18, 56)
    process_videos(start_time, end_time)
