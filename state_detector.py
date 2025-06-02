import cv2, json, torch, threading
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import multiprocessing
from tqdm import tqdm

multiprocessing.set_start_method('spawn', force=True)

def parse_video_time(video_name):
    """Parse the time information in the video file name"""
    parts = video_name.stem.split("_")
    start_str = parts[-2]
    end_str = parts[-1]
    
    start_time = datetime.strptime(start_str, "%Y%m%d%H%M%S")
    end_time = datetime.strptime(end_str, "%Y%m%d%H%M%S")
    return start_time, end_time

def get_video_metadata(video_path):
    """预获取视频元数据"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps

def process_single_video(video_info, global_start, global_end, update_queue):
    """处理单个视频的独立函数"""
    video_start, video_end, video_path = video_info

    # 必须在函数内部初始化CUDA相关组件
    import cv2
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort

    # 显式释放CUDA缓存
    torch.cuda.empty_cache()
    
    # 初始化模型（每个进程独立实例）
    model = YOLO("yolov8n.pt").to("cuda")
    face_detector, face_recognizer, known_faces = init_face_system()
    
    # 计算实际处理范围
    clip_start = max(video_start, global_start)
    clip_end = min(video_end, global_end)
    
    # 处理逻辑（与原代码相似，略作调整）
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_offset = max(0, round((clip_start - video_start).total_seconds() * fps))
    end_offset = min(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 
                    round((clip_end - video_start).total_seconds() * fps))
    
    last_report = start_offset
    report_interval = max(1, (end_offset - start_offset) // 1000)  # 至少1帧

    # 独立跟踪器实例
    time_interval = 2
    max_age = int(10 / time_interval)
    tracker = DeepSort(max_age=max_age, n_init=2)
    frame_interval = round(fps * time_interval)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_offset)
    
    video_data = []
    frame_number = start_offset

    try:
        while running and cap.isOpened() and frame_number <= end_offset:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                # Calculate absolute timestamp
                current_time = video_start + timedelta(seconds=frame_number/fps)
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
                    video_data=video_data,
                )
    
            frame_number += 1

            if frame_number - last_report >= report_interval:
                update_queue.put( (video_path, frame_number - last_report) )
                last_report = frame_number
                
    finally:
        # 提交剩余进度
        if frame_number > last_report:
            update_queue.put( (video_path, frame_number - last_report) )
        cap.release()
            
    return video_data

def process_videos(start_time: datetime, end_time: datetime):
    """Process multiple videos within the specified time range"""

    # Get video files in the range
    video_dir = Path("videos")
    video_files = []
    for video_path in video_dir.glob("video_*.mp4"):
        try:
            video_start, video_end = parse_video_time(video_path)
            # Determine if there is an overlap in the time range
            if (video_end > start_time) and (video_start < end_time):
                video_files.append((video_start, video_end, video_path))
        except:
            continue

    # Sort video files by start time
    video_files.sort(key=lambda x: x[0])

    ctx = multiprocessing.get_context('spawn')

    # 预加载所有视频元数据
    video_metas = []
    for v in video_files:
        video_path = v[2]
        try:
            total_frames, fps = get_video_metadata(video_path)
            video_metas.append((*v, total_frames, fps))
        except Exception as e:
            print(f"Error reading {video_path}: {str(e)}")
            continue
    
    # 根据GPU数量设置workers
    gpu_count = 16  # 根据实际GPU数量调整
    max_workers = min(gpu_count, len(video_files))

    # 创建全量进度条
    progress_bars = {}
    max_desc_width = max(len(str(v[2].stem)) for v in video_metas)
    for idx, meta in enumerate(video_metas):
        video_start, video_end, video_path, total_frames, fps = meta
        desc = f"📽 {video_path.stem[:max_desc_width]}".ljust(max_desc_width+2)
        pbar = tqdm(
            total=total_frames,
            desc=desc,
            position=idx,
            bar_format="{desc}: {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            leave=False
        )
        progress_bars[video_path] = pbar

    # 创建进度更新队列
    manager = multiprocessing.Manager()
    update_queue = manager.Queue()

    # 启动进度监听线程
    def progress_monitor():
        while True:
            msg = update_queue.get()
            if msg is None:  # 终止信号
                return
            video_path, delta = msg
            if video_path in progress_bars:
                progress_bars[video_path].update(delta)

    monitor_thread = threading.Thread(target=progress_monitor)
    monitor_thread.start()

    # 修改多进程任务参数
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(max_workers) as pool:
        tasks = [( (*meta[:3],), start_time, end_time, update_queue) 
                for meta in video_metas]
        
        try:
            results = []
            for res in pool.starmap(process_single_video, tasks):
                results.extend(res)
        finally:
            update_queue.put(None)
            monitor_thread.join()
            # 关闭所有进度条
            [pbar.close() for pbar in progress_bars.values()]
    
    # 按时间排序
    results.sort(key=lambda x: x["time"])
    return results


def init_face_system():
    """Initialize the face recognition system"""
    face_detector = cv2.FaceDetectorYN.create(
        "face_detection_yunet_2023mar.onnx", "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )
    
    face_recognizer = cv2.FaceRecognizerSF.create(
        "face_recognition_sface_2021dec.onnx", 
        "", 
    )

    known_faces = []
    try:
        with open("face_features.json", "r") as f:
            features_data = json.load(f)
            
            for name, features in features_data.items():
                for feature in features:
                    # 将list转换回numpy array
                    np_feature = np.array(feature, dtype=np.float32).reshape(1, -1)
                    known_faces.append({
                        "name": name,
                        "feature": np_feature
                    })
    except FileNotFoundError:
        print("face_features.json not found, please run preprocess_faces.py")
        exit(1)
    
    return face_detector, face_recognizer, known_faces

def process_frame(**kwargs):
    """Process a single frame"""
    # Unpack parameters
    frame = kwargs["frame"]
    current_time = kwargs["current_time"]
    tracker = kwargs["tracker"]
    
    # YOLO detection
    results = kwargs["model"].predict(
        frame, 
        conf=0.15, 
        iou=0.2, 
        classes=[0], 
        imgsz=1024, 
        verbose=False,
        device='cuda'
    )
    
    # Process detection results
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    
    # Build detection results
    detections = []
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        conf = confidences[i]
        if class_ids[i] == 0:  # Only process the person class
            detections.append(([x1, y1, x2-x1, y2-y1], conf, "person"))
    
    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    frame_entry = {
        "time": current_time.strftime("%Y%m%d%H%M%S"),
        "tracks": []
    }

    # Process each tracking target
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        # Get tracking information
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        # State detection and face recognition logic
        state = detect_state(frame, track, kwargs["face_detector"])
        face_id = recognize_face(frame, track, kwargs["face_detector"], 
                                kwargs["face_recognizer"], kwargs["known_faces"])
        
        # Save track data
        track_data = {
            "track_id": track_id,
            "bbox": [x1, y1, x2, y2],
            "face_id": face_id.split("_")[0],  # Remove suffix "_glasses"
            "state": state,
        }
        frame_entry["tracks"].append(track_data)

    # Save frame data
    kwargs["video_data"].append(frame_entry)


def detect_state(frame, track, face_detector):
    """Detect the state of the target"""
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
    """Face recognition"""
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


running = True

if __name__ == "__main__":
    start_time = datetime(2025, 3, 7, 7, 0)
    end_time = datetime(2025, 3, 7, 8, 0)
    data = process_videos(start_time, end_time)

    # Save final data
    data_name = f"{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}.json"
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir/data_name, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {output_dir/data_name}")

