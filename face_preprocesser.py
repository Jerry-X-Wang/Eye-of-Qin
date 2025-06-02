import cv2
import json
from pathlib import Path
import numpy as np

def preprocess_face_features():
    """预处理人脸特征并保存到文件"""
    face_detector = cv2.FaceDetectorYN.create(
        "face_detection_yunet_2023mar.onnx", "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )
    
    face_recognizer = cv2.FaceRecognizerSF.create(
        "face_recognition_sface_2021dec.onnx", 
        ""
    )

    known_features = {}
    faces_dir = Path("faces")
    
    for img_path in faces_dir.iterdir():
        if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            name = img_path.stem.split('_')[0]  # 去除_glasses等后缀
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            face_detector.setInputSize((img.shape[1], img.shape[0]))
            _, faces = face_detector.detect(img)
            
            if faces is not None and len(faces) > 0:
                aligned_face = face_recognizer.alignCrop(img, faces[0])
                feature = face_recognizer.feature(aligned_face)
                
                # 转换为可序列化的Python list
                feature_list = feature.flatten().tolist()
                
                if name not in known_features:
                    known_features[name] = []
                known_features[name].append(feature_list)
                print(f"Processed: {name}")

    # 保存特征数据
    with open("face_features.json", "w") as f:
        json.dump(known_features, f)
        
    print("Face features saved to face_features.json")

if __name__ == "__main__":
    preprocess_face_features()
