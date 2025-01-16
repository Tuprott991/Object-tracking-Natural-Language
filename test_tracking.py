import cv2
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# Khởi tạo YOLO (Sử dụng YOLOv5)
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Tải YOLOv5 small model

# Khởi tạo mô hình CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Hàm để phát hiện đối tượng sử dụng YOLO
def detect_objects(frame):  
    results = model_yolo(frame)  # Phát hiện đối tượng trong frame
    detections = results.pandas().xywh[0]  # Chuyển kết quả sang dạng pandas
    return detections

# Hàm để tìm đối tượng giống nhất với câu query
def find_best_match(detections, query):
    inputs = clip_processor(text=[query], images=detections['image'].tolist(), return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    similarities = outputs.logits_per_image.detach().numpy()
    best_match_index = np.argmax(similarities)
    return detections.iloc[best_match_index]

# Hàm để theo dõi đối tượng qua các frame
def track_object(frame, object_bbox):
    tracker = cv2.TrackerCSRT_create()  # Sử dụng tracker CSRT
    tracker.init(frame, tuple(object_bbox))  # Khởi tạo tracker với bounding box của đối tượng
    return tracker

# Hàm chính để chạy toàn bộ hệ thống
def track_object_in_video(video_path, query):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    # Bước 1: Phát hiện đối tượng tại frame t0
    detections = detect_objects(frame)
    best_match = find_best_match(detections, query)

    # Lấy bounding box của đối tượng cần theo dõi
    object_bbox = best_match[['xmin', 'ymin', 'xmax', 'ymax']].values

    # Bước 2: Khởi tạo tracking
    tracker = track_object(frame, object_bbox)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Bước 3: Theo dõi đối tượng trong frame tiếp theo
        success, bbox = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in bbox]
            # Vẽ bounding box lên frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            # Nếu mất vết đối tượng, phát hiện lại
            detections = detect_objects(frame)
            best_match = find_best_match(detections, query)
            object_bbox = best_match[['xmin', 'ymin', 'xmax', 'ymax']].values
            tracker = track_object(frame, object_bbox)

        # Hiển thị frame
        cv2.imshow("Tracking", frame)
        
        # Dừng nếu nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ví dụ sử dụng
query = "white car"  # Câu ngôn ngữ để tìm đối tượng
video_path = "data_ext/test.mp4"  # Đường dẫn đến video cần theo dõi
track_object_in_video(video_path, query)
