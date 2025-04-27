import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import clip
from query_processing import Translation
from PIL import Image

# Khởi tạo dịch câu truy vấn
translator = Translation()

# Khởi tạo CLIP backbone9
clip_backbone = "ViT-B/32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_clip, processor_clip = clip.load(clip_backbone, device=device)

# Config giá trị đầu vào
video_path = "data_ext/test.mp4"
output_video_path = "output_video.mp4"  # Đường dẫn để lưu video
conf_threshold = 0.5  # Threshold confidence

# Khởi tạo DeepSort
tracker = DeepSort(max_age=1)  # Sau 20 frame mất dấu thì loại object

# Khởi tạo YOLO
model_yolo = DetectMultiBackend(weights="weights/yolov9-c.pt", device=device, fuse=True)
model_yolo = AutoShape(model_yolo)

# Load classname từ file classes.names
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

# Khởi tạo màu sắc cho bounding box
colors = np.random.randint(0, 255, size=(len(class_names), 3))

# Khởi tạo VideoCapture để đọc từ file video
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Khởi tạo VideoWriter để lưu video đầu ra
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_t0 = True
query_vector = None
tracked_object_id = None  # ID của object được track
tracked_bbox = None       # BBox của object được track
saved_image_count = 0
main_label = None

def initialize_models(device):
    return model_clip, processor_clip, model_yolo, tracker, class_names
 
# Hàm so khớp object gần nhất với truy vấn
def match_object_with_query(query_vector, results, frame, first_frame=True):
    min_distance = float('inf')
    matched_bbox = None
    conf_threshold2 = 0.65

    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)

        if confidence < 0.2 or x2 <= x1 or y2 <= y1:
            continue

        cropped_obj = frame[y1:y2, x1:x2]
        if cropped_obj.size == 0:
            continue

        cropped_obj_pil = Image.fromarray(cropped_obj)
        obj_tensor = processor_clip(cropped_obj_pil).unsqueeze(0).to(device)
        obj_vector = model_clip.encode_image(obj_tensor).cpu().detach().numpy().astype(np.float32)

        distance = np.linalg.norm(query_vector - obj_vector)

        if distance < min_distance:
            min_distance = distance
            matched_bbox = [x1, y1, x2, y2]

    return matched_bbox, False  # Return False for first_frame flag


# Vòng lặp đọc frame

if __name__ == "__main__":
    text = input("Nhập câu truy vấn: ")
    text = clip.tokenize([text]).to(device)
    query_vector = model_clip.encode_text(text).cpu().detach().numpy().astype(np.float32)

    saved_image_count = 0
    first_time  = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolo(frame)
        tracked_bbox, first_frame = match_object_with_query(query_vector, results, frame, first_frame=True)

        # Phát hiện và lấy bounding box cho lần đầu tiên
        if tracked_bbox:
            tracked_object_id = 1  # Chỉ theo dõi một đối tượng duy nhất
            print(f"Đã phát hiện đối tượng, ID: {tracked_object_id}")
        else:
            first_frame = False  # Sau lần đầu tiên, không cần so khớp lại CLIP
        

        if tracked_object_id is not None and tracked_bbox:

            detect = []

            x1, y1, x2, y2 = tracked_bbox
            print(tracked_bbox)
            detect.append([[x1, y1, x2 - x1, y2 - y1], conf_threshold, tracked_object_id])

            tracks = tracker.update_tracks(detect, frame=frame)
            # if first_time:
            #     x1, y1, x2, y2 = tracked_bbox
            #     detect.append([[x1, y1, x2 - x1, y2 - y1], conf_threshold, tracked_object_id])
            #     first_time = False
            # else:
            #     for detect_object in results.pred[0]:
            #         label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4] #bbox là tỉ lệ

            #         #Ánh xạ lại số nguyên, tại output là số thực .
            #         x1, y1, x2, y2 = map(int, bbox)
            #         class_id = int(label)

            #         if confidence < conf_threshold: # Confidence bé hơn ngưỡng thì loại
            #             continue
            #         else: # Nếu cái class_id detect khác với cái tracking_class mong muốn hoặc confidence không đủ thì pass
            #             if class_id != main_label or confidence < conf_threshold:
            #                 continue
            #         detect.append([[x1, y1, x2 - x1, y2 - y1], conf_threshold, tracked_object_id])

            # Cập nhật theo dõi đối tượng duy nhất
            cropped_obj = frame[y1:y2, x1:x2]

            if cropped_obj.size > 0:  # Kiểm tra bounding box không rỗng
                save_path = f"tracked_images/track_{saved_image_count}.jpg"
                cv2.imwrite(save_path, cropped_obj)
                print(f"Saved image: {save_path}")
                saved_image_count += 1

        

            # tracked_ids = [track.track_id for track in tracks if track.is_confirmed()]
            # print(tracked_ids)
            # if str(tracked_object_id) not in tracked_ids:
            #     print("Đối tượng bị mất dấu, cần tìm lại...")
            #     tracked_bbox, first_frame = match_object_with_query(query_vector, results, frame, first_frame=True)
            #     if tracked_bbox:
            #         print(f"Tìm lại đối tượng, ID: {tracked_object_id}")
            #     else:
            #         tracked_object_id = None  # Nếu không tìm lại được, reset ID đối tượng
            #         print("Không tìm lại được đối tượng.")

            for track in tracks:
                if track.is_confirmed(): #and track.track_id == tracked_object_id:
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    color = [73,248,255]
                    B, G, R = map(int, color)

                    label = f"Tracked-{tracked_object_id}"
                    print(label)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                    cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



        # DeepSORT cập nhật tracking nếu đã có bounding box
        # detect = []
        # if tracked_bbox:
        #     x1, y1, x2, y2 = tracked_bbox
        #     detect.append([[x1, y1, x2 - x1, y2 - y1], conf_threshold, tracked_object_id])
        #     cropped_obj = frame[y1:y2, x1:x2]

        #     if cropped_obj.size > 0:  # Kiểm tra bounding box không rỗng
        #         save_path = f"tracked_images/track_{saved_image_count}.jpg"
        #         cv2.imwrite(save_path, cropped_obj)
        #         print(f"Saved image: {save_path}")
        #         saved_image_count += 1


        # tracks = tracker.update_tracks(detect, frame=frame)

        # tracked_ids = [track.track_id for track in tracks if track.is_confirmed()]

        # if tracked_object_id not in tracked_ids:
        #     results = model_yolo(frame)
            
        #     tracked_bbox = match_object_with_query(query_vector, results, frame)
        #     print(tracked_bbox)
        #     if tracked_bbox:
        #         tracked_object_id = f'{text}'  # Gán ID cố định để track object duy nhất\

        # Hiển thị và ghi video
        out.write(frame)
        resized_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Tracking test", resized_frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
