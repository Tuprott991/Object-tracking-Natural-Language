import gradio as gr
import cv2
import torch
import numpy as np
from object_tracking_clip import match_object_with_query, initialize_models
import clip
from fastapi.exceptions import HTTPException


# Initialize models and other necessary components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_clip, processor_clip, model_yolo, tracker, class_names = initialize_models(device)
conf_threshold = 0.55

# Khởi tạo màu sắc cho bounding box
colors = np.random.randint(0, 255, size=(len(class_names), 3))

def process_video(video_file, query_text, mode):
    try:
        # Check if video_file is a string (file path) or a file-like object
        video_path = video_file.name if hasattr(video_file, 'name') else video_file
        cap = cv2.VideoCapture(video_path)
        
        
        output_frames = []
        saved_image_count = 0
        tracked_object_id = None

        if mode == "full_output":
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            output_video_path = "output_video.mp4"  # Đường dẫn để lưu video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Encode the query text
        text = clip.tokenize([query_text]).to(device)
        query_vector = model_clip.encode_text(text).cpu().detach().numpy().astype(np.float32)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model_yolo(frame)
            tracked_bbox, first_frame = match_object_with_query(query_vector, results, frame, first_frame=True)

            # Validate tracked_bbox
            if tracked_bbox is not None and isinstance(tracked_bbox, (list, tuple)) and len(tracked_bbox) == 4:
                x1, y1, x2, y2 = tracked_bbox
                if all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]) and x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
                    tracked_object_id = 1  # Track a single object
                    print(f"Object detected, ID: {tracked_object_id}")

                    cropped_obj = frame[int(y1):int(y2), int(x1):int(x2)]
                    if cropped_obj.size > 0:  # Ensure bounding box is not empty
                        save_path = f"tracked_images/track_{saved_image_count}.jpg"
                        cv2.imwrite(save_path, cropped_obj)
                        print(f"Saved image: {save_path}")
                        saved_image_count += 1

                    detect = [[[x1, y1, x2 - x1, y2 - y1], conf_threshold, tracked_object_id]]
                    tracks = tracker.update_tracks(detect, frame=frame)

                    for track in tracks:
                        if track.is_confirmed():
                            ltrb = track.to_ltrb()
                            x1, y1, x2, y2 = map(int, ltrb)
                            color = [73, 248, 255]
                            B, G, R = map(int, color)

                            label = f"Tracked-{tracked_object_id}"
                            print(label)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
                            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                            output_frames.append(frame)
                else:
                    print("Invalid bounding box coordinates.")
            else:
                print("No valid bounding box detected.")

            # If mode is streaming, yield each frame
            if mode == "streaming":
                yield frame

        cap.release()

        # If mode is full_output, create a video from the frames
        if mode == "full_output":
            for frame in output_frames:
                out.write(frame)

            out.release()
            
            with open(output_video_path, "rb") as video_file:
                out_video = out_video.read()
                return out_video

    except ConnectionResetError:
        print("Client disconnected.")
    except HTTPException as e:
        print(f"HTTP Exception: {e}")

iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Textbox(label="Enter Query Text"),
        gr.Dropdown(choices=["streaming", "full_output"], label="Mode")
    ],
    outputs=gr.Video(label="Output Video"),
    title="Object Tracking with Language Query",
    description="Upload a video and enter a query to track objects based on the language prompt."
)

iface.queue()  # Bật hàng đợi để hỗ trợ yield
iface.launch()