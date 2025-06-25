import cv2
from ultralytics import YOLO
from rapidocr import RapidOCR
import numpy as np
import time

FRAMES_TO_PERSIST = 7
MIN_AREA_FOR_VEHICLE = 5000 
MAX_AREA_FOR_VEHICLE = 150000 
ASPECT_RATIO_MIN = 0.5 
ASPECT_RATIO_MAX = 3.0
MOVEMENT_DETECTED_PERSISTENCE = 10
YOLO_TRIGGER_THRESHOLD = 8

engine = RapidOCR()

def get_camera_ip():
    ip = input("Enter Camera IP (e.g., 192.168.0.2): ").strip() or "192.168.0.2"
    username = input("Enter Camera Username (default: admin): ").strip() or "admin"
    password = input("Enter Camera Password (default: admin1234): ").strip() or "admin1234"
    stream_type = input("Enter Stream Type (0 for Main, 1 for Sub): ").strip() or "1"
    
    rtsp_url = f"rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype={stream_type}"
    return rtsp_url

use_camera = input("Use camera? (y/n): ").lower().strip() == 'y'
if use_camera:
    video_source = get_camera_ip()
else:
    video_source = './test.mp4'

cap = cv2.VideoCapture(video_source)
first_frame = None
next_frame = None
font = cv2.FONT_HERSHEY_SIMPLEX
delay_counter = 0
movement_persistent_counter = 0
frame_id = 0

prev_time = time.time()
fps = 0
fps_counter = 0
fps_update_interval = 10

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

roi_y_factor = 0.69

def preprocess_plate_image(plate_img):
    sharpened = cv2.addWeighted(plate_img, 1.1, plate_img, 0, 0)
    enhanced = cv2.detailEnhance(sharpened, sigma_s=10, sigma_r=0.15)
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 15)
    gray_denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    return gray_denoised

while True:
    start_time = time.time()
    vehicle_detected = False
    
    ret, frame = cap.read()
    if not ret:
        if video_source == './test.mp4':
            cap = cv2.VideoCapture(video_source)
            ret, frame = cap.read()
            if not ret:
                break
        else:
            break

    frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))

    roi_y = int(frame.shape[0] * roi_y_factor)
    roi = frame[roi_y:, :]

    fg_mask = bg_subtractor.apply(roi)
    
    # Noise removal
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        area = cv2.contourArea(c)
        if MIN_AREA_FOR_VEHICLE < area < MAX_AREA_FOR_VEHICLE:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h

            if ASPECT_RATIO_MIN < aspect_ratio < ASPECT_RATIO_MAX:
                vehicle_detected = True
                cv2.rectangle(frame, (x, y + roi_y), (x + w, y + h + roi_y), (0, 255, 0), 2)
                cv2.putText(frame, f"Vehicle: {area:.0f}", (x, y + roi_y - 5), font, 0.5, (0, 255, 0), 2)

    if vehicle_detected:
        movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE
        # cv2.imwrite(f"breach_alert_frame_{frame_id + 1}.png", frame)
        if movement_persistent_counter >= YOLO_TRIGGER_THRESHOLD:
            cv2.imwrite(f"LPR_frame_{frame_id + 1}.png", frame)

            if 'model' not in locals():
                model = YOLO("./LPR-YOLO-best.pt")
                model.to("cpu")
                model.fuse()
            
            results = model(frame, imgsz=(1280, 736))
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    box_coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, box_coords)
                    conf = float(box.conf[0])
                    if conf > 0.55:
                        plate_img = frame[y1:y2, x1:x2]
                        if plate_img.size > 0 and plate_img.shape[0] > 0 and plate_img.shape[1] > 0:
                            processed_plate = preprocess_plate_image(plate_img)
                            result = engine(processed_plate)
                            print(result)

    fps_counter += 1
    if fps_counter >= fps_update_interval:
        current_time = time.time()
        fps = fps_counter / (current_time - prev_time)
        prev_time = current_time
        fps_counter = 0

    status_text = f"Vehicle Detected {movement_persistent_counter}" if movement_persistent_counter > 0 else "No Vehicle Detected"
    movement_persistent_counter = max(0, movement_persistent_counter - 1)

    cv2.putText(frame, status_text, (10, 30), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), font, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.line(frame, (0, roi_y), (frame.shape[1], roi_y), (255, 0, 0), 2)

    fg_mask_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    
    fg_mask_resized = cv2.resize(fg_mask_color, (frame.shape[1], frame.shape[0]))
    frame_resized = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
    
    combined_view = np.hstack((fg_mask_resized, frame_resized))
    cv2.imshow("Vehicle Detection", combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    process_time = time.time() - start_time
    if process_time < 0.03:
        time.sleep(0.03 - process_time)

cap.release()
cv2.destroyAllWindows()