import cv2
from ultralytics import YOLO
import numpy as np
import time
import os
import subprocess
import threading
import json, re

MIN_AREA_FOR_VEHICLE = 7500 
MAX_AREA_FOR_VEHICLE = 35000 
ASPECT_RATIO_MIN = 0.25
ASPECT_RATIO_MAX = 1.75
MOVEMENT_DETECTED_PERSISTENCE = 32
YOLO_TRIGGER_THRESHOLD = 14
YOLO_CONFIDENCE_THRESHOLD = 0.15
MIN_PLATE_DETECTION_INTERVAL = 1
SAVE_BEST_CROPS_ONLY = True
MAX_CROPS_PER_VEHICLE = 1
VERIFIED_VEHICLE_CONFIDENCE = 0.4
MIN_TIME_BETWEEN_VEHICLES = 2.5

def run_listener():
    try:
        subprocess.run(["python", "D:\OnePlus\KPI_4\listener.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running listener.py: {e}")

def get_camera_ip():
    ip = input("Enter Camera IP (e.g., 192.168.0.2): ").strip() or "192.168.0.2"
    username = input("Enter Camera Username (default: admin): ").strip() or "admin"
    password = input("Enter Camera Password (default: admin1234): ").strip() or "admin1234"
    
    rtsp_url = f"rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype={stream_type}"
    return rtsp_url

def preprocess_plate_image(plate_img):
    sharpened = cv2.addWeighted(plate_img, 1.1, plate_img, 0, 0)
    enhanced = cv2.detailEnhance(sharpened, sigma_s=10, sigma_r=0.15)
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 15)
    return denoised

def get_ocr_result(crop_filename):
    try:
        subprocess.Popen(
            ["python", "./listener.py", crop_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return None
    except Exception as e:
        print(f"Error starting listener process: {e}")
        return None

os.makedirs("plate_crops", exist_ok=True)
os.makedirs("verified_plates", exist_ok=True)

use_camera = input("Use camera? (y/n): ").lower().strip() == 'y'
if use_camera:
    video_source = get_camera_ip()
else:
    video_source = './test.mp4'

cap = cv2.VideoCapture(video_source)
font = cv2.FONT_HERSHEY_SIMPLEX
movement_persistent_counter = 0
frame_id = 0
last_plate_detection_time = 0
vehicle_tracking_id = 0
current_vehicle_crops = 0
last_new_vehicle_time = 0
current_vehicle_verified = False
current_vehicle_plate = None
vehicle_plates = {}
vehicle_was_detected = False
listener_thread = None
last_movement_state = 0

prev_time = time.time()
fps = 0
fps_counter = 0
fps_update_interval = 10

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

roi_y_factor = 0.635

model = None
best_confidence = 0
best_plate_image = None

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

    if movement_persistent_counter == 0:
        fg_mask = bg_subtractor.apply(roi)

        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_time = time.time()

        enough_time_passed = (current_time - last_new_vehicle_time) > MIN_TIME_BETWEEN_VEHICLES
        
        for c in cnts:
            area = cv2.contourArea(c)
            if MIN_AREA_FOR_VEHICLE < area < MAX_AREA_FOR_VEHICLE:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h

                if ASPECT_RATIO_MIN < aspect_ratio < ASPECT_RATIO_MAX:
                    vehicle_detected = True
                    cv2.rectangle(frame, (x, y + roi_y), (x + w, y + h + roi_y), (0, 255, 0), 2)
                    cv2.putText(frame, f"Vehicle: {area:.0f}", (x, y + roi_y - 5), font, 0.5, (0, 255, 0), 2)

                    if enough_time_passed:
                        vehicle_tracking_id += 1
                        current_vehicle_crops = 0
                        best_confidence = 0
                        best_plate_image = None
                        current_vehicle_verified = False
                        current_vehicle_plate = None
                        last_new_vehicle_time = current_time
    else:
        vehicle_detected = True

    current_time = time.time()

    if vehicle_detected:
        vehicle_was_detected = True
        if movement_persistent_counter == 0:
            movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

        if (movement_persistent_counter >= YOLO_TRIGGER_THRESHOLD and 
            current_time - last_plate_detection_time > MIN_PLATE_DETECTION_INTERVAL and
            current_vehicle_crops < MAX_CROPS_PER_VEHICLE and
            not current_vehicle_verified):
            
            last_plate_detection_time = current_time

            if model is None:
                model = YOLO("./LPR-YOLO-best.pt")
                model.to("cpu")
                model.fuse()
            
            results = model(frame, imgsz=(1280, 736), conf=YOLO_CONFIDENCE_THRESHOLD)
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    box_coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, box_coords)
                    conf = float(box.conf[0])
                    
                    if conf > YOLO_CONFIDENCE_THRESHOLD:
                        plate_img = frame[y1:y2, x1:x2]
                        
                        if plate_img.size > 0 and plate_img.shape[0] > 0 and plate_img.shape[1] > 0:
                            processed_plate = preprocess_plate_image(plate_img)
                            
                            if SAVE_BEST_CROPS_ONLY:
                                if conf > best_confidence:
                                    best_confidence = conf
                                    best_plate_image = processed_plate.copy()
  
                                    crop_filename = f"plate_crops/plate_{vehicle_tracking_id}_{current_vehicle_crops}_{conf:.2f}.jpg"
                                    cv2.imwrite(crop_filename, processed_plate)

                                    ocr_result = get_ocr_result(crop_filename)
                                    
                                    if ocr_result and ocr_result['confidence'] >= VERIFIED_VEHICLE_CONFIDENCE:
                                        verified_filename = f"verified_plates/plate_{vehicle_tracking_id}_{ocr_result['plate_text']}_{ocr_result['confidence']:.2f}.jpg"
                                        cv2.imwrite(verified_filename, processed_plate)

                                        current_vehicle_verified = True
                                        current_vehicle_plate = ocr_result['plate_text']
                                        vehicle_plates[vehicle_tracking_id] = current_vehicle_plate
                            else:
                                crop_filename = f"plate_crops/plate_{vehicle_tracking_id}_{current_vehicle_crops}_{conf:.2f}.jpg"
                                cv2.imwrite(crop_filename, processed_plate)
                                current_vehicle_crops += 1
    else:
        if vehicle_was_detected and movement_persistent_counter == 0:
            if listener_thread is None or not listener_thread.is_alive():
                listener_thread = threading.Thread(target=run_listener)
                listener_thread.start()
            vehicle_was_detected = False
            
        if movement_persistent_counter == 1 and best_plate_image is not None and best_confidence > 0:
            best_filename = f"plate_crops/best_plate_{vehicle_tracking_id}_{best_confidence:.2f}.jpg"
            cv2.imwrite(best_filename, best_plate_image)
            best_plate_image = None
            best_confidence = 0

    fps_counter += 1
    if fps_counter >= fps_update_interval:
        fps = fps_counter / (current_time - prev_time)
        prev_time = current_time
        fps_counter = 0

    status_text = f"Vehicle Motion Detected {movement_persistent_counter}" if movement_persistent_counter > 0 else "No Vehicle Motion Detected"
    movement_persistent_counter = max(0, movement_persistent_counter - 1)

    cv2.putText(frame, status_text, (10, 30), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), font, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"VEHICLE COUNT: {vehicle_tracking_id}", (10, 90), font, 0.75, (255, 0, 255), 2, cv2.LINE_AA)

    if current_vehicle_plate:
        cv2.putText(frame, f"Plate: {current_vehicle_plate}", (10, 120), font, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

    verification_status = "Verified" if current_vehicle_verified else "Unverified"
    cv2.putText(frame, verification_status, (10, 150), font, 0.75, (0, 255, 0) if current_vehicle_verified else (0, 0, 255), 2, cv2.LINE_AA)

    cv2.line(frame, (0, roi_y), (frame.shape[1], roi_y), (255, 0, 0), 2)

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    process_time = time.time() - start_time
    if process_time < 0.03:
        time.sleep(0.03 - process_time)

cap.release()
cv2.destroyAllWindows()