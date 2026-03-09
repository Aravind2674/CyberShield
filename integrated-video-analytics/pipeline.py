import cv2
import torch
import numpy as np
import time
from deepface import DeepFace
import easyocr
import supervision as sv
from database import log_event

class VideoPipeline:
    def __init__(self):
        # Load the YOLOv5 model from PyTorch Hub
        print("Loading YOLOv5 model...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device)
        self.model.eval()
        print(f"YOLOv5 model loaded on {self.device}")
        
        # Load dedicated Plate YOLO model (Ultralytics v8 fallback to standard repo approach if needed)
        try:
            from ultralytics import YOLO
            self.plate_model = YOLO("keremberke/yolov8m-license-plate") # Or download a specific .pt
            print("Loaded YOLO plate model")
        except:
            self.plate_model = None
            print("Could not load yolov8 plate model, will fallback to heuristic cropping")

        # Initialize EasyOCR for Number Plates
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("EasyOCR loaded")

        # Load Haar Cascade for quick face detection before passing to DeepFace
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.target_classes = {
            0: 'person',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Supervision initialization
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator()
        self.heat_annotator = sv.HeatMapAnnotator()
        
        # We define a polygon zone. In real use cases, this should be configurable.
        # For now, let's just make a generic zone covering the lower half of a 1080p frame, scaled properly.
        # We will initialize it with a dummy resolution (e.g., 1920x1080)
        self.zone_polygon = np.array([
            [0, 540],
            [1920, 540],
            [1920, 1080],
            [0, 1080]
        ])
        self.zone = sv.PolygonZone(polygon=self.zone_polygon, triggering_anchors=(sv.Position.CENTER,))
        self.zone_annotator = sv.PolygonZoneAnnotator(zone=self.zone, color=sv.Color.BLUE)
        self.tracked_vehicles = set()
        self.tracked_people = set()

    def process_frame(self, frame, analytics_state):
        # Convert BGR (OpenCV) to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = self.model([img_rgb], size=640)
        
        # Supervision parsing
        detections = sv.Detections.from_yolov5(results)
        
        # Filter detections for target classes
        target_ids = list(self.target_classes.keys())
        detections = detections[np.isin(detections.class_id, target_ids) & (detections.confidence > 0.4)]
        
        # Track
        detections = self.tracker.update_with_detections(detections)
        
        # Update zone counts
        zone_trigger = self.zone.trigger(detections=detections)
        
        # Draw Heatmap & Zone
        frame = self.heat_annotator.annotate(scene=frame, detections=detections)
        frame = self.zone_annotator.annotate(scene=frame)
        
        labels = []
        for box, mask, confidence, class_id, tracker_id, data in detections:
            class_name = self.target_classes[class_id]
            labels.append(f"#{tracker_id} {class_name} {confidence:.2f}")
            
            x1, y1, x2, y2 = map(int, box)
            
            if class_name == 'person':
                self.tracked_people.add(tracker_id)
            else:
                self.tracked_vehicles.add(tracker_id)
                if class_name in analytics_state['vehicle_types'] and tracker_id not in self.tracked_vehicles:
                    # Increment specific type only once per ID
                    analytics_state['vehicle_types'][class_name] += 1
                
                # Number Plate Recognition on Vehicles
                if confidence > 0.5 and (y2 - y1) > 60 and (x2 - x1) > 60:
                    # Random skip to avoid lag
                    if np.random.rand() < 0.2:
                        try:
                            vehicle_crop = frame[y1:y2, x1:x2]
                            plates_regions = []
                            
                            # Use dedicated plate model if available
                            if self.plate_model is not None:
                                plate_results = self.plate_model(vehicle_crop, verbose=False)
                                for p_det in plate_results[0].boxes:
                                    if p_det.conf > 0.4:
                                        px1, py1, px2, py2 = map(int, p_det.xyxy[0])
                                        plates_regions.append(vehicle_crop[py1:py2, px1:px2])
                            
                            # Fallback heuristics if model failed or missing
                            if not plates_regions:
                                h = y2 - y1
                                plate_y1 = int(y1 + (h * 0.6))
                                plates_regions.append(frame[plate_y1:y2, x1:x2])
                                
                            for p_crop in plates_regions:
                                gray_crop = cv2.cvtColor(p_crop, cv2.COLOR_BGR2GRAY)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            enhanced_crop = clahe.apply(gray_crop)
                            
                            ocr_results = self.reader.readtext(enhanced_crop)
                            for (bbox, text, prob) in ocr_results:
                                clean_text = "".join(e for e in text if e.isalnum()).upper()
                                
                                if prob > 0.45 and len(clean_text) >= 5:
                                    is_duplicate = False
                                    for log in analytics_state['event_logs'][:15]:
                                        if log['type'] == 'ANPR Match' and clean_text in log['detail']:
                                            is_duplicate = True
                                            break
                                            
                                    if not is_duplicate:
                                        analytics_state['plates_detected'] += 1
                                        timestamp = time.strftime("%H:%M:%S")
                                        
                                        detail = f"Plate '{clean_text}' on {class_name} #{tracker_id}"
                                        
                                        analytics_state['event_logs'].insert(0, {
                                            "time": timestamp,
                                            "type": "ANPR Match",
                                            "detail": detail
                                        })
                                        # Persist to database
                                        log_event("ANPR Match", detail)
                                        
                                    cv2.putText(frame, clean_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        except Exception as e:
                            pass
        
        # Annotate Boxes
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # Update global state counts properly with tracked IDs
        analytics_state['vehicle_count'] = len(self.tracked_vehicles)
        analytics_state['people_count'] = len(self.tracked_people)
        analytics_state['zone_count'] = int(zone_trigger.sum())
        
        # Face Detection & Gender Analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            analytics_state['faces_detected'] += 1
            
            # Extract face ROI for deeper analysis
            # To avoid huge lag in real-time, only sample occasionally or use threading
            if np.random.rand() < 0.05: # Sample 5% of frames for DeepFace
                try:
                    face_roi = img_rgb[y:y+h, x:x+w]
                    
                    # 1. Watchlist matching
                    match_name = None
                    try:
                        matches = DeepFace.find(face_roi, db_path='watchlist', enforce_detection=False, silent=True)
                        if len(matches) > 0 and len(matches[0]) > 0:
                            # Parse identity from path, assuming watchlist/Name.jpg
                            import os
                            identity_path = matches[0].iloc[0]['identity']
                            match_name = os.path.basename(identity_path).split('.')[0]
                    except Exception as e:
                        pass
                    
                    # 2. Demographic analysis
                    analysis = DeepFace.analyze(face_roi, actions=['gender', 'age'], enforce_detection=False, silent=True)
                    gender = analysis[0]['dominant_gender']
                    age = analysis[0]['age']
                    
                    if gender in analytics_state['gender_stats']:
                        analytics_state['gender_stats'][gender] += 1
                        
                    # Add to searchable event logs
                    timestamp = time.strftime("%H:%M:%S")
                    
                    if match_name:
                        detail = f"Match Found: {match_name} ({gender}, ~{age}yo)"
                        analytics_state['event_logs'].insert(0, {
                            "time": timestamp,
                            "type": "Watchlist Alert",
                            "detail": detail
                        })
                        log_event("Watchlist Alert", detail) # Persist to DB
                        
                        cv2.putText(frame, f"ALERT: {match_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        detail = f"Recognized {gender}, ~{age}yo"
                        analytics_state['event_logs'].insert(0, {
                            "time": timestamp,
                            "type": "Face Analytics",
                            "detail": detail
                        })
                        log_event("Face Analytics", detail) # Persist to DB
                        
                        cv2.putText(frame, f"{gender} {age}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        
                except Exception as e:
                    cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            else:
                cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
        # Keep event logs constrained to recent 50 to avoid memory leak
        if len(analytics_state['event_logs']) > 50:
            analytics_state['event_logs'] = analytics_state['event_logs'][:50]

        # Number Plate Recognition (Simplified Logic)
        # Assuming our vehicle bounding box could contain a plate, but for now
        # let's just randomly scan the lower half of vehicles if needed.
        # To avoid massive slowdown, tracking is usually employed.
        # We will increment plates_detected based on simulated log for now or if easyocr finds text.

        return frame
