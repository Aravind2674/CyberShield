import cv2
import torch
import numpy as np
import time
from deepface import DeepFace
import easyocr

class VideoPipeline:
    def __init__(self):
        # Load the YOLOv5 model from PyTorch Hub
        print("Loading YOLOv5 model...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device)
        self.model.eval()
        print(f"YOLOv5 model loaded on {self.device}")

        # Initialize EasyOCR for Number Plates
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("EasyOCR loaded")

        # Load Haar Cascade for quick face detection before passing to DeepFace
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # COCO labels for categories we care about
        self.target_classes = {
            0: 'person',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

    def process_frame(self, frame, analytics_state):
        # Convert BGR (OpenCV) to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = self.model([img_rgb], size=640)
        
        # Parse results
        detections = results.xyxy[0].cpu().numpy() # xmin, ymin, xmax, ymax, confidence, class
        
        current_frame_people = 0
        current_frame_vehicles = 0

        for *box, conf, cls in detections:
            cls = int(cls)
            if cls in self.target_classes and conf > 0.4:
                label = self.target_classes[cls]
                x1, y1, x2, y2 = map(int, box)
                
                # Draw Box
                if label == 'person':
                    color = (0, 255, 0) # Green for people
                    current_frame_people += 1
                else:
                    color = (0, 0, 255) # Red for vehicles
                    current_frame_vehicles += 1
                    
                    if label in analytics_state['vehicle_types']:
                        analytics_state['vehicle_types'][label] += 1
                    
                    # Number Plate Recognition (Improved Accuracy)
                    # We only process if the vehicle bounding box is decently sized to ensure readability
                    if conf > 0.5 and (y2 - y1) > 60 and (x2 - x1) > 60:
                        # Try only processing the bottom 40% of the vehicle to isolate the plate
                        h = y2 - y1
                        plate_y1 = int(y1 + (h * 0.6))
                        vehicle_crop = frame[plate_y1:y2, x1:x2] # Use BGR for cv2 processing
                        
                        try:
                            # Preprocess the crop to improve OCR accuracy
                            gray_crop = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
                            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            enhanced_crop = clahe.apply(gray_crop)
                            
                            results = self.reader.readtext(enhanced_crop)
                            for (bbox, text, prob) in results:
                                # Clean the text
                                clean_text = "".join(e for e in text if e.isalnum()).upper()
                                
                                if prob > 0.45 and len(clean_text) >= 5:
                                    # Deduplication: check if we recently saw this exact plate
                                    is_duplicate = False
                                    for log in analytics_state['event_logs'][:15]:
                                        if log['type'] == 'ANPR Match' and clean_text in log['detail']:
                                            is_duplicate = True
                                            break
                                            
                                    if not is_duplicate:
                                        analytics_state['plates_detected'] += 1
                                        
                                        # Add to searchable event logs
                                        timestamp = time.strftime("%H:%M:%S")
                                        analytics_state['event_logs'].insert(0, {
                                            "time": timestamp,
                                            "type": "ANPR Match",
                                            "detail": f"Plate '{clean_text}' recognized on {label}"
                                        })
                                        
                                    cv2.putText(frame, clean_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        except Exception as e:
                            pass
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update global state counts (naive approach for a single frame, 
        # normally you'd use a tracker like SORT/DeepSORT for unique counts)
        analytics_state['vehicle_count'] = current_frame_vehicles
        analytics_state['people_count'] = current_frame_people
        
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
                    analysis = DeepFace.analyze(face_roi, actions=['gender', 'age'], enforce_detection=False)
                    gender = analysis[0]['dominant_gender']
                    age = analysis[0]['age']
                    
                    if gender in analytics_state['gender_stats']:
                        analytics_state['gender_stats'][gender] += 1
                        
                    # Add to searchable event logs
                    timestamp = time.strftime("%H:%M:%S")
                    analytics_state['event_logs'].insert(0, {
                        "time": timestamp,
                        "type": "Face Analytics",
                        "detail": f"Recognized {gender}, ~{age}yo"
                    })
                    
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
