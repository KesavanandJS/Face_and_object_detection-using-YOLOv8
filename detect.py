import cv2
from ultralytics import YOLO

# Load YOLOv8 model for object detection
model = YOLO("yolov8n.pt")  # You can also use larger versions like yolov8s.pt, yolov8m.pt

# Load Haar Cascade for face detection (if specific face detection is needed)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load age detection model
age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # YOLOv8 object detection
    results = model(frame)  # Perform object detection
    
    # Extract boxes, scores, and class IDs from YOLOv8 results
    boxes = results[0].boxes  # The boxes detected in the image
    
    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Set detection mode based on detections
    detection_mode = "Idle"
    if len(faces) and len(boxes)> 0:
        detection_mode = "Detecting "
    

    # Display detection mode status
    cv2.putText(frame, f"Status: {detection_mode}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # YOLOv8 object detection results (show bounding boxes and labels for all objects)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract box coordinates
        conf = box.conf[0]  # Confidence score
        class_id = int(box.cls[0])  # Class ID
        label = model.names[class_id]

        # Draw bounding box and display label (green box for all objects)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_with_conf = f"{label} {conf * 100:.2f}%"
        cv2.putText(frame, label_with_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # If the detected object is a person, try to detect and display age
        if label == 'person':
            age_label = ""
            for (fx, fy, fw, fh) in faces:
                # Check if the face is within the detected person's bounding box
                if x1 < fx < x2 and y1 < fy < y2:
                    # Age detection
                    face_blob = cv2.dnn.blobFromImage(frame[fy:fy+fh, fx:fx+fw], 1.0, (227, 227),
                                                      (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                    age_net.setInput(face_blob)
                    age_preds = age_net.forward()
                    age = AGE_LIST[age_preds[0].argmax()]
                    age_label = f" | Age: {age}"
                    break

            # Append age information if available
            label_with_conf += age_label
            cv2.putText(frame, label_with_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("YOLOv8 Object and Face Detection with Age Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
