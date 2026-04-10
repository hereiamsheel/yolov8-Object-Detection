from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0) #replace 0 with test-video as cap = cv2.VideoCapture('test1.mp4')
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam", 640, 480)

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ] 

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame. Check camera connection.")
        break

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (640, 480))
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:

            #Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1) , int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 3)
            w, h = x2-x1, y2- y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            #Confidence
            conf = math.ceil((box.conf[0] * 100))
            #Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf} %', (max(0,x1), max(35,y1)), scale=1, thickness=2)


    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()

''' These properties are predefined as part of the Boxes class in the Ultralytics YOLO engine. 
When you run a model, it automatically populates these specific attributes for every object it finds.
Think of the box object as a "packet" of info that always contains these specific fields. 
You don't have to define them; you just "read" them.'''

''' Property	Description	                            Unpacking Tip
    .xyxy	    Box corners (Top-Left, Bottom-Right)	box.xyxy[0]
    .conf	    Confidence score (0.0 to 1.0)	        box.conf[0]
    .cls	    Class index (e.g., 0, 1, 2)	            int(box.cls[0])
    .xywh	    Center point, Width, and Height	        box.xywh[0]
    .id	        Track ID (if using model.track)	        box.id[0]     '''
