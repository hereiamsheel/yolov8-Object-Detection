import cv2
from ultralytics import YOLO

# Load the model
model = YOLO('../Yolo-Weights/yolov8n.pt')

# Run inference (set show=False so we can control the window ourselves)
results = model("test1.jpg")

annotated_frame = results[0].plot()
cv2.imshow("YOLOv8 Detection Results", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()