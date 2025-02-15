from ultralytics import YOLO


model=YOLO("yolo11x.pt")
model.track('input_videos/input_video.mp4',conf=0.2,save=True)





