from django.contrib.admin.templatetags.admin_list import results
from ultralytics import YOLO
model = YOLO("yolov8n.yaml")
results = model.train (data="data.yaml", epochs=70) #train model
