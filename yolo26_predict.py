from ultralytics import YOLO
# 加载预训练的 YOLO26n 模型
# model = YOLO('weights/yolo26n.pt')
model = YOLO('runs/detect/100-MuSGD/weights/best.pt')
source = 'ultralytics/assets/bus.jpg' #更改为自己的图片路径
results = model(source)
results[0].show()
# 运行推理，并附加参数
# model.predict(source, save=True)
