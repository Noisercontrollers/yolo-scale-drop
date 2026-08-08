import time
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
  start_time = time.time()
  model = YOLO('ultralytics/cfg/models/26/yolo26n.yaml')
  model.load('weights/yolo26n.pt')  #注释则不加载
  results = model.train(
    data='VisDrone/VisDrone.yaml',  #数据集配置文件的路径
    epochs=100,  #训练轮次总数
    batch=32,  #批量大小，即单次输入多少图片训练
    imgsz=640,  #训练图像尺寸
    workers=8,  #加载数据的工作线程数
    device= 0,  #指定训练的计算设备，无nvidia显卡则改为 'cpu'
    optimizer='MuSGD',  #训练使用优化器，可选 auto,SGD,Adam,AdamW 等
    amp= True,  #True 或者 False, 解释为：自动混合精度(AMP) 训练
    cache='disk'  # True 在内存中缓存数据集图像，
  )
  elapsed_time = time.time() - start_time
  hours = int(elapsed_time // 3600)
  minutes = int((elapsed_time % 3600) // 60)
  print(f'程序总运行时间: {hours}小时{minutes}分钟')
