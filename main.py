from ultralytics import YOLO

# 加载模型
model = YOLO("D:/runs/detect/train21/weights/best.pt")  # 替换为你的 best.pt 路径

# 用图片验证
image_path = "D:/learning/python-workspace/YOLO/dataset/img/val/PixPin_2025-03-25_15-40-43.png"  # 替换为你的图片路径
results = model.predict(image_path, save=True, conf=0.25, iou=0.45)

# 打印检测结果
for result in results:
    print("检测到的目标:")
    for box in result.boxes:
        class_id = int(box.cls)  # 类别 ID
        label = result.names[class_id]  # 类别名称
        confidence = box.conf.item()  # 置信度
        coords = box.xywh[0].tolist()  # 边界框坐标 (x_center, y_center, width, height)
        print(f"类别: {label}, 置信度: {confidence:.2f}, 坐标: {coords}")