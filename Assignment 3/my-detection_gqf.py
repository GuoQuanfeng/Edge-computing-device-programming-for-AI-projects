# 导入jetson.inference和jetson.utils两个模块。
import jetson.inference
import jetson.utils

# 初始化对象检测网络，加载预训练好的"ssd-mobilenet-v2"模型，并设置检测阈值为0.5。
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# 从磁盘加载一张名称为'image.png'的图像文件。
img = jetson.utils.loadImage('image.png')

# 使用之前初始化好的对象检测网络对加载的图像进行检测，得到检测结果列表。
detections = net.Detect(img)

# 设定我们感兴趣的类别ID为1（比如人或者汽车等，具体的类别取决于模型被训练识别的对象类别）。
class_of_interest = 1

# 遍历所有检测到的对象。
for detection in detections:
    # 判断每一个检测到的对象是否为我们感兴趣的类别ID。
    if detection.ClassID == class_of_interest:
        # 如果是，则打印出该对象的相关信息：
        print("-- ClassID: {}".format(detection.ClassID))  # 类别ID
        print("-- Confidence: {:.6f}".format(detection.Confidence))  # 检测置信度
        print("-- Left: {}".format(detection.Left))  # 边界框左边界的位置
        print("-- Top: {}".format(detection.Top))  # 边界框顶部边界的位置
        print("-- Right: {}".format(detection.Right))  # 边界框右边界的位置
        print("-- Bottom: {}".format(detection.Bottom))  # 边界框下边界的位置
        print("-- Width: {:.3f}".format(detection.Width))  # 边界框的宽度
        print("-- Height: {:.3f}".format(detection.Height))  # 边界框的高度
        print("-- Area: {}".format(detection.Area))  # 边界框覆盖的面积
        print("-- Center: ({:.3f}, {:.3f})".format(detection.Center[0], detection.Center[1]))  # 边界框的中心点坐标
        print()  # 在控制台输出一个空行，用于分隔不同的检测结果。

# 定义将要保存的带有标注框的输出图像的路径和文件名。
output_image_path = 'saved_image.png'

# 使用jetson.utils模块的saveImage函数将经过标注（如果有的话）的图像保存到磁盘上。
jetson.utils.saveImage(output_image_path, img)

# 至此，图片已经被处理并且存储到了指定的路径，程序可以结束。
