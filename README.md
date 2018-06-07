# YOLOv2-Tensorflow<br>
## 声明：<br>
更详细的代码解读[Tensorflow实现YOLO2](https://zhuanlan.zhihu.com/p/36902889).<br>
欢迎关注[我的知乎](https://www.zhihu.com/people/chensicheng/posts).<br><br>

## 运行环境：<br>
Python3 + Tensorflow1.5 + OpenCV-python3.3.1 + Numpy1.13<br>
windows和ubuntu环境都可以<br><br>

## 准备工作：<br>
请在[yolo2检测模型](https://pan.baidu.com/s/1ZeT5HerjQxyUZ_L9d3X52w)下载模型，并放到yolo2_model文件夹下<br><br>

## 文件说明：<br>
1、model_darknet19.py：yolo2网络模型——darknet19<br>
2、decode.py：解码darknet19网络得到的参数<br>
3、utils.py：功能函数，包含：预处理输入图片、筛选边界框NMS、绘制筛选后的边界框<br>
4、config.py：配置文件，包含anchor尺寸、coco数据集的80个classes类别名称<br>
5、Main.py：YOLO_v2主函数，对应程序有三个步骤：<br>
（1）输入图片进入darknet19网络得到特征图，并进行解码得到：xmin xmax表示的边界框、置信度、类别概率<br>
（2）筛选解码后的回归边界框——NMS<br>
（3）绘制筛选后的边界框<br>
6、Loss.py：Yolo_v2 Loss损失函数（train时候用，预测时候没有调用此程序）<br>
（1）IOU值最大的那个anchor与ground truth匹配，对应的预测框用来预测这个ground truth:计算xywh、置信度c(目标值为1)、类别概率p误差。<br>
（2）IOU小于某阈值的anchor对应的预测框：只计算置信度c(目标值为0)误差。<br>
（3）剩下IOU大于某阈值但不是max的anchor对应的预测框：丢弃，不计算任何误差。<br>
7、yolo2_data文件夹：包含待检测输入图片car.jpg、检测后的输出图片detection.jpg、coco数据集80个类别名称coco_classes.txt<br><br>

## 运行Main.py即可得到效果图：<br>
1、car.jpg：输入的待检测图片<br><br>
![image](https://github.com/KOD-Chen/YOLOv2-Tensorflow/blob/master/yolo2_data/car.jpg)<br>
2、detected.jpg：检测结果可视化<br><br>
![image](https://github.com/KOD-Chen/YOLOv2-Tensorflow/blob/master/yolo2_data/detection.jpg)<br>
