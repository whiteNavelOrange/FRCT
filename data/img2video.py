import cv2
import os
# 其它格式的图片也可以
img_array = []
path = "/home/sgitai/FRCT/real_data/train/bimanual_handover/all_variations/episodes/episode2/"  # 图片文件路径
img_path = os.path.join(path, 'front_rgb')
filelist = os.listdir(img_path)  # 获取该目录下的所有文件名
filelist.sort()
for filename in filelist:
    #挨个读取图片
    img = cv2.imread(os.path.join(img_path, filename))
    #获取图片高，宽，通道数信息
    height, width, layers = img.shape
    #设置尺寸
    width = 1240
    img = img[:, :1240, :]
    # 设置文字相关参数
    text = "Handover" # 要添加的文本
    position = (520, 1000) # 文本在图片上的位置
    font = cv2.FONT_HERSHEY_SIMPLEX # 字体类型
    font_scale = 2 # 字体大小
    color = (0, 0, 255) # 文本颜色，这里使用蓝色
    thickness = 4 # 文本线条的粗细
    # 在图片上添加文字
    cv2.putText(img, text, position, font, font_scale, color, thickness)
    # 显示图片

    size = (width, height)
    #将图片添加到一个大“数组中”
    img_array.append(img)
print("this is ok")
# avi：视频类型，mp4也可以
# cv2.VideoWriter_fourcc(*'DIVX')：编码格式，不同的编码格式有不同的视频存储类型
# fps：视频帧率
# size:视频中图片大小
fps=30
# img_array.sort()
videopath=path + 'handover.mp4'#图片保存地址及格式
out1 = cv2.VideoWriter(videopath,cv2.VideoWriter_fourcc(*'mp4v'),fps, size)
for i in range(len(img_array)):
    #写成视频操作
    out1.write(img_array[i])
out1.release()
print("all is ok")

