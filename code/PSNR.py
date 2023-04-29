import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
def calculate_video_psnr(video_path):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not video.isOpened():
        print("无法打开视频文件")
        return
    # 获取视频信息
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    # 准备用于计算平均PSNR的变量
    total_psnr = 0.0
    # 读取视频帧并计算PSNR
    prev_frame = None
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # 将帧转换为灰度图像
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            # 计算当前帧与上一帧之间的PSNR
            psnr = cv2.PSNR(prev_frame, frame_gray)
            total_psnr += psnr
        prev_frame = frame_gray
    # 关闭视频文件
    video.release()
    # 计算平均PSNR
    avg_psnr = total_psnr / (frame_count - 1)
    return avg_psnr
psnr_1 = calculate_video_psnr('../拍摄视频/left.mp4')
print("视频1平均PSNR：", psnr_1)
psnr_2 = calculate_video_psnr('../拍摄视频/right.mp4')
print("视频2平均PSNR：", psnr_2)
psnr_3 = calculate_video_psnr('../result/out1.mp4')
print("视频3平均PSNR：", psnr_3)
# 数据
x = ['left', 'right','end']
y = [psnr_1, psnr_2 ,psnr_3]
# 绘制柱状图
plt.bar(x, y)
# 在柱形上标注数值
for i in range(len(x)):
    plt.text(x[i], y[i], str(y[i]), ha='center', va='bottom')
# 设置标题和轴标签
plt.title('PSNR')
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim(0, 100)
# 显示图形
plt.show()
