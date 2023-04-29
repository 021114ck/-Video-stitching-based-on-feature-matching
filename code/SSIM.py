import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import skvideo.io
from skimage.metrics import structural_similarity as ssim
def calculate_ssim(video_path):
    # 读取视频
    video = skvideo.io.vread(video_path)
    # 将视频帧转换为灰度图像
    gray_frames = [frame.mean(axis=2) for frame in video]
    # 初始化SSIM总和和帧数
    total_ssim = 0.0
    num_frames = len(gray_frames)
    # 计算每一帧与前一帧之间的SSIM，并将其累加到总和中
    for i in range(1, num_frames):
        ssim_score = ssim(gray_frames[i - 1], gray_frames[i])
        total_ssim += ssim_score
        print(i)
    # 计算平均SSIM
    avg_ssim = total_ssim / (num_frames - 1)
    return avg_ssim
# 读取视频
ssim_1 = calculate_ssim('left.mp4')
print("视频1平均SSIM：", ssim_1)
ssim_2 = calculate_ssim('right.mp4')
print("视频2平均SSIM：", ssim_2)
ssim_3= calculate_ssim('out1.mp4')
print("视频2平均SSIM：", ssim_3)
# 数据
x = ['left', 'right','end']
y = [ssim_1, ssim_2,ssim_3]
# 绘制柱状图
plt.bar(x, y)
# 在柱形上标注数值
for i in range(len(x)):
    plt.text(x[i], y[i], str(y[i]), ha='center', va='bottom')
# 设置标题和轴标签
plt.title('SSIM')
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim(0, 1)
# 显示图形
plt.show()