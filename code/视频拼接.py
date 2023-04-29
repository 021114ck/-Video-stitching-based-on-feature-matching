import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
# 读取视频
video1 = cv2.VideoCapture('../拍摄视频/right.mp4')
video2 = cv2.VideoCapture('../拍摄视频/left.mp4')
# 视频参数
width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video1.get(cv2.CAP_PROP_FPS))
# 输出视频
out = cv2.VideoWriter('../result/out1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))
# 参数 sift特征匹配
M = None
sift = cv2.SIFT_create()
i = 0
good_matches=[]
# 记录拼接开始时间
start_time = time.time()
while (video1.isOpened() and video2.isOpened()):
   try:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        if ((i) % 130 == 0):
            # 计算视角变换矩阵，然后后面的帧都用这个矩阵
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            kps1, features1 = sift.detectAndCompute(gray1, None)
            kps11 = np.float32([kp.pt for kp in kps1])  # 每一行代表一个特征点的坐标
            kps2, features2 = sift.detectAndCompute(gray2, None)
            kps22 = np.float32([kp.pt for kp in kps2])
            bf = cv2.BFMatcher()
            # 使用KNN检测来自两张图片的sift特征匹配对，K=2
            matches = bf.knnMatch(features1, features2, 2)
            good_matches = []
            for m in matches:
                # 当最近距离跟次近距离的比值小于0.75时，保留此匹配对
                if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                    # 存储两个点在featuresA, featuresB中的索引值
                    good_matches.append((m[0].trainIdx, m[0].queryIdx))
                # 当筛选后的匹配对大于4时，计算视角变换矩阵
            if len(good_matches) > 4:
                # 获取匹配对的点坐标
                ptsA = np.float32([kps11[i] for (_, i) in good_matches])
                ptsB = np.float32([kps22[i] for (i, _) in good_matches])
                # 计算视角变换矩阵
                M, status= cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5)
                # M是3x3视角变换矩阵
        result = cv2.warpPerspective(frame1, M, (frame1.shape[1] + frame2.shape[1], frame2.shape[0]))
        # 将图片frame2传入result图片最左端
        result[0:frame2.shape[0], 0:frame2.shape[1]] = frame2
        # 融合图像
        print(i)
        i += 1
        out.write(result)
        result = cv2.resize(result, dsize=(500, 500), fx=1, fy=1)
        cv2.imshow('result', result)
        cv2.waitKey(1)
   except Exception as error:
       print(error)
       break
# 记录程序结束时间
end_time = time.time()
# 计算程序运行时间（秒）
run_time = end_time - start_time
print("处理每一帧的时间", run_time/i, "秒")
video1.release()
video2.release()
cv2.destroyAllWindows()
