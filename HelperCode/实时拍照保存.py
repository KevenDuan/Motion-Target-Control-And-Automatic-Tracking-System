# coding:utf-8
import cv2
cap = cv2.VideoCapture(0)
flag = cap.isOpened()
index = 1
while (flag):
    ret, frame = cap.read()
    cv2.imshow('1', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):  # 按下s键，进入下面的保存图片操作
        cv2.imwrite("/Users/duanhao/Desktop/position6/" + str(index) + ".jpg", frame)
        print("save" + str(index) + ".jpg successfuly!")
        print("-------------------------")
        index += 1
    elif k == ord('q'):  # 按下q键，程序退出
        break
cap.release() # 释放摄像头
cv2.destroyAllWindows()# 释放并销毁窗口
