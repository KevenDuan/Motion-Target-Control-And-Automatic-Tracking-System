import cv2
cap = cv2.VideoCapture(0)
# 设置窗口大小
cap.set(3, 640)
cap.set(4, 480)
while (True):
    # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像
    hx, frame = cap.read()
    # 如果hx为Flase表示开启摄像头失败，那么就输出"read vido error"并退出程序
    if hx is False:
        # 打印报错
        print('read video error')
        # 退出程序
        exit(0)
    cv2.imshow('frame', frame)
    # 监测键盘输入是否为q，为q则退出程序
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
        break

cv2.destroyAllWindows()