# import cv2
# cap = cv2.VideoCapture(1)  # 或替换为RTSP地址
# if not cap.isOpened():
#     print("无法打开相机！")
# else:
#     ret, frame = cap.read()
#     if ret:
#         cv2.imshow('Test', frame)
#         cv2.waitKey(0)
#     else:
#         print("读取帧失败！")
# cap.release()
import cv2

cap = cv2.VideoCapture(1)  # 调用摄像头‘0’一般是打开电脑自带摄像头，‘1’是打开外部摄像头（只有一个摄像头的情况）

if False == cap.isOpened():
    print(0)
else:
    print(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3200)  # 设置图像宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2400)  # 设置图像高度
cap.set(cv2.CAP_PROP_FPS, 20)  # 设置帧率
# 显示图像
while True:
    ret, frame = cap.read()
    # print(ret)  #
    ########图像不处理的情况
    frame_1 = cv2.resize(frame, (640, 512))
    cv2.imshow("frame", frame_1)

    input = cv2.waitKey(1)
    if input == ord('q'):
        break

cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 销毁窗口