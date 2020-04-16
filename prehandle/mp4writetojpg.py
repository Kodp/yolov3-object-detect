import cv2
import matplotlib.pyplot as plt

from PIL import Image

import numpy


def get_img_from_camera_net(folder_path):
    # cap = cv2.VideoCapture(r"C:/log/test.mp4")#获取摄像机
    cap = cv2.VideoCapture(r"C:/Users/P2912/Desktop/测试/test000.mp4")  # 获取摄像机
    i = 0
    while True:
        i += 1
        ret, frame = cap.read()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.save("C:/Users/P2912/Desktop/测试/mp4img/" + str(i) + ".jpg", quality=95)
        print(i)
        # cv2.imshow("capture", frame2)
        # r_image,htimes = yolo.detect_image(image,htimes)
        # img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
        # cv2.imshow("test",img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # cv2.imwrite(folder_path + str(i) + '.jpg', frame)# 存储为图像

    cap.release()
    cv2.destroyAllWindows()


# 测试
if __name__ == '__main__':
    folder_path = 'D:\\Anacon'
    get_img_from_camera_net(folder_path)
