
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from nets.core import yolo_body
from keras.layers import Input
from yolo import YOLO
import tensorflow as tf
import numpy

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
yolo = YOLO()


def get_img_from_camera_net(folder_path):
    # cap = cv2.VideoCapture("https://vdept.bdstatic.com/46435a67676150353843484b347a7038/485873744732716d/e41b9df78efa0e236672b49b3bbcf8bf7ae714191b59e35a9eecb045f776a7d81e3d0a6877b5e2d2e26ab6ff47bf5506.mp4")#获取摄像机
    # cap = cv2.VideoCapture("http://192.168.0.101:4747/video")#获取摄像机\
    cap = cv2.VideoCapture(r"C:\Users\Administrator\Desktop\test1.mp4")  # 获取摄像机\
    htimes = 0
    while True:
        ret, frame = cap.read()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # cv2.imshow("capture", frame2)
        r_image, htimes = yolo.detect_image(image, htimes)
        img = cv2.cvtColor(numpy.asarray(r_image), cv2.COLOR_RGB2BGR)
        cv2.imshow("test", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # cv2.imwrite(folder_path + str(i) + '.jpg', frame)# 存储为图像

    cap.release()
    cv2.destroyAllWindows()


# 测试
if __name__ == '__main__':
    folder_path = 'D:\\Anacon'

    get_img_from_camera_net(folder_path)

# if __name__ == '__main__':
#     #folder_path = 'D:\\Anacon\\'
#     get_img_from_camera_net(1)