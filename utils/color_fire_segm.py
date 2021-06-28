# https://blog.csdn.net/weixin_41987641/article/details/81812823
import cv2 as cv
import numpy as np

def contrast_brightness_demo(image, c, b):  #其中c为对比度，b为每个像素加上的值（调节亮度）
    blank = np.zeros(image.shape, image.dtype)   #创建一张与原图像大小及通道数都相同的黑色图像
    dst = cv.addWeighted(image, c, blank, 1-c, b) #c为加权值，b为每个像素所加的像素值
    ret, dst = cv.threshold(dst, 25, 255, cv.THRESH_BINARY)
    return dst

def get_color_seg(img):
    redThre = 105           #考虑用统计信息 进行自适应阈值
    saturationTh =  42

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    minValue = np.array(np.where(R <= G, np.where(G <= B, R, np.where(R <= B, R, B)), np.where(G <= B, G, B)))
    S = 1 - 3.0 * minValue / (R + G + B + 1)
    fireImg = np.array(np.where(R > redThre, np.where(R >= G, np.where(G >= B, np.where(S >= 0.2, np.where(S >= (255 - R)*saturationTh/redThre, 255, 0), 0), 0), 0), 0))
    gray_fireImg = np.zeros([fireImg.shape[0], fireImg.shape[1], 1], np.uint8)
    gray_fireImg[:, :, 0] = fireImg
    gray_fireImg = cv.GaussianBlur(gray_fireImg, (7, 7), 0)
    gray_fireImg = contrast_brightness_demo(gray_fireImg, 5.0, 25)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    gray_fireImg = cv.morphologyEx(gray_fireImg, cv.MORPH_CLOSE, kernel)
    dst = cv.bitwise_and(img, img, mask=gray_fireImg)
    
    gray_fn=cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    return gray_fn

    #cv.imshow("fn",gray_fn)
    #cv.imshow("row",img)
    #cv.imshow("fire", dst)
    #cv.imshow("gray_fireImg", gray_fireImg)
    #cv.waitKey()

if __name__=="__main__":
    print("giaogiao!")
    img = cv.imread('fire1.jpg')
    ret = get_color_seg(img)
    cv.imshow("fn",ret)
    cv.waitKey()









# def contrast_brightness_demo(image, c, b):  #其中c为对比度，b为每个像素加上的值（调节亮度）
#     blank = np.zeros(image.shape, image.dtype)   #创建一张与原图像大小及通道数都相同的黑色图像
#     dst = cv.addWeighted(image, c, blank, 1-c, b) #c为加权值，b为每个像素所加的像素值
#     ret, dst = cv.threshold(dst, 25, 255, cv.THRESH_BINARY)
#     return dst


# capture = cv.VideoCapture("C:\\Users\\xxx\\Desktop\\2.mp4")
# redThre = 105
# saturationTh = 42
# while(True):
#     ret, frame = capture.read()
#     cv.imshow("frame", frame)
#     B = frame[:, :, 0]
#     G = frame[:, :, 1]
#     R = frame[:, :, 2]
#     minValue = np.array(np.where(R <= G, np.where(G <= B, R, np.where(R <= B, R, B)), np.where(G <= B, G, B)))
#     S = 1 - 3.0 * minValue / (R + G + B + 1)
#     fireImg = np.array(np.where(R > redThre, np.where(R >= G, np.where(G >= B, np.where(S >= 0.2, np.where(S >= (255 - R)*saturationTh/redThre, 255, 0), 0), 0), 0), 0))
#     gray_fireImg = np.zeros([fireImg.shape[0], fireImg.shape[1], 1], np.uint8)
#     gray_fireImg[:, :, 0] = fireImg
#     gray_fireImg = cv.GaussianBlur(gray_fireImg, (7, 7), 0)
#     gray_fireImg = contrast_brightness_demo(gray_fireImg, 5.0, 25)
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
#     gray_fireImg = cv.morphologyEx(gray_fireImg, cv.MORPH_CLOSE, kernel)
#     dst = cv.bitwise_and(frame, frame, mask=gray_fireImg)
#     cv.imshow("fire", dst)
#     cv.imshow("gray_fireImg", gray_fireImg)
#     c = cv.waitKey(40)
#     if c == 27:
#         break