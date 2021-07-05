# https://blog.csdn.net/herobinbin/article/details/50223205

# 实现ohta算法

import cv2 as cv
import numpy as np

def get_ohta_seg(imgt):
    print("###################")
    img=imgt.astype(np.int16)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    I1=(R+G+B)/3
    I2= (R-B)/2         # (cv.subtract(R,B)/2) #.astype(np.uint8)    #I2=(R-B)/2   ### 直接减法会存在无符号整数溢出现象 采用opencv自带减法
    #I3=(2*G-R-B)/4  not use

    ## way1: I1取[100,220]内  I2取[18，120]内 
    #        计算I’1+I’2   将I’1+I’2与I2 按位与运算 得到 r    ####  论文有些混乱  故这里直接将r=I’1+I’2
    #        对r进行 otsu 得到火焰分隔区域
    I1[I1<100]=0
    I1[I1>220]=0
    I2_=I2.copy()
    I2_[I2_<18]=0
    I2_[I2_>120]=0         # 可以用opencv自带threshold
    r0=I1+I2_
    #r1=np.bitwise_and(r0,I2)
    #r1=r0 & I2
    r0[r0>255]=255
    ############ 0->35   ########## TODO 考虑自适应阈值  现为手动的35
    I2[I2<35]=0

    to=I2.shape[0]*I2.shape[1]
    if np.sum(I2>40)/(to) > 0.015:  
        #print(np.sum(I2>40))
        #print(">40={},{}% 30-40={},{}% 20-30={},{}% <20={},{}%".format(c40,100*c40/to, c30,100*c30/to, c20,100*c20/to,c00,100*c00/to))
        r1 = cv.bitwise_and(r0.astype(np.uint8), I2.astype(np.uint8))
        th1, ret1 = cv.threshold(r1, 0, 255, cv.THRESH_OTSU) 
    else :
        ret1 = np.zeros([I2.shape[0],I2.shape[1]],dtype=np.uint8)

    return ret1
    #print(ret1.shape)
    #cv.imshow("fn0",imgt)   
    #cv.imshow("fn1",ret1)  ##########
    #cv.waitKey()


if __name__ == "__main__":
    img3=cv.imread("./fire3.jpg")
    rst = get_ohta_seg(img3)
    cv.imshow("fn0",rst)  
    cv.waitKey()
    # imgs1=cv.imread("./smoke1.jpg")
    # imgs2=cv.imread("./smoke2.jpg")
    # imgs3=cv.imread("./smoke3.jpg")
    # img1=cv.imread("./fire1.jpg")
    # img2=cv.imread("./fire2.jpg")
    # img3=cv.imread("./fire3.jpg")

    # ohta(imgs1)
    # ohta(imgs2)
    # ohta(imgs3)
    # ohta(img1)
    # ohta(img2)
    # ohta(img3)


