import numpy as np
import cv2
import glob

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((6*9,3), np.float32)  # 7x8的格子 此处参数根据使用棋盘格规格进行修改
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) 
# 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y  

objpoints = [] # 存储3D点
imgpoints = [] # 存储2D点

images = glob.glob('R.jpg') # 文件存储路径，存储需要标定的摄像头拍摄的棋盘格图片

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    if ret == True:
        objpoints.append(objp)
        
        # 在原角点的基础上寻找亚像素角点
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        if corners2.any():
            imgpoints.append(corners2)
        else:
            imgpoints.append(corners)

        # 绘制角点并显示
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(5000)

cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print ("ret:",ret  )                                                                      
print ("mtx:\n",mtx )       # 内参数矩阵                                                  
print ("dist:\n",dist)      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)   
print ("rvecs:\n",rvecs)    # 旋转向量  # 外参数                                          
print ("tvecs:\n",tvecs )   # 平移向量  # 外参数     
# print ("objpoints\n", objpoints)
# print ("imgpoints\n", imgpoints)

# Save
ret, rvecs, tvecs = np.array(ret), np.array(rvecs), np.array(tvecs)
data = [ ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints ]
filenames = [   'data_R_ret.npy', 'data_R_mtx.npy', 'data_R_dist.npy', 'data_R_rvecs.npy', 'data_R_tvecs.npy', 
                'data_R_objpoints.npy', 'data_R_imgpoints.npy' ]
for arr, filename in zip(data, filenames):
    np.save(filename, arr)