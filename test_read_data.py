import numpy as np
import cv2
import glob

# Read
filenames = [   'data_R_ret.npy', 'data_R_mtx.npy', 'data_R_dist.npy', 'data_R_rvecs.npy', 'data_R_tvecs.npy', 
                'data_R_objpoints.npy', 'data_R_imgpoints.npy' ]
data = []
for filename in filenames:
    arr = np.load(filename)
    data.append(arr)
ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = data
convert = lambda arr: [ np.array(x) for x in arr.tolist() ]
ret, rvecs, tvecs, objpoints, imgpoints = ret.tolist(), convert(rvecs), convert(tvecs), convert(objpoints), convert(imgpoints)

# 畸变校正
images = glob.glob('R.jpg')

img = cv2.imread(images[0])  
h, w = img.shape[:2]  
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))  
print (newcameramtx ) 
print("------------------使用undistort函数-------------------")  
dst = cv2.undistort(img,mtx,dist,None,newcameramtx)  
x,y,w,h = roi  
dst1 = dst[y:y+h,x:x+w]  
cv2.imwrite('calib_result11.jpg', dst1)  
print( "方法一:dst的大小为:", dst1.shape  )
  
# undistort方法二  
print("-------------------使用重映射的方式-----------------------")  
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)  # 获取映射方程  
#dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)      # 重映射  
dst = cv2.remap(img,mapx,mapy,cv2.INTER_CUBIC)        # 重映射后，图像变小了  
x,y,w,h = roi  
dst2 = dst[y:y+h,x:x+w]  
cv2.imwrite('calib_result11_2.jpg', dst2)  
print ("方法二:dst的大小为:", dst2.shape )       # 图像比方法一的小  
  
print("-------------------计算反向投影误差-----------------------")  
tot_error = 0  
for i in range(len(objpoints)):  
    img_points2, _ = cv2.projectPoints(objpoints[i],rvecs[i],tvecs[i],mtx,dist)  
    error = cv2.norm(imgpoints[i],img_points2, cv2.NORM_L2)/len(img_points2)  
    tot_error += error  
  
mean_error = tot_error/len(objpoints)  
print ("total error: ", tot_error  )
print ("mean error: ", mean_error)


