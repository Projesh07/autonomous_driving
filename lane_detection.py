import cv2
import matplotlib.pyplot as plt
import numpy as np


class LaneDetection(object):
	"""docstring for LaneDetection"""
	def __init__(self):
		pass

	"""docstring for image bgr to gray scale"""
	def chanel_conversion(self,img):

		gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		return gray_img

	"""docstring for image noise reduction by gaussian blur"""
	def feature_extraction(self,img):
		#step 1 , parameter need to tune
		blur=cv2.GaussianBlur(img,(5,5),0)
		
		#step 2 edge detection by canny
		c_img=cv2.Canny(blur,50,150)

		return c_img

	"""docstring for finding roi"""
	def roi(self,img):
		height=img.shape[0]
		roi=np.array([[(50,height),(275,height),(150,140)]])
		mask_img=self.mask_img(roi,img)
		roi=cv2.bitwise_and(img,mask_img)
		return roi

	"""docstring masking roi"""
	def mask_img(self,roi,img):
		mask=np.zeros_like(img)
		cv2.fillPoly(mask,roi,255)
		return mask


	"""docstring perform detection base on HoughTransformP"""
	def hough_p_transform(self,img):

		lines=cv2.HoughLinesP(img,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)

		return lines

	"""docstring perform detection"""
	def lane_detection(self,img):

		#step 1
		test_img=np.copy(img)
		gray_img=self.chanel_conversion(test_img)
		#step 2
		canny_img=self.feature_extraction(gray_img)
		#step 3
		cropped_img=self.roi(canny_img)
		#step 4
		lines=self.hough_p_transform(cropped_img)
		#step 4
		lane_lines=np.zeros_like(test_img)

		for line in lines:
			x1,y1,x2,y2=line.reshape(4)
			cv2.line(lane_lines,(x1,y1),(x2,y2),(255,0,0),10)

		lane=cv2.addWeighted(img,0.8,lane_lines,1,1)
		return lane



if __name__=="__main__":

	lane_detection=LaneDetection()

	cap=cv2.VideoCapture("Road.mp4")

	while (cap.isOpened()):

		#step 1 read image
		ret,lane_img=cap.read()
		#lane_img=cv2.imread("test_image.jpg")
		lane_img=cv2.resize(lane_img,(320,320))
		lane=lane_detection.lane_detection(lane_img)
		cv2.imshow("lane detected image",lane)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()	
	cv2.destroyAllWindows()


		

