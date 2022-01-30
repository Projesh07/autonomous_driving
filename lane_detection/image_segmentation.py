import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.image as mpimg


class ImageSegmentation(object):
	"""docstring for ImageSegmentation"""
	def __init__(self):
		pass


	def threshold_rel(self,img, lo, hi):
	    vmin = np.min(img)
	    vmax = np.max(img)
	    vlo = vmin + (vmax - vmin) * lo
	    vhi = vmin + (vmax - vmin) * hi
	    return np.uint8((img >= vlo) & (img <= vhi)) * 255

	def threshold_abs(self,img, lo, hi):
	    return np.uint8((img >= lo) & (img <= hi)) * 255

	"""docstring for image bgr to gray scale"""
	def chanel_conversion(self,img):

		gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		return gray_img

	"""docstring for image noise reduction by gaussian blur"""
	def feature_extraction(self,img):
		#step 1 , bluring or noise reduction parameter need to tune
		blur=cv2.GaussianBlur(img,(3,3),0)
		
		#step 2 edge detection by canny
		c_img=cv2.Canny(blur,150,200)

		return c_img

	"""docstring perform detection base on HoughTransformP"""
	def hough_p_transform(self,img):

		img=self.chanel_conversion(img)
		img=self.feature_extraction(img)
		lines=cv2.HoughLinesP(img,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
		return lines

	def hsv_color_segmentation(self,img):

		#This par can replace by masking and edge detecting
		hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

		h_channel = hls[:,:,0]
		l_channel = hls[:,:,1]
		s_channel = hls[:,:,2]
		v_channel = hsv[:,:,2]

		right_lane = self.threshold_rel(l_channel, 0.8, 1.0)

		right_lane[:,:img.shape[1]//2] = 0

		left_lane = self.threshold_abs(h_channel, 50, 100)

		left_lane &= self.threshold_rel(v_channel, 0.7, 1.0)


		left_lane[:,img.shape[1]//2:] = 0

		segmented_image = right_lane | left_lane 

		return segmented_image
		