import cv2
import numpy as np

import lane_detection.utils as utility

class FeatureExtractor(object):
	"""docstring for ClassName"""
	def __init__(self):
		self.left_fit=[]
		self.right_fit=[]

		self.utility=utility.Utalities()
		pass


	def _window(self,img,nwindows=9):
		""" Extract features from a binary image

		Parameters:
		img (np.array): A binary image
		"""
		img = img
		# Height of of windows - based on nwindows and image shape
		window_height = np.int(img.shape[0]//nwindows)

		#print("window height "+str(window_height))

		# Identify the x and y positions of all nonzero pixel in the image
		nonzero = img.nonzero()


		"""
		lane_lines=np.zeros_like(img)

		if nonzero is not None:
			for x in range(20502):

				##print(x)
				##print(intensity)
				#print("Hello")
				#cv2.line(img,(x,img.shape[0]),(x,img.shape[0]-intensity//255//4),(255,0,255),10)

				cv2.circle(lane_lines,(x_cordinates[x],y_cordinates[x]),20,(255,255,255),cv2.FILLED)

		plt.imshow(lane_lines)
		plt.show()
		"""
		return window_height, nonzero


	def _histogram(self,img,region=2):


		#taking part of the image
		bottom_half = img[img.shape[0]//region:,:]
		return np.sum(bottom_half, axis=0)


	def _find_histogram_points(self,img,histogram):

		#print(histogram.shape)

	    # find middel of the histogram
		midpoint = histogram.shape[0]//2
		#print(midpoint)

		#draw middle point
		#cv2.circle(img,(midpoint,img.shape[0]),50,(255,255,0),cv2.FILLED)

		# find left of the histogram
		leftx_base = np.argmax(histogram[:midpoint])

		#draw left point
		#cv2.circle(img,(leftx_base,img.shape[0]),50,(255,255,0),cv2.FILLED)

		
		# find right of the histogram
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		#draw right point
		#cv2.circle(img,(rightx_base,img.shape[0]),50,(255,255,0),cv2.FILLED)

		return leftx_base,midpoint,rightx_base

	def features_in_window(self, img,nonzero,center, margin, height):
		x_cordinates = np.array(nonzero[1])
		y_cordinates= np.array(nonzero[0])
		""" Return all pixel that in a specific window

		Parameters:
		center (tuple): coordinate of the center of the window
		margin (int): half width of the window
		height (int): height of the window

		Returns:
		pixelx (np.array): x coordinates of pixels that lie inside the window
		pixely (np.array): y coordinates of pixels that lie inside the window
		"""
		topleft = (center[0]-margin, center[1]-height//2)
		bottomright = (center[0]+margin, center[1]+height//2)
		condx = (topleft[0] <= x_cordinates) & (x_cordinates <= bottomright[0])
		condy = (topleft[1] <= y_cordinates) & (y_cordinates <= bottomright[1])
		
		return x_cordinates[condx&condy], y_cordinates[condx&condy]

	


	def extract_feature(self,img,nonzero,nwindows,window_height,window_width,leftx_base,rightx_base,minpix=50):
		# Current position to be update later for each window in nwindows
		leftx_current = leftx_base
		##print("leftx_current "+str(leftx_current))

		rightx_current = rightx_base
		##print("rightx_current "+str(rightx_current))
		y_current = img.shape[0] + window_height//2

		# Create empty lists to reveice left and right lane pixel
		leftx, lefty, rightx, righty = [], [], [], []

		# Step through the windows one by one
		for _ in range(nwindows):
			##print(y_current)
			y_current -= window_height
			center_left = (leftx_current, y_current)
			center_right = (rightx_current, y_current)
			#print("center_left "+str(center_left))
			#print("center_right "+str(center_right))


			# view each point on the left, right
			#cv2.circle(img,(center_left[0],center_left[1]),50,(255,255,25),cv2.FILLED)
			#cv2.circle(img,(center_right[0],center_right[1]),50,(255,255,25),cv2.FILLED)

			good_left_x, good_left_y = self.features_in_window(img,nonzero,center_left, window_width, window_height)
			#cv2.circle(img,bottomright,50,(255,255,0),cv2.FILLED)

			good_right_x, good_right_y = self.features_in_window(img,nonzero,center_right, window_width, window_height)

			# Append these indices to the lists
			leftx.extend(good_left_x)
			lefty.extend(good_left_y)
			rightx.extend(good_right_x)
			righty.extend(good_right_y)

			#return img
			if len(good_left_x) > minpix:
				leftx_current = np.int32(np.mean(good_left_x))
			if len(good_right_x) > minpix:
				rightx_current = np.int32(np.mean(good_right_x))


		print(len(rightx))
		print(len(rightx))
		#4388,4388, 18386,18386
		return leftx, lefty, rightx, righty

	def fit_poly(self, leftx, lefty, rightx, righty,img):
		"""Find the lane line from an image and draw it.

		Parameters:
		img (np.array): a binary warped image

		Returns:
		out_img (np.array): a RGB image that have lane line drawn on that.
		"""

		print(len(rightx))
		print(len(rightx))
		# Create an output image to draw on and visualize the result
		out_img = np.dstack((img, img, img))

		#return out_img
		
		#left_fit=None
		#right_fit=None

		print(len(righty))

		if len(lefty) > 1500:
			self.left_fit = np.polyfit(lefty, leftx, 2)
		if len(righty) > 1500:
			self.right_fit = np.polyfit(righty, rightx, 2)

		#print(len(lefty) )

		# Generate x and y values for plotting
		maxy = out_img.shape[0] - 1
		miny = out_img.shape[0] // 3
		if len(lefty):
			maxy = max(maxy, np.max(lefty))
			miny = min(miny, np.min(lefty))

		if len(righty):
			maxy = max(maxy, np.max(righty))
			miny = min(miny, np.min(righty))

		ploty = np.linspace(miny, maxy, img.shape[0])

		print("ploty len"+str(len(ploty)))

		##print(right_fit[0])
		
		##print(right_fit[1])
		#print(right_fit[2])

		if len(self.left_fit) and len(self.right_fit): 

			left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty+self.left_fit[2]
			right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty+self.right_fit[2]

			#print("fit left"+str(left_fitx))
			
			#print("right_fitx "+str(right_fitx))

			# Visualization
			for i, y in enumerate(ploty):
				l = int(left_fitx[i])
				r = int(right_fitx[i])
				y = int(y)
				cv2.line(out_img, (l, y), (r, y), (255, 0, 0))

		#lR, rR, pos = self.utility.measure_curvature(self.left_fit,self.right_fit)

		return out_img

	def window_feature_extractor(self,img,nwindows=9,region=2,minpix=50):
		""" Extract features from a binary image"""

		window_height, nonzero=self._window(img)
		window_width=100

	

		histogram=self._histogram(img,region)

		leftx_base,midpoint,rightx_base=self._find_histogram_points(img,histogram)

		leftx, lefty, rightx, righty=self.extract_feature(img,nonzero,nwindows,window_height,window_width,leftx_base,rightx_base,minpix)

		output_img=self.fit_poly(leftx, lefty, rightx, righty,img)
		return output_img,np.array([self.left_fit,self.right_fit])
