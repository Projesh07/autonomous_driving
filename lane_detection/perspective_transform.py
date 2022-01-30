import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.image as mpimg


class PerspectiveTransform(object):
	"""docstring for PerspectiveTransform"""
	def __init__(self):
		pass


	"""Return transformation matrix and inverse matrix"""
	def transformation_matrix(self,img,src=None,dst=None):

		src=src
		dst=dst
		
		if  not (src and dst):
			src = np.float32(
			[(550, 540),     # top-left
			(250, 720),     # bottom-left
			(800, 720),    # bottom-right
			(770, 540)])    # top-right

			dst = np.float32([(100, 0),
			(100, 720),
			(1100, 720),
			(1100, 0)])

		t_matrix=cv2.getPerspectiveTransform(src,dst)
		t_inv_matrix=cv2.getPerspectiveTransform(dst,src)

		return t_matrix,t_inv_matrix


	"""Perfrom transformation based on matrix and inverse matrix if flag 1 inverse transform else inverse"""

	def forward_transformation(self,img,src=None,dst=None,flags=None):

		img_size=img.shape[1],img.shape[0]
		t_matrix,t_inv_matrix=self.transformation_matrix(src,dst)

		return cv2.warpPerspective(img, t_matrix, img_size, cv2.INTER_LINEAR)


	def backward_transformation(self,img,src=None,dst=None,flags=None):

		img_size=img.shape[1],img.shape[0]
		t_matrix,t_inv_matrix=self.transformation_matrix(src,dst)

		return cv2.warpPerspective(img, t_inv_matrix, img_size, cv2.INTER_LINEAR)
		

		



