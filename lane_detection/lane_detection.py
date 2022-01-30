import cv2
import numpy as np
import lane_detection.perspective_transform as ptrasnfrom
import lane_detection.image_segmentation as imgs
import lane_detection.feature_extractor as fex

import lane_detection.utils as utility



class LaneDetection(object):
	"""docstring for LaneDetection"""
	def __init__(self):

		self.perspective_transfrom=ptrasnfrom.PerspectiveTransform()
		self.segmentation=imgs.ImageSegmentation()
		self.feature_extraction=fex.FeatureExtractor()
		self.utility = utility.Utalities()
		pass


	"""docstring lane detection"""
	def lane_detection(self,img):

		#step 1, make a copy of the image
		test_img=np.copy(img)

		#step 2 perfrom perspective transform

		ft_img=self.perspective_transfrom.forward_transformation(test_img,src=None,dst=None,flags=1)
		#backward_transfrom=self.perspective_transfrom.backward_transformation(test_img,src=None,dst=None,flags=1)

		#step 3 perfrom segmentation or other feature extraction
		#return color_segmentation

		color_segmentation=self.segmentation.hsv_color_segmentation(ft_img)
		
		#step 4 perfrom color space and window sliding feature extraction
		#return features

		features,postions=self.feature_extraction.window_feature_extractor(color_segmentation)



		#step 2 perfrom  backward perspective transform and draw segmented part
		feature_map=self.perspective_transfrom.backward_transformation(features,src=None,dst=None)

		output_image=feature_map



		left_curveR, right_curveR, pos=self.utility.measure_curvature(postions[0],postions[1])


		print(left_curveR, right_curveR, postions)

		print(output_image.shape)

		output_image=self.utility.plot(output_image,postions[0],postions[1])
		if output_image.shape[2]!=test_img.shape[2]:
			input_image_alphachann = np.full((feature_map.shape[0],feature_map.shape[1]), 128, dtype=test_img.dtype)
			output_image = np.dstack((feature_map, input_image_alphachann))


		lane = cv2.addWeighted(test_img, 1, output_image, 0.9, 0)



		return lane





			





		

