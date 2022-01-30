import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.image as mpimg


class Utalities():
    """docstring for Utalities"""

    def __init__(self):
        self.clear_visibility = True
        self.dir = []
        self.left_curve_img = mpimg.imread('left_turn.png')
        self.right_curve_img = mpimg.imread('right_turn.png')
        self.keep_straight_img = mpimg.imread('straight.png')
        self.left_curve_img = cv2.normalize(src=self.left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.right_curve_img = cv2.normalize(src=self.right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.keep_straight_img = cv2.normalize(src=self.keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        pass

    def plot(self, out_img,left_fit,right_fit):
        np.set_printoptions(precision=6, suppress=True)
        lR, rR, pos = self.measure_curvature(left_fit,right_fit)

        value = None
        if abs(left_fit[0]) > abs(right_fit[0]):
            value = left_fit[0]
        else:
            value = right_fit[0]

        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')

        if len(self.dir) > 10:
            self.dir.pop(0)

        W = 400
        H = 400
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        widget[0, :] = [255, 255, 255]
        widget[-1, :] = [255, 255, 255]
        widget[:, 0] = [255, 255, 255]
        widget[:, -1] = [255, 255, 255]
        #out_img[:H, :W] = widget

        direction = max(set(self.dir), key=self.dir.count)
        msg = "Keep Straight Ahead"
        curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
        if direction == 'L':
            #y, x = self.left_curve_img[:, :, 3].nonzero()
            #out_img[y, x - 100 + W // 2] = self.left_curve_img[y, x, :3]
            msg = "Left Curve Ahead"
        if direction == 'R':
            #y, x = self.right_curve_img[:, :, 3].nonzero()
            #out_img[y, x - 100 + W // 2] = self.right_curve_img[y, x, :3]
            msg = "Right Curve Ahead"
        if direction == 'F':
            pass
            #y, x = self.keep_straight_img[:, :, 3].nonzero()
            #out_img[y, x - 100 + W // 2] = self.keep_straight_img[y, x, :3]

        cv2.putText(out_img, msg, org=(10, 580), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255),
                    thickness=1)
        if direction in 'LR':
            cv2.putText(out_img, curvature_msg, org=(10, 600), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                        color=(255, 255, 0), thickness=1)

        cv2.putText(
            out_img,
            "Good Lane Keeping",
            org=(10, 620),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=2)

        cv2.putText(
            out_img,
            "Vehicle is {:.2f} m away from center".format(pos),
            org=(10, 640),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 255),
            thickness=2)

        return out_img

    def measure_curvature(self,left_fit_val,right_fit_val):
        ym = 30 / 720
        xm = 3.7 / 700

        left_fit = left_fit_val
        right_fit = right_fit_val
        y_eval = 700 * ym

        # Compute R_curve (radius of curvature)
        left_curveR = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curveR = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        xl = np.dot(left_fit_val, [700 ** 2, 700, 1])
        xr = np.dot(left_fit_val, [700 ** 2, 700, 1])
        pos = (1280 // 2 - (xl + xr) // 2) * xm
        return left_curveR, right_curveR, pos