import cv2 as cv
import numpy as np

from math import atan2, degrees, pi

# frame = cv.imread('test_data.jpg')
frame = cv.imread('img1_300.jpg')
overtaking = 20


def get_map():
	gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	thresh = 127
	frame_bin = cv.threshold(gray_frame, thresh, 255, cv.THRESH_BINARY)[1]

	bitwiseNot = cv.bitwise_not(frame_bin)

	contours, hierarchy = cv.findContours(
		bitwiseNot, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

	return [contours[0]]


def get_angle(dot1, dot2):

	dx = dot2[0] - dot1[0]
	dy = dot2[1] - dot1[1]
	rads = atan2(-dy, dx)
	rads %= 2*pi
	degs = degrees(rads)

	return degs


contour = get_map()
do_flag = True

# pts1 = np.float32([[-200, 0], [400, 0],
#                    [0, 150], [200, 150]])
# pts2 = np.float32([[0, 0], [200, 0],
#                    [0, 200], [200, 200]])
pts1 = np.float32([[0, 0], [200, 0],
				   [67, 150], [120, 150]])
pts2 = np.float32([[0, 0], [200, 0],
				   [0, 200], [200, 200]])

# Apply Perspective Transform Algorithm
matrix = cv.getPerspectiveTransform(pts1, pts2)


while do_flag:
	for i, dot in enumerate(contour[0]):
		try:
			dot1, dot2 = tuple(dot[0]), tuple(contour[0][i+overtaking][0])
		except Exception as e:
			break

		black = np.zeros_like(frame)

		black = cv.drawContours(black, contour, -1, (255, 255, 255), 2)

		black = cv.circle(black, dot1,
						  5, (0, 255, 255), -1)
		black = cv.circle(black, dot2,
						  5, (255, 255, 0), -1)

		black = cv.line(black, dot1, dot2,
						(255, 0, 0), 3)

		angle = -get_angle(dot1=dot1, dot2=dot2) + 90

		# M = cv.getRotationMatrix2D(
		#     (black.shape[1]//2, black.shape[0]//2), angle, 1)

		M = cv.getRotationMatrix2D(
			(int(dot1[0]), int(dot1[1])), angle, 1)

		# rotated = cv.warpAffine(black, M, (black.shape[1], black.shape[0]*2))
		rotated = cv.warpAffine(frame, M, (black.shape[1], black.shape[0]*2))
		cropped = rotated[dot1[1]-150:dot1[1], dot1[0]-100:dot1[0]+100]
		result = cv.warpPerspective(cropped, matrix, (200, 200))
		
		ksize = (7, 7) 
  
		# Using cv2.blur() method  
		result = cv.blur(result, ksize) 

		cv.imshow('Result Image', result)
		cv.imshow('Cropped Image', cropped)
		# cv.imshow('Rotated Image', rotated)

		cv.imshow("aboba", black)
		k = cv.waitKey(1)
		if (k == 27):
			do_flag = False
			break

cv.destroyAllWindows()
