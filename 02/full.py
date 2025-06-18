import cv2
import numpy as np

# Deklarasi awal
banner = cv2.imread("banner.png")
img = cv2.imread("btr.jpg")
(h, w) = img.shape[:2]
center = (w // 2, h // 2)

# 01
matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, matrix, (w, h))

# 02
zoomed = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

# 03
flip_horizontal = cv2.flip(img, 1)  # Horizontal
flip_vertical = cv2.flip(img, 0)  # Vertikal

# 04
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
matrix = cv2.getAffineTransform(pts1, pts2)
result = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

# 05
imgGray = cv2.imread("btr.jpg", cv2.IMREAD_GRAYSCALE)

kernel_blur = np.ones((3, 3), np.float32) / 9

blurred = cv2.filter2D(imgGray, -1, kernel_blur)

kernel_edge = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

edge = cv2.filter2D(imgGray, -1, kernel_edge)

# Output
cv2.imshow("Nama", banner)
cv2.imshow("Normal", img)
cv2.imshow("Image Rotated", rotated)
cv2.imshow("Zoomed Image", zoomed)
cv2.imshow("Horizontal", flip_horizontal)
cv2.imshow("Vertical", flip_vertical)
cv2.imshow("Wrapping Image", result)
cv2.imshow("Grayed", imgGray)
cv2.imshow("Blur", blurred)
cv2.imshow("Kernel Edge", edge)

cv2.waitKey(0)
cv2.destroyAllWindows()
