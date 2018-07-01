import cv2
import numpy as np

class HogFeature:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name)

    def single_channel_gradient(self, img):
        kernelx = np.array([-1, 0, 1])
        kernely = kernelx.T
        gradientx = cv2.filter2D(img, -1, kernelx)
        gradienty = cv2.filter2D(img, -1, kernely)
        return gradientx, gradienty

    def gradient(self):
        (B, G, R) = cv2.split(self.img)
        gradient_B = self.single_channel_gradient(B)
        gradient_G = self.single_channel_gradient(G)
        gradient_R = self.single_channel_gradient(R)
        print(type(gradient_G))
        gradient_img = map(max, gradient_B, gradient_G, gradient_R)
        return gradient_img




hog = HogFeature("11.jpg")
img = hog.gradient()
print(type(img))
# cv2.imshow("img", img)
# cv2.waitKey(0)
