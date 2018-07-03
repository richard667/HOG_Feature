import cv2
import numpy as np
import json
import matplotlib.pylab as plt

# image->window->block->cell->pixel
class HogFeature:
    def __init__(self, img_name, win_w, win_h, cell_size, block_size, block_stride, bin_num=9, grad_as_weight=False):
        img = cv2.imread(img_name)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.win_width = win_w
        self.win_height = win_h
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.bin_num = bin_num
        # self.gradient, self.alpha = self._gradient()
        self.gradient_as_weight = grad_as_weight

    # 假设输入为黑白图像
    def _gradient(self):
        # img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        kernelx = np.array([-1, 0, 1])
        kernely = kernelx.T
        gradientx = cv2.filter2D(self.img, -1, kernelx)
        gradienty = cv2.filter2D(self.img, -1, kernely)
        grad_img = np.sqrt(gradientx*gradientx + gradienty*gradienty)
        alpha = np.arctan2(gradienty, gradientx)
        return grad_img, alpha

    def _cell_statistics(self, grad_cell, alpha_cell):
        # cell is a 1D array of size cell_size*cell_size
        bin_range = 180 / self.bin_num
        bin_statistics = np.zeros(self.bin_num)
        if self.gradient_as_weight:
            for grad, alpha in zip(grad_cell, alpha_cell):
                bin_statistics[alpha%180//bin_range] += grad  # alpha范围在0-360，alpha%180相当于对大于180的alpha减180
        else:
            for alpha in alpha_cell:
                bin_statistics[alpha%180//bin_range] += 1
        return bin_statistics

    def _block_statistics(self, grad_block, alpha_block):
        '''
        :param grad_block:  2D array size of block_size * block_size
        :param alpha_block:
        :return: block_feature: a vector, represent HOG feature of a block
                 num_cell**2  : length of vector
        '''
        num_cell = (self.block_size / self.cell_size)
        block_feature = []
        assert type(num_cell) == int, "block size must be exactly dividable of cell size"
        for i in range(num_cell):
            for j in range(num_cell):
                grad_cell = grad_block[i*self.cell_size:(i+1)*self.cell_size, j*self.cell_size:(j+1)*self.cell_size]
                alpha_cell = alpha_block[i*self.cell_size:(i+1)*self.cell_size, j*self.cell_size:(j+1)*self.cell_size]
                bin_statis = self.cell_statistics(grad_cell.ravel(), alpha_cell.ravel())
                block_feature += bin_statis.tolist()
        block_feature = np.array(block_feature)
        norm_2 = np.linalg.norm(block_feature)  # 向量二范数
        block_feature = block_feature / norm_2
        return block_feature

    def _window_process(self, grad_window, alpha_window):
        '''
        :param grad_window:
        :param alpha_window:
        :return: win_feature: HOG feature of a window,
                num_h*num_w: length of HOG feature vector
        '''
        num_h = self.win_height / self.block_size
        num_w = self.win_width / self.block_size
        win_feature = []
        for i in range(num_h):
            for j in range(num_w):
                row = self.cell_size*self.block_size
                param_1 = grad_window[i*row:(i+1)*row, j*row:(j+1)*row]
                param_2 = alpha_window[i*row:(i+1)*row, j*row:(j+1)*row]
                win_feature += self._block_statistics(param_1, param_2).tolist()
        return win_feature, num_h*num_w

    def img_process(self):
        '''
        实际上训练数据只有一个窗口
        :return:
        '''
        img_h, img_w = self.img.shape
        h_iter = img_h - self.win_height + 1
        w_iter = img_w - self.win_width + 1
        gradient_img, alpha_img = self._gradient()
        img_hog = []
        for i in range(h_iter):
            for j in range(w_iter):
                gradient_window = gradient_img[i:self.win_height + i, j:j+self.win_width]
                alpha_window = alpha_img[i:self.win_height + i, j:j+self.win_width]
                win_feature, n = self._window_process(gradient_window, alpha_window)
                img_hog.append(win_feature)
        return img_hog
                # cv2.imshow("window", gradient_window.astype(np.uint8))
                # print(j)
                # cv2.waitKey(1)
        # cv2.imshow("img", gradient_img.astype(np.uint8))
        # cv2.waitKey(0)
        # with open("img.json", "w") as f:
        #     json.dump(gradient_img.tolist(), f)







# self, img_name, win_w, win_h, cell_size, block_size, block_stride, bin_num=9, grad_as_weight=False
hog = HogFeature("12.jpg", 80, 80, 8, 16, 8)
img = hog.img_process()
# print(type(img))
# cv2.imshow("img", img)
# cv2.waitKey(0)


# def single_channel_gradient(self, img):
#     kernelx = np.array([-1, 0, 1])
#     kernely = kernelx.T
#     gradientx = cv2.filter2D(img, -1, kernelx)
#     gradienty = cv2.filter2D(img, -1, kernely)
#     return gradientx, gradienty
#
#
# def gradient(self):
#     (B, G, R) = cv2.split(self.img)
#     gradient_B = self.single_channel_gradient(B)
#     gradient_G = self.single_channel_gradient(G)
#     gradient_R = self.single_channel_gradient(R)
#     print(type(gradient_G))
#     gradient_img = map(max, gradient_B, gradient_G, gradient_R)
#     return gradient_img

