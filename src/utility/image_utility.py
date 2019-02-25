import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_name, cmap=1):
    return cv.imread(image_name, cmap)


def resize_image(image, size):
    return cv.resize(image, size)


def show_image(image, image_name, image_category='', use_plt=False):
    if len(image_category) == 0:
        title = 'Image: ' + image_name
    else:
        title = 'Category: ' + image_category + ' | Image: ' + image_name
    if use_plt:
        plt.axis("off")
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()
    else:
        cv.imshow(title, image)
        cv.waitKey(0)
        cv.destroyAllWindows()


def img_to_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


def equalize_img(image):
    return cv.equalizeHist(image)


def normalize_img(image, alpha=0, beta=1):
    if alpha == 0 and beta == 1:
        return cv.normalize(image, None, alpha=alpha, beta=beta, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    else:
        return cv.normalize(image, None, alpha=alpha, beta=beta, norm_type=cv.NORM_MINMAX)

