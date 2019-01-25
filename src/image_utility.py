import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_name, cmap=1):
    return cv2.imread(image_name, cmap)


def show_image(image, image_name, image_category='', use_plt=False):
    if len(image_category) == 0:
        title = 'Image: ' + image_name
    else:
        title = 'Category: ' + image_category + ' | Image: ' + image_name
    if use_plt:
        plt.axis("off")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()
    else:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

