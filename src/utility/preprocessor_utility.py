from src.utility.image_utility import resize_image


def preprocess_image(image, img_size=(46, 46)):
    resized = resize_image(image, img_size)
    return resized