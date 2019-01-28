

def get_labels(n_labels):
    return ['0000' + str(i) if i < 10 else '000' + str(i) for i in range(n_labels)]


def get_image_label(label_code, labels):
    return [1 if label_code == i else 0 for i in labels]
