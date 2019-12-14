import torch.utils.data as data

from PIL import Image
import os
import os.path
from scipy import io
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_multilabel_dataset(dir, labels):
    images_path = []
    images_labels = []
    dir = os.path.expanduser(dir)
    for fname in sorted(os.listdir(dir)):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            images_path.append(path)
            images_labels.append(labels[int(fname.split('.')[0]),:])

    return images_path, np.array(images_labels)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(data.Dataset):
    def __init__(self, root='/data/.data5/xiaoshuang/data/NUSWIDE', loader=default_loader):
        labels = io.loadmat(os.path.join(root, 'Label.mat'))['labels']
        imgs_path, imgs_labels = make_multilabel_dataset(root, labels)
        if len(imgs_labels) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs_path = imgs_path
        self.imgs_labels = imgs_labels
        self.loader = loader
