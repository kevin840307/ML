import scipy.io
import os
import numpy as np
import skimage.io as io
from skimage.transform import resize
from tensorflow.examples.tutorials.mnist import input_data

class MnistDataset():
    def __init__(self, image_height=28, image_width=28, path='./MNIST/', one_hot=False):
        self.path = path
        self.image_height = image_height
        self.image_width = image_width
        self.mnist = input_data.read_data_sets(path, one_hot=one_hot)

    def get_minbatch(self, batch_size, time, type='train', augmentation=False):
        images = []
        labels = []
        mnist = self.mnist
        if type == 'train':
            all_images, all_labels = mnist.train.images, mnist.train.labels
        elif type == 'test':
            all_images, all_labels = mnist.test.images, mnist.test.labels
        else:
            all_images, all_labels = mnist.validation.images, mnist.validation.labels
        all_images = np.reshape(all_images, [-1, 28, 28, 1])

        echoe = (len(all_images) - batch_size) // batch_size
        start_index = (time % echoe) * batch_size
        np.random.seed((time // echoe) + 1)
        data_index = np.arange(0, len(all_images), 1, dtype=np.int)
        np.random.shuffle(data_index)
        
        for index in range(start_index , start_index + batch_size, 1):
            img_index = data_index[index]
            image = all_images[img_index]
            image = resize(image, (self.image_height, self.image_width))

            if image.max() > 1:
                image = image / 255.

            if augmentation and np.random.randint(0, 10) < 5:
                image = np.rot90(image, np.random.randint(1, 4))

            images.append(image)
            labels.append(all_labels[img_index])

        return np.array(images), np.reshape(labels, (-1, 1))

    def get_size(self, type='train'):
        if type == 'validation':
            return len(self.mnist.validation.images)
        if type == 'train':
            return len(self.mnist.train.images)
        return len(self.mnist.test.images)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    mnist = MnistDataset(image_height=28, image_width=28)
    imgs, lables = mnist.get_minbatch(100, 0)
    print(imgs)
    print(lables)
    plt.imshow(np.tile(imgs[0], [1, 1, 3]))
    plt.show()
