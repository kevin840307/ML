import scipy.io
import os
import numpy as np
import skimage.io as io
from skimage.transform import resize

class FlowerDataset():
    def __init__(self, image_height=100, image_width=100, mat_path='./flower', img_path='./flower/crop_image'):
        self.mat_path = mat_path
        self.img_path = img_path
        self.image_height = image_height
        self.image_width = image_width
        self.build()

    def build(self):
        labels = scipy.io.loadmat(self.mat_path + '/imagelabels.mat')
        self.labels = labels['labels'][0] - 1
        ids = scipy.io.loadmat(self.mat_path + '/setid.mat')
        self.train = ids['trnid'][0] - 1
        self.validation = ids['valid'][0] - 1
        self.test = ids['tstid'][0] - 1

    def get_minbatch(self, batch_size, time, type='train', augmentation=True):
        images = []
        labels = []
        data_dir_list = os.listdir(self.img_path)
        data_index = np.copy(self.validation)
        if type == 'train':
            data_index = np.copy(self.train)
        elif type == 'test':
            data_index = np.copy(self.test)

        echoe = (len(data_index) - batch_size) // batch_size
        start_index = (time % echoe) * batch_size
        np.random.seed((time // echoe) + 1)
        np.random.shuffle(data_index)

        for index in range(start_index , start_index + batch_size, 1):
            img_index = data_index[index]
            image = io.imread(os.path.join(self.img_path + '/' + data_dir_list[img_index]), as_gray=False)
            image = resize(image, (self.image_height, self.image_width))

            if image.max() > 1:
                image = image / 255.

            if augmentation and np.random.randint(0, 10) < 5:
                image = np.rot90(image, np.random.randint(1, 4))

            images.append(image)
            labels.append(self.labels[img_index])

        return np.array(images), np.reshape(labels, (-1, 1))

    def get_size(self, type='train'):
        if type == 'validation':
            return len(self.validation)
        if type == 'train':
            return len(self.train)
        return len(self.test)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #flower = FlowerDataset(image_height=50, image_width=50)
    #imgs, lables = flower.get_minbatch(4, 0)
    #print(imgs)
    #print(lables)
    #plt.imshow(imgs[0])
    #plt.show()

    # make crop image
    #for filename in os.listdir('./flower/image'):
    #    img = Image.open('./flower/image/' + filename)
    #    h=500
    #    x=int((img.width-h)/2)
    #    y=int((img.height-h)/2)
    #    img =img.crop([x,y,x+h,y+h])
    #    img.save('./flower/crop_image/' + filename)
    from PIL import Image
    flower = FlowerDataset(image_height=50, image_width=50)
    imgs, labels = flower.get_minbatch(50, 0)
    #save_index = 0
    #for index in range(30):
    #    imgs, labels = flower.get_minbatch(50, index)
    #    labels = np.reshape(labels, -1)
    #    imgs = imgs[labels == 1]
    #    for img in imgs:
    #        Image.fromarray((img * 255).astype(np.uint8)).save('./save/'+str(save_index)+'.jpg')
    #        save_index += 1


    #for index in range(100):
    #    imgs, labels = flower.get_minbatch(50, index, type='validation')
    #    labels = np.reshape(labels, -1)
    #    imgs = imgs[labels == 1]
    #    for img in imgs:
    #        Image.fromarray((img * 255).astype(np.uint8)).save('./save/v_'+str(save_index)+'.jpg')
    #        save_index += 1

