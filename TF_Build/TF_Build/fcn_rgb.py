import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as trans
import tensorflow_tools_v2 as tf_tools
import os
import math
from skimage import color
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate = 0.00005
batch_size = 2
train_times = 10000
train_step = 10

num_class = 256
image_width = 256
image_height = 256
 
# [filter size, filter height, filter weight, filter depth]
conv1_size1 = [3, 3, 3, 64]
conv1_size2 = [3, 3, 64, 64]
conv2_size1 = [3, 3, 64, 128]
conv2_size2 = [3, 3, 128, 128]
conv3_size1 = [3, 3, 128, 256]
conv3_size2 = [3, 3, 256, 64]
conv4_size1 = [3, 3, 64, 64]
conv4_size2 = [3, 3, 64, 64]
conv5_size1 = [3, 3, 64, 64]
conv5_size2 = [3, 3, 64, 64]

conv6_size1 = [5, 5, 64, 32]
conv7_size1 = [1, 1, 32, num_class]


def predict(x):
    with tf.variable_scope("conv1_scope"):
        with tf.variable_scope("conv1_1"):
            conv1_out = tf_tools.conv2d(x, conv1_size1, activation='relu')
        with tf.variable_scope("conv1_2"):
            conv1_out = tf_tools.conv2d(conv1_out, conv1_size2, activation='relu')
        pool1 = tf_tools.max_pool(conv1_out)

    with tf.variable_scope("conv2_scope"):
        with tf.variable_scope("conv2_1"):
            conv2_out = tf_tools.conv2d(pool1, conv2_size1, activation='relu')
        with tf.variable_scope("conv2_2"):
            conv2_out = tf_tools.conv2d(conv2_out, conv2_size2, activation='relu')
        pool2 = tf_tools.max_pool(conv2_out)

    with tf.variable_scope("conv3_scope"):
        with tf.variable_scope("conv3_1"):
            conv3_out = tf_tools.conv2d(pool2, conv3_size1, activation='relu')
        with tf.variable_scope("conv3_2"):
            conv3_out = tf_tools.conv2d(conv3_out, conv3_size2, activation='relu')
        pool3 = tf_tools.max_pool(conv3_out)

    with tf.variable_scope("conv4_scope"):
        with tf.variable_scope("conv4_1"):
            conv4_out = tf_tools.conv2d(pool3, conv4_size1, activation='relu')
        with tf.variable_scope("conv4_2"):
            conv4_out = tf_tools.conv2d(conv4_out, conv4_size2, activation='relu')
        pool4 = tf_tools.max_pool(conv4_out)

    with tf.variable_scope("conv5_scope"):
        with tf.variable_scope("conv5_1"):
            conv5_out = tf_tools.conv2d(pool4, conv5_size1, activation='relu')
        with tf.variable_scope("conv5_2"):
            conv5_out = tf_tools.conv2d(conv5_out, conv5_size2, activation='relu')
        pool5 = tf_tools.max_pool(conv5_out)

    # fcn conv
    with tf.variable_scope("conv6_scope"):
        with tf.variable_scope("conv6_1"):
            conv6_out = tf_tools.conv2d(pool5, conv6_size1, activation='relu')

    with tf.variable_scope("conv7_scope"):
        with tf.variable_scope("conv7_1"):
            conv7_out = tf_tools.conv2d(conv6_out, conv7_size1, activation='relu')

    # fcn
    with tf.variable_scope("fcn16"):
        shape = pool4.get_shape().as_list()
        conv8_out = tf_tools.conv2d_transpose(conv7_out, shape[3], kernel_size=5)
        conv8_out = tf.add(pool4, conv8_out)

    with tf.variable_scope("fcn8"):
        shape = pool3.get_shape().as_list()
        conv9_out = tf_tools.conv2d_transpose(conv8_out, shape[3], kernel_size=5)
        conv9_out = tf.add(pool3, conv9_out)

    with tf.variable_scope("fcn32"):
        conv10_out = tf_tools.conv2d_transpose(conv9_out, output_size=[image_width, image_height], out_channel=num_class * 3, kernel_size=31, strides=[1, 8, 8, 1])
        conv10_out = tf.reshape(conv10_out, shape=[-1,image_width, image_height * 3, num_class])
        predict_output = tf.argmax(conv10_out, 3)
        predict_output = tf.expand_dims(predict_output, 3)
        predict_output = tf.reshape(predict_output, shape=[-1, image_width, image_height * 3, 1])
        
    return conv10_out, tf.cast(predict_output, tf.float32)


def loss(y, t):
    t = tf.cast(t, tf.int32)
    result = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(t, squeeze_dims=[3]), logits=y)
    return result

def train(loss, index):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=index)

def image_summary(label, image_data, channel=1):
    reshap_data = tf.reshape(image_data, [-1, image_width, image_height, channel])
    tf.summary.image(label, reshap_data, batch_size)


def accuracy(pred_image, label_image, real_image):
    image_summary("pred_image",  pred_image, channel=3)
    image_summary("label_image", label_image, channel=3)
    image_summary("real_image", real_image, channel=3)

    return 1. - (tf.reduce_mean(tf.abs(tf.subtract(label_image, pred_image))) / 255.)


def IOU(pred_image, label_image):
    # logical_and and logical_or only with binaray class
    # A * B / A + B - A * B

    IOUs = np.zeros(shape=(num_class))
    for index in range(num_class):
        Intersection = np.sum((label_image == index) * (pred_image == index))
        Union = np.sum(label_image == index) + np.sum(pred_image == index) - Intersection
        IOUs[index] = np.mean(Intersection[Union > 0] / Union[Union > 0])
        if math.isnan(IOUs[index]):
            IOUs[index] = 0

    return IOUs





#def adjustData(img,mask,flag_multi_class,num_class):
#    if(flag_multi_class):#此程序中不是多类情况，所以不考虑这个
#        img = img / 255
#        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
##if
##else的简洁写法，一行表达式，为真时放在前面，不明白mask.shape=4的情况是什么，由于有batch_size，所以mask就有3维[batch_size,wigth,heigh],估计mask[:,:,0]是写错了，应该写成[0,:,:],这样可以得到一片图片，
#        new_mask = np.zeros(mask.shape + (num_class,))
##np.zeros里面是shape元组，此目的是将数据厚度扩展到num_class层，以在层的方向实现one-hot结构
 
#        for i in range(num_class):
#            #for one pixel in the image, find the class in mask and convert it
#            into one-hot vector
#            #index = np.where(mask == i)
#            #index_mask =
#            (index[0],index[1],index[2],np.zeros(len(index[0]),dtype =
#            np.int64) + i) if (len(mask.shape) == 4) else
#            (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
#            #new_mask[index_mask] = 1
#            new_mask[mask == i,i] = 1#将平面的mask的每类，都单独变成一层，
#        new_mask =
#        np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3]))
#        if flag_multi_class else
#        np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
#        mask = new_mask
#    elif(np.max(img) > 1):
#        img = img / 255
#        mask = mask /255
#        mask[mask > 0.5] = 1
#        mask[mask <= 0.5] = 0
#    return (img,mask)
def trainGenerator(batch_size,train_path,image_folder,mask_folder,image_color_mode="rgb",
                    mask_color_mode="rgb",image_save_prefix="image",mask_save_prefix="mask",
                    flag_multi_class=False,num_class=5,save_to_dir=None,target_size=(image_width,image_height),seed=1):
    #rotation_range:隨機旋轉度數範圍
    #width_shift_range:隨機平移範圍
    #height_shift_range:隨機上下範圍
    #shear_range:隨機錯位(固定x或y，則非固定的則會像平移)
    #zoom_range:縮放範圍
    #horizontal_flip:隨機水平翻轉
    #fill_mode:內插法

    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')

    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_generator = image_datagen.flow_from_directory(train_path,#训练数据文件夹路径
            classes = [image_folder],#类别文件夹,对哪一个类进行增强
            class_mode = None,#不返回标签
            color_mode = image_color_mode,#灰度，单通道模式
            target_size = target_size,#转换后的目标图片大小
            batch_size = batch_size,#每次产生的（进行转换的）图片张数
            save_to_dir = save_to_dir,#保存的图片路径
            save_prefix  = image_save_prefix,#生成图片的前缀，仅当提供save_to_dir时有效
            seed = seed)
    mask_generator = mask_datagen.flow_from_directory(train_path,
            classes = [mask_folder],
            class_mode = None,
            color_mode = mask_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = mask_save_prefix,
            seed = seed)

    train_generator = zip(image_generator, mask_generator)#组合成一个生成器



    for (img,mask) in train_generator:
        img = img / 255.
        mask = np.reshape(mask, [batch_size, image_width, image_height * 3, 1])
        yield (img, mask)
        #img,mask =
        #adjustData(img,mask,flag_multi_class,num_class)#返回的img依旧是[2,256,256]
        #yield (img,mask)
def testGenerator(test_path, image_folder, mask_folder,num_image=30,target_size=(image_width,image_height),flag_multi_class=False,as_gray=True):
    for index in range(num_image):
        imgData = io.imread(os.path.join(test_path + image_folder + '/',"%d.jpg" % index),as_gray = as_gray)
        #imgData = io.imread(os.path.join(test_path,"%d.png" % index),as_gray =
        #as_gray)
        imgData = trans.resize(imgData,target_size)
        #imgData = np.reshape(imgData,imgData.shape + (1,)) if (not
        #flag_multi_class) else imgData
        imgData = np.reshape(imgData, (1,) + imgData.shape)
        #imgData = np.concatenate([imgData, imgData])

        label_size = (image_width, image_height * 3)
        imgLabel = io.imread(os.path.join(test_path + mask_folder + '/',"%d.png" % index),as_gray = as_gray)
        #imgLabel = io.imread(os.path.join(test_path,"%d_predict.png" %
        #index),as_gray = as_gray)
        imgLabel = color.rgba2rgb(imgLabel)
        imgLabel = trans.resize(imgLabel,target_size)
        imgLabel = np.reshape(imgLabel, (1,) + label_size + (1,))
        #imgLabel = np.concatenate([imgLabel, imgLabel])
        yield (imgData, imgLabel)


from PIL import Image
if __name__ == '__main__':

    # init
    trainGen = trainGenerator(batch_size,'data/membrane/train','image_2','label_2', save_to_dir = 'data/membrane/train/aug')
    testGene = testGenerator('data/membrane/train/','image_2','label_2', as_gray=False)
    imgData, imgMask = next(testGene)

    input_x = tf.placeholder(tf.float32, shape=[batch_size, image_width, image_height, 3], name="input_x")
    input_y = tf.placeholder(tf.float32, shape=[batch_size, image_width, image_height * 3, 1], name="input_y")

    # predict
    predict_op, real_op = predict(input_x)

    # real
    #real_op = predict_op
    
    # loss
    loss_op = loss(predict_op, input_y)

    # train
    index = tf.Variable(0, name="train_time")
    train_op = train(loss_op, index)

    # accuracy
    accuracy_op = accuracy(real_op, input_y, input_x)
    
    # graph
    summary_op = tf.summary.merge_all()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    summary_writer = tf.summary.FileWriter("log/", graph=session.graph)

    init_value = tf.global_variables_initializer()
    session.run(init_value)

    saver = tf.train.Saver()

    avg_loss = 0

    for time in range(train_times):

        minibatch_img, minibatch_mask = next(trainGen)
        #pred_image = session.run(predict_op, feed_dict={input_x:
        #minibatch_img, input_y: minibatch_mask})
        #print(pred_image)
        session.run(train_op, feed_dict={input_x: minibatch_img, input_y: minibatch_mask})
        avg_loss += session.run(loss_op, feed_dict={input_x: minibatch_img, input_y: minibatch_mask}) / train_step

        if (time + 1) % train_step == 0:
            testData = np.concatenate([imgData, minibatch_img[0].reshape(1, image_width, image_height, 3)])
            testMask = np.concatenate([imgMask, minibatch_mask[0].reshape(1, image_width, image_height * 3, 1)])



            summary_str = session.run(summary_op, feed_dict={input_x: testData, input_y: testMask})
            summary_writer.add_summary(summary_str, session.run(index))

            pred_image = session.run(real_op, feed_dict={input_x: testData, input_y: testMask})
            acc = session.run(accuracy_op, feed_dict={input_x: testData, input_y: testMask})

            avg_IOU = np.mean(IOU(pred_image, testMask))
            print("train times:", (time + 1),
                        " avg_loss:", np.mean(avg_loss),
                        " accuracy:", acc,
                        " avg_IOU:", avg_IOU)
            avg_loss = 0

    imgData, imgMask = next(testGene)
    imgData = np.concatenate([imgData, imgData])
    imgMask = np.concatenate([imgMask, imgMask])
    preMask = session.run(real_op, feed_dict={input_x:imgData})

    plt.figure(figsize=(12, 7))
    plt.subplot(2, 3, 1)
    plt.imshow(imgData[0].reshape(image_width,image_height, 3))
    plt.subplot(2, 3, 2)
    plt.imshow(imgMask[0].reshape(image_width,image_height, 3))
    plt.subplot(2, 3, 3)
    plt.imshow(preMask[0].reshape(image_width,image_height, 3))

    preMask = session.run(real_op, feed_dict={input_x:minibatch_img})
    plt.subplot(2, 3, 4)
    plt.imshow(minibatch_img[0].reshape(image_width,image_height, 3))
    plt.subplot(2, 3, 5)
    plt.imshow(minibatch_mask[0].reshape(image_width,image_height, 3))
    plt.subplot(2, 3, 6)
    plt.imshow(preMask[0].reshape(image_width,image_height, 3))

    plt.savefig('fnc_result.png')
    plt.show()

    session.close()