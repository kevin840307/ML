import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as trans
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate = 0.0002
batch_size = 2
train_times = 1000
train_step = 10
 
# [kernel height, filter weight, input channel, out channel]
conv1_size1 = [3, 3, 1, 64]
conv1_size2 = [3, 3, 64, 64]
conv2_size1 = [3, 3, 64, 128]
conv2_size2 = [3, 3, 128, 128]
conv3_size1 = [3, 3, 128, 256]
conv3_size2 = [3, 3, 256, 256]
conv4_size1 = [3, 3, 256, 512]
conv4_size2 = [3, 3, 512, 512]
conv5_size1 = [3, 3, 512, 1024]
conv5_size2 = [3, 3, 1024, 1024]

conv6_size1 = [2, 2, 1024, 512]
conv6_size2 = [3, 3, 512, 512]
conv6_size3 = [3, 3, 512, 512]
conv7_size1 = [2, 2, 512, 256]
conv7_size2 = [3, 3, 256, 256]
conv7_size3 = [3, 3, 256, 256]
conv8_size1 = [2, 2, 256, 128]
conv8_size2 = [3, 3, 128, 128]
conv8_size3 = [3, 3, 128, 128]
conv9_size1 = [2, 2, 128, 64]
conv9_size2 = [3, 3, 64, 64]
conv9_size3 = [3, 3, 64, 2]

conv10_size1 = [1, 1, 2, 1]


def conv_batch_norm(x, n_out, train):
    beta = tf.get_variable("beta", [n_out], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
    gamma = tf.get_variable("gamma", [n_out], initializer=tf.constant_initializer(value=1.0, dtype=tf.float32))
    
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(train, mean_var_with_update, lambda:(ema_mean, ema_var))
    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, True)

    mean_hist = tf.summary.histogram("meanHistogram", mean)
    var_hist = tf.summary.histogram("varHistogram", var)
    return normed

def conv2d(input, weight_shape, activation='relu'):
    size = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weights_init = tf.random_normal_initializer(stddev=np.sqrt(2. / size))
    biases_init = tf.zeros_initializer()
    weights = tf.get_variable(name="weights", shape=weight_shape, initializer=weights_init)
    biases = tf.get_variable(name="biases", shape=weight_shape[3], initializer=biases_init)

    conv_out = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    conv_add = tf.nn.bias_add(conv_out, biases)
    conv_batch = conv_batch_norm(conv_add, weight_shape[3], tf.constant(True, dtype=tf.bool))

    if activation == 'relu':
        output = tf.nn.relu(conv_batch)
    elif activation == 'sigmoid':
        output = tf.nn.sigmoid(conv_batch)
    else:
        output = conv_batch

    return output

def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def upsampling(input, size):
    return tf.image.resize_images(input, size)


#def conv2d_transpose(input, size, filters, kernel_size=2):
#    with tf.variable_scope("transpose_scope"):
        #weights_init = tf.random_normal_initializer(stddev=np.sqrt(2. / filters ** 2 * kernel_size))
        #biases_init = tf.zeros_initializer()
        #weights = tf.get_variable(name="weights", shape=[filters, filters, kernel_size, size[3]], initializer=weights_init)
        #biases = tf.get_variable(name="biases", shape=size[3], initializer=biases_init)
        #conv_out = tf.nn.conv2d_transpose(value=input, filter=weights, output_shape=size, strides=[1, 2, 2, 1], padding='SAME')
        #conv_add = tf.nn.bias_add(conv_out, biases)
        #return conv_add


def conv2d_transpose(input, filters, kernel_size=2):
    with tf.variable_scope("transpose_scope"):
        conv2d_tran = tf.layers.conv2d_transpose(input, filters=filters, kernel_size=kernel_size, strides=2, padding='SAME')

        return conv2d_tran

def predict(x):
    with tf.variable_scope("conv1_scope"):
        with tf.variable_scope("conv1_1"):
            conv1_out = conv2d(x, conv1_size1)
        with tf.variable_scope("conv1_2"):
            conv1_out = conv2d(conv1_out, conv1_size2)
        pool1 = max_pool(conv1_out)

    with tf.variable_scope("conv2_scope"):
        with tf.variable_scope("conv2_1"):
            conv2_out = conv2d(pool1, conv2_size1)
        with tf.variable_scope("conv2_2"):
            conv2_out = conv2d(conv2_out, conv2_size2)
        pool2 = max_pool(conv2_out)

    with tf.variable_scope("conv3_scope"):
        with tf.variable_scope("conv3_1"):
            conv3_out = conv2d(pool2, conv3_size1)
        with tf.variable_scope("conv3_2"):
            conv3_out = conv2d(conv3_out, conv3_size2)
        pool3 = max_pool(conv3_out)

    with tf.variable_scope("conv4_scope"):
        with tf.variable_scope("conv4_1"):
            conv4_out = conv2d(pool3, conv4_size1)
        with tf.variable_scope("conv4_2"):
            conv4_out = conv2d(conv4_out, conv4_size2)
        pool4 = max_pool(conv4_out)

    with tf.variable_scope("conv5_scope"):
        with tf.variable_scope("conv5_1"):
            conv5_out = conv2d(pool4, conv5_size1)
        with tf.variable_scope("conv5_2"):
            conv5_out = conv2d(conv5_out, conv5_size2)



    # upmap
    with tf.variable_scope("conv6_scope"):
        with tf.variable_scope("conv6_1"):
            conv6_out = conv2d(conv5_out, conv6_size1)
        shape = conv4_out.get_shape()
        conv6_out = upsampling(conv6_out, [shape[1], shape[2]])
        #conv6_out = conv2d_transpose(conv6_out, shape[3])
        conv6_out = tf.add(conv4_out, conv6_out)
        with tf.variable_scope("conv6_2"):
            conv6_out = conv2d(conv6_out, conv6_size2)
        with tf.variable_scope("conv6_3"):
            conv6_out = conv2d(conv6_out, conv6_size3)

    with tf.variable_scope("conv7_scope"):
        with tf.variable_scope("conv7_1"):
            conv7_out = conv2d(conv6_out, conv7_size1)
        shape = conv3_out.get_shape()
        conv7_out = upsampling(conv7_out, [shape[1], shape[2]])
        #conv7_out = conv2d_transpose(conv7_out, shape[3])
        conv7_out = tf.add(conv3_out, conv7_out)
        with tf.variable_scope("conv7_2"):
            conv7_out = conv2d(conv7_out, conv7_size2)
        with tf.variable_scope("conv7_3"):
            conv7_out = conv2d(conv7_out, conv7_size3)

    with tf.variable_scope("conv8_scope"):
        with tf.variable_scope("conv8_1"):
            conv8_out = conv2d(conv7_out, conv8_size1)
        shape = conv2_out.get_shape()
        conv8_out = upsampling(conv8_out, [shape[1], shape[2]])
        #conv8_out = conv2d_transpose(conv8_out, shape[3])
        conv8_out = tf.add(conv2_out, conv8_out)
        with tf.variable_scope("conv8_2"):
            conv8_out = conv2d(conv8_out, conv8_size2)
        with tf.variable_scope("conv8_3"):
            conv8_out = conv2d(conv8_out, conv8_size3)

    with tf.variable_scope("conv9_scope"):
        with tf.variable_scope("conv9_1"):
            conv9_out = conv2d(conv8_out, conv9_size1)
        shape = conv1_out.get_shape()
        conv9_out = upsampling(conv9_out, [shape[1], shape[2]])
        #conv9_out = conv2d_transpose(conv9_out, shape[3])
        conv9_out = tf.add(conv1_out, conv9_out)
        with tf.variable_scope("conv9_2"):
            conv9_out = conv2d(conv9_out, conv9_size2)
        with tf.variable_scope("conv9_3"):
            conv9_out = conv2d(conv9_out, conv9_size3)

    with tf.variable_scope("conv10_scope"):
        conv10_out = conv2d(conv9_out, conv10_size1, activation=None)

    return conv10_out


def loss(y, t):
    #return tf.reduce_mean(tf.keras.losses.binary_crossentropy(
    #        t, y))

    #result = -tf.reduce_mean(t * tf.log(y + 1e-8) + (1 - t) * tf.log(1 - y + 1e-8))
    #return result

    #pred_flat = tf.reshape(y, [batch_size, -1])
    #true_flat = tf.reshape(t, [batch_size, -1])

    #intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    #denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-7
    #result = tf.reduce_mean(intersection / denominator)
    #loss_his = tf.summary.scalar("loss", result)
    #return result

    cross = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=t)
    result = tf.reduce_mean(cross)
    loss_his = tf.summary.scalar("loss", result)

    return result

def train(loss, index):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=index)
    #return tf.train.RMSPropOptimizer(learning_rate, decay=0.9).minimize(loss,
    #global_step=index)
    #return tf.train.AdagradOptimizer(learning_rate).minimize(loss,
    #global_step=index)
    #return tf.train.MomentumOptimizer(learning_rate,
    #momentum=0.9).minimize(loss, global_step=index)
    #return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
    #global_step=index)



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
def trainGenerator(batch_size,train_path,image_folder,mask_folder,image_color_mode="grayscale",
                    mask_color_mode="grayscale",image_save_prefix="image",mask_save_prefix="mask",
                    flag_multi_class=False,num_class=2,save_to_dir=None,target_size=(256,256),seed=1):
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
        mask = mask / 255.
        mask[mask > 0.5] = 1.
        mask[mask <= 0.5] = 0.
        yield (img, mask)
        #img,mask =
        #adjustData(img,mask,flag_multi_class,num_class)#返回的img依旧是[2,256,256]
        #yield (img,mask)

def testGenerator(test_path, image_folder, mask_folder,num_image=30,target_size=(256,256),flag_multi_class=False,as_gray=True):
    for index in range(num_image):
        imgData = io.imread(os.path.join(test_path + image_folder,"%d.png" % index),as_gray = as_gray)
        imgData = imgData / 255.
        imgData = trans.resize(imgData,target_size)
        imgData = np.reshape(imgData,imgData.shape + (1,)) if (not flag_multi_class) else imgData
        imgData = np.reshape(imgData, (1,) + imgData.shape)
        imgData = np.concatenate([imgData, imgData])

        imgLabel = io.imread(os.path.join(test_path + mask_folder,"%d_predict.png" % index),as_gray = as_gray)
        imgLabel = imgLabel / 255.
        imgLabel = trans.resize(imgLabel,target_size)
        imgLabel = np.reshape(imgLabel,imgLabel.shape + (1,)) if (not flag_multi_class) else imgLabel
        imgLabel = np.reshape(imgLabel,(1,) + imgLabel.shape)
        imgLabel = np.concatenate([imgLabel, imgLabel])

        yield (imgData, imgLabel)



if __name__ == '__main__':
    # init
    trainGen = trainGenerator(batch_size,'data/membrane/train','image','label',save_to_dir = 'data/membrane/train/aug')
    testGene = testGenerator('data/membrane/test/','image','label')
    imgData, imgMask = next(testGene)
    input_x = tf.placeholder(tf.float32, shape=[batch_size, 256, 256, 1], name="input_x")
    input_y = tf.placeholder(tf.float32, shape=[batch_size, 256, 256, 1], name="input_y")

    # predict
    predict_op = predict(input_x)

    # real
    real_op = tf.sigmoid(predict_op)
    
    # loss
    loss_op = loss(predict_op, input_y)

    # train
    index = tf.Variable(0, name="train_time")
    train_op = train(loss_op, index)
    
    # graph
    summary_op = tf.summary.merge_all()
    session = tf.Session()
    summary_writer = tf.summary.FileWriter("log/", graph=session.graph)

    init_value = tf.global_variables_initializer()
    session.run(init_value)

    saver = tf.train.Saver()

    avg_loss = 0

    for time in range(train_times):

        minibatch_img, minibatch_mask = next(trainGen)
        session.run(train_op, feed_dict={input_x: minibatch_img, input_y: minibatch_mask})
        avg_loss += session.run(loss_op, feed_dict={input_x: minibatch_img, input_y: minibatch_mask}) / train_step

        if (time + 1) % train_step == 0:
            summary_str = session.run(summary_op, feed_dict={input_x: imgData, input_y: imgMask})
            summary_writer.add_summary(summary_str, session.run(index))
            print("train times:", (time + 1),
                        " avg_loss:", avg_loss)
            avg_loss = 0

    preMask = session.run(real_op, feed_dict={input_x:imgData})
    plt.subplot(2, 3, 1)
    plt.imshow(imgData[0].reshape(256, 256), cmap=plt.get_cmap('gray'))
    plt.subplot(2, 3, 2)
    plt.imshow(imgMask[0].reshape(256, 256), cmap=plt.get_cmap('gray'))
    plt.subplot(2, 3, 3)
    plt.imshow(preMask[0].reshape(256, 256), cmap=plt.get_cmap('gray'))

    preMask = session.run(real_op, feed_dict={input_x:minibatch_img})
    plt.subplot(2, 3, 4)
    plt.imshow(minibatch_img[0].reshape(256, 256), cmap=plt.get_cmap('gray'))
    plt.subplot(2, 3, 5)
    plt.imshow(minibatch_mask[0].reshape(256, 256), cmap=plt.get_cmap('gray'))
    plt.subplot(2, 3, 6)
    plt.imshow(preMask[0].reshape(256, 256), cmap=plt.get_cmap('gray'))

    plt.savefig('fnc_result.png')
    plt.show()

    session.close()