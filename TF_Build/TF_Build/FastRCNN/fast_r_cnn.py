import tensorflow as tf
import numpy as np 

import tool.tensorflow_tools as tf_tools
from module import Model
from tool.roi_pooling import ROIPooling
from tool.block import cnn_block, small_se_res_block


def focal_loss(logits, labels, alpha, epsilon = 1e-7, gamma=2.0, multi_dim = False):
    '''
    :param logits:  [batch_size, n_class]
    :param labels: [batch_size]  not one-hot !!!
    :return: -alpha*(1-y)^r * log(y)
    它是在哪实现 1- y 的？ 通过gather选择的就是1-p,而不是通过计算实现的；
    logits soft max之后是多个类别的概率，也就是二分类时候的1-P和P；多分类的时候不是1-p了；

    怎么把alpha的权重加上去？
    通过gather把alpha选择后变成batch长度，同时达到了选择和维度变换的目的

    是否需要对logits转换后的概率值进行限制？
    需要的，避免极端情况的影响

    针对输入是 (N，P，C )和  (N，P)怎么处理？
    先把他转换为和常规的一样形状，（N*P，C） 和 （N*P,）

    bug:
    ValueError: Cannot convert an unknown Dimension to a Tensor: ?
    因为输入的尺寸有时是未知的，导致了该bug,如果batchsize是确定的，可以直接修改为batchsize

    '''


    if multi_dim:
        logits = tf.reshape(logits, [-1, logits.shape[2]])
        labels = tf.reshape(labels, [-1])

    # (Class ,1)
    alpha = tf.constant(alpha, dtype=tf.float32)

    labels = tf.cast(labels, dtype=tf.int32)
    logits = tf.cast(logits, tf.float32)
    # (N,Class) > N*Class
    softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
    # (N,) > (N,) ,但是数值变换了，变成了每个label在N*Class中的位置
    labels_shift = tf.range(0, tf.shape(logits)[0]) * tf.shape(logits)[1] + labels
    #labels_shift = tf.range(0, batch_size*32) * logits.shape[1] + labels
    # (N*Class,) > (N,)
    prob = tf.gather(softmax, labels_shift)
    # 预防预测概率值为0的情况  ; (N,)
    prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
    # (Class ,1) > (N,)
    alpha_choice = tf.gather(alpha, labels)
    # (N,) > (N,)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    weight = tf.multiply(alpha_choice, weight)
    # (N,) > 1
    loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
    return loss

def smooth_l1_loss(bbox_pred, bbox_targets, mask):
    box_diff = bbox_pred - bbox_targets
    in_box_diff = mask * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1)))
    in_loss_box = tf.pow(in_box_diff, 2) * 0.5 * smoothL1_sign + (abs_in_box_diff - 0.5) * (1. - smoothL1_sign)
    out_loss_box = mask * in_loss_box
    sum = tf.reduce_sum(out_loss_box, axis=1)
    loss_box = tf.reduce_mean(sum)
    return loss_box

class FastRCNN(Model):
    def __init__(self, scope_name, class_num, channel_min, channel_rate=2):
        super(FastRCNN, self).__init__(scope_name)
        self.class_num = class_num
        self.scope_name = scope_name
        self.channel_min = channel_min
        self.channel_rate = channel_rate

    def buind(self, train_fn=tf_tools.adam_fn, lr=1e-5):
        self.input_img = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_img")
        self.input_roi = tf.placeholder(tf.float32, shape=[None, None, 4], name="input_roi")
        self.input_labels = tf.placeholder(tf.float32, shape=[None, 5], name="input_labels")

        self.bbox_logit, self.class_logit, self.class_act = self.buind_network()
        self.var_list = tf.trainable_variables(scope=self.scope_name)
        self.index = tf.Variable(0)
        self.train_fn = train_fn(lr)
        self.build_optimization()
           
    def buind_network(self):
        with tf.variable_scope("vgg_16"):
            with tf.variable_scope("conv1"):
                output = tf_tools.conv2d(self.input_img, out_channel=64, activation='relu', name='conv1_1/')
                output = tf_tools.conv2d(output, out_channel=64, activation='relu', name='conv1_2/')
            output = tf_tools.max_pool(output, name='pool1')

            with tf.variable_scope("conv2"):
                output = tf_tools.conv2d(output, out_channel=128, activation='relu', name='conv2_1/')
                output = tf_tools.conv2d(output, out_channel=128, activation='relu', name='conv2_2/')
            output = tf_tools.max_pool(output, name='pool2')

            with tf.variable_scope("conv3"):
                output = tf_tools.conv2d(output, out_channel=256, activation='relu', name='conv3_1/')
                output = tf_tools.conv2d(output, out_channel=256, activation='relu', name='conv3_2/')
            output = tf_tools.max_pool(output, name='pool3')

            with tf.variable_scope("conv4"):
                output = tf_tools.conv2d(output, out_channel=512, activation='relu', name='conv4_1/')
                output = tf_tools.conv2d(output, out_channel=512, activation='relu', name='conv4_2/')
            output = tf_tools.max_pool(output, name='pool4')

            with tf.variable_scope("conv5"):
                output = tf_tools.conv2d(output, out_channel=512, activation='relu', name='conv5_1/')
                output = tf_tools.conv2d(output, out_channel=512, activation='relu', name='conv5_2/')
                output = tf_tools.conv2d(output, out_channel=512, activation='relu', name='conv5_3/')

            output = ROIPooling(pooled_height=7, pooled_width=7)([output, self.input_roi])
            output = tf.reshape(output, [-1, 7 * 7 * 512], name='flatten')
            #output = tf_tools.layer(output, weights_shape=[7, 7, 512, 4096], activation='relu', name='fc6/')
            #output = tf_tools.layer(output, weights_shape=[1, 1, 4096, 4096], activation='relu', name='fc7/')
        with tf.variable_scope(self.scope_name) as scope:
            layer1 = tf_tools.layer(output, out_size=512, activation='relu', name='layer1')
            class_logit = tf_tools.layer(layer1, out_size=self.class_num, name='logit')
            class_act = tf.argmax(class_logit, -1)
            class_act = tf.expand_dims(class_act, -1)
            class_act = tf.reshape(class_act, shape=[-1, 1])
            class_act = tf.cast(class_act, tf.float32)

            bbox_logit = tf_tools.layer(layer1, out_size=4, name='rect')

            return bbox_logit, class_logit, class_act
            
    def build_optimization(self):
        class_label = tf.expand_dims(self.input_labels[:, -1], axis=-1)
        class_label = tf.cast(class_label, tf.int32)
        class_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(class_label, squeeze_dims=[1]), logits=self.class_logit)
        self.class_loss_op = tf.reduce_mean(class_sum)
        #alpha=np.ones((21))
        #alpha[0] = 1
        #self.class_loss_op = focal_loss(self.class_logit, self.input_label, alpha)

        mask = 1. - tf.cast(tf.equal(class_label, 0), tf.float32)
        bbox_label = self.input_labels[:, 0:4]
        self.rect_loss_op = smooth_l1_loss(self.bbox_logit, bbox_label, mask)
        #rect_sum = tf.reduce_sum(tf.square(bbox_label * mask - self.bbox_logit * mask) * mask, axis=-1)
        #self.rect_loss_op = tf.reduce_mean(rect_sum)

        self.loss_op = self.class_loss_op + self.rect_loss_op
        self.train_op = self.train_fn.minimize(self.loss_op, global_step=self.index)
        
    def predict(self, img, roi):
        session = tf.get_default_session()
        feed_dict = {self.input_img: img, self.input_roi: roi}
        out_class, out_rect = session.run([self.class_act, self.bbox_logit], feed_dict=feed_dict)
        return out_class, out_rect
    
    def train(self, img, roi, labels):
        session = tf.get_default_session()
        feed_dict = {self.input_img: img, self.input_roi: roi, self.input_labels: labels}
        session.run(self.train_op, feed_dict=feed_dict)
        return self.loss(img, roi, labels)
    
    def loss(self, img, roi, labels):
        session = tf.get_default_session()
        feed_dict = {self.input_img: img, self.input_roi: roi, self.input_labels: labels}
        class_loss, rect_loss = session.run([self.class_loss_op, self.rect_loss_op], feed_dict=feed_dict)
        return class_loss, rect_loss

    def test(self, img, roi, labels):
        session = tf.get_default_session()
        feed_dict = {self.input_img: img, self.input_roi: roi, self.input_labels: labels}
        tests = session.run(self.bbox_logit, feed_dict=feed_dict)
        return tests

if __name__ == '__main__':
    import numpy as np

    from dataset.voc_dataset import VOCDataset, bbox_visualization, bbox_transform_inv, bbox_transform, bb_norm, bb_denorm
    from tool.tool import array_img_save

    dataset = VOCDataset()
    network = FastRCNN('FastRCNN', 21, 64)
    network.buind()
    network.complete()
    restorer_fc = tf.train.Saver(tf.trainable_variables(scope='vgg_16'))
    session = tf.get_default_session()
    restorer_fc.restore(session, "./VGG/vgg_16.ckpt")
    
    for step in range(100000):
        images, labels, rois = dataset.get_minbatch(1, step % 3, selectivesearch=True)
        labels = np.reshape(labels, (-1, 5))
        labels[:, 0:4] = bbox_transform_inv(rois[0], labels[:, 0:4])
        rois[0] = bb_norm(images[0].shape[1], images[0].shape[0], rois[0])

        class_loss, rect_loss = network.train(images, rois, labels)
        if step % 5 == 0:
            print(class_loss, '   ', rect_loss)
            pred_class, pred_rect = network.predict(images, rois)
            rois[0] = bb_denorm(images[0].shape[1], images[0].shape[0], rois[0])
            pred_class = np.reshape(pred_class, (-1)).astype(np.int)
            img = bbox_visualization(images[0], bbox_transform(rois[0], pred_rect), pred_class)
            img = np.array(img)
            array_img_save(img, "./save/" + str(step) + ".tif", binary=False, normal=False)