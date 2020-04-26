import numpy as np
import tensorflow as tf

import tensorflow_tools as tf_tools
from module import Model

from generate_anchors import generate_anchors_pre
from proposal_layer import proposal_layer
from anchor_target_layer import anchor_target_layer
from proposal_target_layer import proposal_target_layer
from bbox_transform import bbox_transform_inv
from cython_bbox import bbox_overlaps

def one_hot_box_transform(pred_class, bbox_pred):
    bboxs = []
    for label, bbox in zip(pred_class, bbox_pred):
        bboxs.append(bbox[label * 4:(label + 1) * 4])
    return np.array(bboxs)

def _reshape_layer(bottom, num_dim, name, _batch_size=1):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name):
        # change the channel to the caffe format
        to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
        # then force it to have channel 2
        reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[_batch_size], [num_dim, -1], [input_shape[2]]]))
        # then swap the channel back
        to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
        return to_tf

def _softmax_layer(bottom, name):
    if name == 'rpn_cls_prob_reshape':
        input_shape = tf.shape(bottom)
        bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
        reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
        return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
        out_loss_box,
        axis=dim
    ))
    return loss_box

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

class FasterRCNN(Model):
    def __init__(self, scope_name, class_num, channel_min, channel_rate=2, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        super(FasterRCNN, self).__init__(scope_name)
        self.class_num = class_num
        self.scope_name = scope_name
        self.channel_min = channel_min
        self.channel_rate = channel_rate
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._predictions = {}
        self._losses = {}

        self._feat_stride = [16,]

        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios

    def buind(self, train_fn=tf_tools.adam_fn, lr=1e-5):
        self.input_img = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_img")
        self.input_labels = tf.placeholder(tf.float32, shape=[None, 5], name="input_labels")

        self.rois, self.cls_score, self.cls_prob, self.bbox_pred = self.buind_network()
        self.index = tf.Variable(0)
        self.train_fn = train_fn(lr)
        self.build_optimization()
           
    def buind_network(self):
        net = self.buind_head()

        rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(net)

        proposal_rois, rois = self.build_proposals(rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)

        cls_score, cls_prob, bbox_pred = self.build_predictions(net, rois)

        pred_cls_score, pred_cls_prob, pred_bbox_pred = self.build_predictions(net, proposal_rois)

        stds = np.tile((0.1, 0.1, 0.1, 0.1), (self.class_num))
        means = np.tile((0., 0., 0., 0.), (self.class_num))
        pred_bbox_pred *= stds
        pred_bbox_pred += means

        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["cls_score"] = cls_score
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred
        self._predictions["rois"] = rois


        return proposal_rois, pred_cls_score, pred_cls_prob, pred_bbox_pred

    def buind_head(self):
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
            return output

    def build_rpn(self, net):
        self._anchor_component()

        rpn = tf_tools.conv2d(net, out_channel=512, activation='relu', name='rpn_conv')
        rpn_cls_score = tf_tools.conv2d(rpn, out_channel=self._num_anchors * 2, kernel_size=1, pad='VALID', name='rpn_cls_score')

        rpn_cls_score_reshape = _reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = _softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_prob = _reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = tf_tools.conv2d(rpn, out_channel=self._num_anchors * 4, kernel_size=1, pad='VALID', name='rpn_bbox_pred')
        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + 'default'):
            # just to get the shape right
            height = tf.to_int32(tf.ceil(tf.shape(self.input_img)[1] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(tf.shape(self.input_img)[2] / np.float32(self._feat_stride[0])))
            anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                [height, width,
                                                 self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                                [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

    def build_proposals(self, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):

        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        proposal_rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois", mode='test')

        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")

        # Try to have a deterministic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
            rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")

        return proposal_rois, rois

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name, mode='train'):
        with tf.variable_scope(name):
            im_info = [tf.shape(self.input_img)[1], tf.shape(self.input_img)[2]]
            rois, rpn_scores = tf.py_func(proposal_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, im_info,
                                           self._feat_stride, self._anchors, self._num_anchors, mode],
                                          [tf.float32, tf.float32])
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    def _anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name):
            im_info = [tf.shape(self.input_img)[1], tf.shape(self.input_img)[2]]
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self.input_labels, im_info, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name):
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self.input_labels, self.class_num],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

            rois.set_shape([256, 5])
            roi_scores.set_shape([256])
            labels.set_shape([256, 1])
            bbox_targets.set_shape([256, self.class_num * 4])
            bbox_inside_weights.set_shape([256, self.class_num * 4])
            bbox_outside_weights.set_shape([256, self.class_num * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            return rois, roi_scores

    def build_predictions(self, net, rois):

        # Crop image ROIs
        pool5 = self._crop_pool_layer(net, rois, "pool5")
        pool5_flat = tf.layers.flatten(pool5, name='flatten')

        with tf.variable_scope("vgg_16", reuse=tf.AUTO_REUSE):
            fc6 = tf_tools.layer(pool5_flat, weights_shape=[7, 7, 512, 4096], activation='relu', name='fc6/')
            fc7 = tf_tools.layer(fc6, weights_shape=[1, 1, 4096, 4096], activation='relu', name='fc7/')


        # Scores and predictions
        with tf.variable_scope("predictions", reuse=tf.AUTO_REUSE):
            cls_score = tf_tools.layer(fc7, out_size=self.class_num, name='cls_score')
            cls_prob = _softmax_layer(cls_score, "cls_prob")
            bbox_prediction = tf_tools.layer(fc7, out_size=self.class_num * 4, name='bbox_pred')

        return cls_score, cls_prob, bbox_prediction

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name):
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be backpropagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = 7 * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
        return tf_tools.max_pool(crops)

    def build_optimization(self, sigma_rpn=3.0):
        with tf.variable_scope('loss'):
            # RPN, class loss
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

            rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            # RCNN, class loss
            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])

            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.reshape(cls_score, [-1, self.class_num]), labels=label))

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

            loss_box = _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            self.loss_op = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            self.train_op = self.train_fn.minimize(self.loss_op, global_step=self.index)
        
    def predict(self, img):
        session = tf.get_default_session()
        feed_dict = {self.input_img: img}
        rois, cls_prob, bbox_pred = session.run([self.rois, self.cls_prob, self.bbox_pred], feed_dict=feed_dict)
        pred_class = np.argmax(cls_prob, axis=-1)
        boxes = rois[:, 1:5]
        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
        pred_boxes = bbox_transform_inv(boxes, bbox_pred)
        pred_boxes = _clip_boxes(pred_boxes, img.shape[1:-1])
        pred_boxes = one_hot_box_transform(pred_class, pred_boxes)
        return pred_class, pred_boxes.astype(np.int)

    
    def train(self, img, labels):
        session = tf.get_default_session()
        feed_dict = {self.input_img: img, self.input_labels: labels}
        session.run(self.train_op, feed_dict=feed_dict)
        return self.loss(img, labels)
    
    def loss(self, img, labels):
        session = tf.get_default_session()
        feed_dict = {self.input_img: img, self.input_labels: labels}
        loss = session.run(self.loss_op, feed_dict=feed_dict)
        return loss

    def test(self, img, labels):
        session = tf.get_default_session()
        feed_dict = {self.input_img: img, self.input_labels: labels}
        num1, num2 = session.run([self.num1, self.num2], feed_dict=feed_dict)
        return num1, num2

if __name__ == '__main__':
    import numpy as np

    from voc_dataset import VOCDataset, bbox_visualization
    from tool import array_img_save

    dataset = VOCDataset()
    network = FasterRCNN('FasterRCNN', 21, 64)
    network.buind()
    network.complete()
    #restorer_fc = tf.train.Saver(tf.trainable_variables(scope='vgg_16'))
    #session = tf.get_default_session()
    #restorer_fc.restore(session, "./VGG/vgg_16.ckpt")

    for step in range(100000):
        images, labels, _ = dataset.get_minbatch(1, step, selectivesearch=False)
        labels = np.reshape(labels, (-1, 5))
        
        loss = network.train(images, labels)
        if step % 5 == 0:
            print(loss)
            pred_class, pred_rect = network.predict(images)
            print(pred_class)
            pred_class = np.reshape(pred_class, (-1)).astype(np.int)
            img = bbox_visualization(images[0], pred_rect, pred_class)
            img = np.array(img)
            array_img_save(img, "./save/" + str(step) + ".tif", binary=False, normal=False)
