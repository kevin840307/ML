

import numpy as np
import matplotlib.pyplot as plt
width = 5
height = 5
kernel = 3
dilated_rate = 1
dilated_kernel = (kernel + (kernel - 1) * dilated_rate)
pad = (dilated_kernel - 1) >> 1


matrix = np.zeros((height + (dilated_kernel - 1), width + (dilated_kernel - 1)))
matrix[pad:-pad, pad:-pad] = np.arange(1, height * width + 1).reshape((height, width))
#print(matrix)
wedith = np.arange(1, kernel * kernel + 1).reshape((kernel, kernel))
dilated_wedith = np.zeros((dilated_kernel, dilated_kernel))
dilated_wedith[0::dilated_rate + 1, 0::dilated_rate + 1] = 1#wedith
result = np.zeros((height, width))
count = np.zeros_like(matrix)
for i in range(height):
    for k in range(width):
        result[i, k] = np.sum(matrix[i:i + dilated_kernel, k:k + dilated_kernel] * dilated_wedith)
        count[i:i + dilated_kernel:dilated_rate + 1, k:k + dilated_kernel: dilated_rate + 1] += 1
#print(result)
print(count[pad:-pad, pad:-pad])

plt.imshow(count[pad:-pad, pad:-pad], cmap='hot', interpolation='nearest')
plt.show()

width = 5
height = 5
kernel = 3
dilated_rate = 0
dilated_kernel = (kernel + (kernel - 1) * dilated_rate)
pad = (dilated_kernel - 1) >> 1


matrix = np.zeros((height + (dilated_kernel - 1), width + (dilated_kernel - 1)))
matrix[pad:-pad, pad:-pad] = np.arange(1, height * width + 1).reshape((height, width))
#print(matrix)
wedith = np.arange(1, kernel * kernel + 1).reshape((kernel, kernel))
dilated_wedith = np.zeros((dilated_kernel, dilated_kernel))
dilated_wedith[0::dilated_rate + 1, 0::dilated_rate + 1] = 1#wedith
result = np.zeros((height, width))
count = np.zeros_like(matrix)
for i in range(height):
    for k in range(width):
        result[i, k] = np.sum(matrix[i:i + dilated_kernel, k:k + dilated_kernel] * dilated_wedith)
        count[i:i + dilated_kernel:dilated_rate + 1, k:k + dilated_kernel: dilated_rate + 1] += 1
#print(result)
print(count[pad:-pad, pad:-pad])

plt.imshow(count[pad:-pad, pad:-pad], cmap='hot', interpolation='nearest')
plt.show()


from tool.metric import BB_IOU

print(BB_IOU([0, 0, 128, 128], [20, 20, 120, 120]))
import dataset.voc_dataset as voc

dataset = voc.VOCDataset()
dataset.build()
images, rois, labels = dataset.get_minbatch(1, 0, selectivesearch=True)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

anchors=[[ -84,  -40,   99,   55,],
 [-176,  -88,  191,  103,],
 [-360, -184,  375,  199,],
 [ -56,  -56,   71,   71,],
 [-120, -120,  135,  135,],
 [-248, -248,  263,  263,],
 [ -36,  -80,   51,   95,],
 [ -80, -168,   95,  183,],
 [-168, -344,  183,  359,]]

fig, ax = plt.subplots(1)
ax.set(xlim=(-400, 400), ylim=(-400, 400))
colors = ['y', 'r', 'b']
index = 0
for anchor in anchors:
    color = colors[index // 3]
    ax.add_patch(patches.Rectangle((anchor[0], anchor[1]), anchor[2] - anchor[0] , anchor[3] - anchor[1], fill=False, color=color))
    index += 1
#plt.show()

import numpy as np
v = np.ones((1, 5, 2))
v[:, :, 0] = 2
print(np.reshape(v, (1, 10)))

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    print(ratio_anchors)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'
    """
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))

    A = anchors.shape[0]
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])

    return anchors, length

import tensorflow as tf
def _reshape_layer(bottom, num_dim, name):
    input_shape = bottom.get_shape().as_list()
    with tf.variable_scope(name):
        # change the channel to the caffe format
        to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
        print(to_caffe.shape)
        # then force it to have channel 2
        reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[1], [num_dim, -1], [input_shape[2]]]))
        print(reshaped.shape)
        # then swap the channel back
        to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
        print(to_tf.shape)
        return to_tf

def _softmax_layer(bottom, name):
    if name == 'rpn_cls_prob_reshape':
        input_shape = tf.shape(bottom)
        bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
        reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
        return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

if __name__ == '__main__':
    import time
    input_img = tf.placeholder(shape=[1, 1, 1, 18], dtype=tf.float32)
    result = _reshape_layer(input_img, 2, 'test')
    result = _softmax_layer(result, 'test2')
    result2 = _reshape_layer(result, 18, "rpn_cls_prob")

    value = np.ones((1, 1, 1, 18))
    value[:, :, :, 0:9] = 0.1
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    result = session.run(result2, feed_dict={input_img:value})
    print(result[:, :, :, 9:])

    #t = time.time()
    #a = generate_anchors()
    #print(time.time() - t)
    ##print(a)

    #a = generate_anchors_pre(np.ceil(800 / 16), np.ceil(600 / 16), [16, ])
    ##print(a[0].shape)
    ##print(a[0][9:18])
    ##print(a)
    #value = np.random.randint(0, 100, 100)
    #print(value)
    #print(np.where(value <= 20)[0])
    #inds = np.where(value <= 20)[0]


#value1 = np.ones(shape=(3, 7 * 7 * 512))
#value2 = np.ones(shape=(7 * 7 * 512, 4096))
#print(np.matmul(value1, value2).shape)
#dataset = VOCDataset()
#images, rois, rectangles, labels = dataset.get_minbatch(1, 0, selectivesearch=True)
#input_img = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)
#input_roi = tf.placeholder(tf.float32, shape=[None, None, 4], name="input_roi")
#output = input_img
#with tf.variable_scope("vgg_16"):
#    with tf.variable_scope("conv1"):
#        output = tf_tools.conv2d(output, out_channel=64, name='conv1_1/')
#        output = tf_tools.conv2d(output, out_channel=64, name='conv1_2/')
#    output = tf_tools.max_pool(output, name='pool1')

#    with tf.variable_scope("conv2"):
#        output = tf_tools.conv2d(output, out_channel=128, name='conv2_1/')
#        output = tf_tools.conv2d(output, out_channel=128, name='conv2_2/')
#    output = tf_tools.max_pool(output, name='pool2')

#    with tf.variable_scope("conv3"):
#        output = tf_tools.conv2d(output, out_channel=256, name='conv3_1/')
#        output = tf_tools.conv2d(output, out_channel=256, name='conv3_2/')
#    output = tf_tools.max_pool(output, name='pool3')

#    with tf.variable_scope("conv4"):
#        output = tf_tools.conv2d(output, out_channel=512, name='conv4_1/')
#        output = tf_tools.conv2d(output, out_channel=512, name='conv4_2/')
#    output = tf_tools.max_pool(output, name='pool4')

#    with tf.variable_scope("conv5"):
#        output = tf_tools.conv2d(output, out_channel=512, name='conv5_1/')
#        output = tf_tools.conv2d(output, out_channel=512, name='conv5_2/')
#        output = tf_tools.conv2d(output, out_channel=512, name='conv5_3/')

#    output = ROIPooling(pooled_height=7, pooled_width=7)([output, input_roi])
#    output = tf.reshape(output, [-1, 7 * 7 * 512], name='flatten')
#    output = tf_tools.layer(output, weights_shape=[7, 7, 512, 4096], name='fc6/')
#    output = tf_tools.layer(output, weights_shape=[1, 1, 4096, 4096], name='fc7/')

#value = tf.trainable_variables(scope='vgg_16')
##print(value)
##conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
#session = tf.Session()
#session.run(tf.global_variables_initializer())
#restorer_fc = tf.train.Saver(value)
#restorer_fc.restore(session, "./VGG/vgg_16.ckpt")



#import numpy as np
#dataset = VOCDataset()
#images, rois, rectangles, labels = dataset.get_minbatch(1, 0, selectivesearch=True)
#rois = [0.028,0.02040816,0.97,0.02040816]
#input_img = tf.placeholder(tf.float32, shape=[1, 343, 500, 3], name="input_img")
#input_roi = tf.placeholder(tf.float32, shape=[4], name="input_roi")
#output = tf_tools.max_pool(input_img, name='pool1')
#output = tf_tools.max_pool(output, name='pool2')
#output = tf_tools.max_pool(output, name='pool3')
#output = tf_tools.max_pool(output, name='pool4')

#roi = [0.028,0.02040816,0.97,0.02040816]
#feature_height = output.get_shape().as_list()[2]
#feature_width = output.get_shape().as_list()[1]
#h_start = int(feature_height  * roi[0])
#w_start = int(feature_width  * roi[1])
#h_end   = min(int(feature_height * roi[2]) + 1, feature_width - 1)
#w_end   = min(int(feature_width  * roi[3]) + 1, feature_height - 1)



#pooled_height = 7
#pooled_width = 7
#session = tf.Session()
#session.run(tf.global_variables_initializer())
#feed_dict = {input_img: images, input_roi: rois}
#feature = session.run(output, feed_dict=feed_dict)
#feature = np.array(feature[0])
#region = feature[h_start:h_end, w_start:w_end, :]

#region_height = h_end - h_start
#region_width  = w_end - w_start
#h_step = int( region_height / pooled_height)
#w_step = int( region_width  / pooled_width)
#print(region_height, ' ', region_width)
#areas = [[(
#            i*h_step, 
#            j*w_step, 
#            (i+1)*h_step if i + 1 < pooled_height else region_height, 
#            (j+1)*w_step if j + 1 < pooled_width else region_width
#            ) 
#            for j in range(pooled_width)] 
#            for i in range(pooled_height)]
#areas = np.array(areas)

#print(areas)
## take the maximum of each area and stack the result
#def pool_area(x):
#    x[2] = max(x[2], x[0] + 1)
#    x[3] = max(x[3], x[1] + 1)
#    return np.max(region[x[0]:x[2], x[1]:x[3], :], axis=(0,1))
        
#pooled_features = np.stack([[pool_area(x) for x in row] for row in areas])
#print(pooled_features)


#value1 = tf.Variable([[0], [1], [2]], dtype=tf.float32)
#value2 = tf.Variable([[3., 2., 1.], [2., 3., 1.], [3., 2., 4.]], dtype=tf.float32)
#cross = tf_tools.sparse_softmax_cross_entropy_with_logits(value2, value1)
#session = tf.Session()
#session.run(tf.global_variables_initializer())
#result = session.run(cross)
#print(result)