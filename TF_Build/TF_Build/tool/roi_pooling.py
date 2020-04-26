import tensorflow as tf

class ROIPooling():
    def __init__(self, pooled_height=7, pooled_width=7):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width

    def __call__(self, datas):
        def curried_pool_rois(x):
            batch_feature = x[0]
            batch_rois = x[1]
            return ROIPooling._pool_rois(batch_feature, batch_rois, 
                                            self.pooled_height, 
                                            self.pooled_width)
        
        pooled_areas = tf.map_fn(curried_pool_rois, datas, dtype=tf.float32)
        return pooled_areas

    
    @staticmethod
    def _pool_roi(feature, roi, pooled_height, pooled_width):
        feature_height = tf.cast(tf.shape(feature)[0], tf.float32)
        feature_width  = tf.cast(tf.shape(feature)[1], tf.float32)
        h_start = tf.cast(feature_height  * roi[0], 'int32')
        w_start = tf.cast(feature_width  * roi[1], 'int32')
        #h_end   = tf.reduce_min(tf.cast(feature_height * roi[2] + 1, 'int32'), tf.cast(feature_width - 1, 'int32'))
        #w_end   = tf.reduce_min(tf.cast(feature_width  * roi[3] + 1, 'int32'), tf.cast(feature_width - 1, 'int32'))
        h_end   = tf.cast(feature_height * roi[2], 'int32') + 1
        w_end   = tf.cast(feature_width  * roi[3], 'int32') + 1

        

        region = feature[h_start:h_end, w_start:w_end, :]

        region_height = h_end - h_start
        region_width  = w_end - w_start
        h_step = tf.cast( region_height / pooled_height, 'int32')
        w_step = tf.cast( region_width  / pooled_width , 'int32')
        
        areas = [[(
                    i*h_step, 
                    j*w_step, 
                    (i+1)*h_step if i+1 < pooled_height else region_height, 
                    (j+1)*w_step if j+1 < pooled_width else region_width
                    ) 
                    for j in range(pooled_width)] 
                    for i in range(pooled_height)]
        
        # take the maximum of each area and stack the result
        def pool_area(x):
            x2 = tf.cond(tf.equal(x[2], x[0]), lambda: x[0] + 1, lambda: x[2])
            x3 = tf.cond(tf.equal(x[3], x[1]), lambda: x[1] + 1, lambda: x[3])
            return tf.reduce_max(region[x[0]:x2, x[1]:x3, :], axis=[0,1])
        
        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])

        return pooled_features

    @staticmethod
    def _pool_rois(feature, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single image and varios ROIs
        """
        def curried_pool_roi(roi): 
            return ROIPooling._pool_roi(feature, roi, 
                                            pooled_height, pooled_width)
        
        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas
