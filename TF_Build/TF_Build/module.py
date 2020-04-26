import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])


import tensorflow as tf
import tensorflow_tools as tf_tools

class Model(object):
    def __init__(self, scope_name):
        self.scope_name = scope_name
        self.train_fn = tf_tools.adam_fn()
        
    def clear(self):
        session = tf.get_default_session()
        if not session == None: 
            session.close()
        tf.reset_default_graph()
        
    def complete(self):
        session = tf.get_default_session()
        if session == None:
            graph = tf.get_default_graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.InteractiveSession(config=config, graph=graph)
            init_value = tf.global_variables_initializer()
            session.run(init_value)
        else:
            init_value = tf.variables_initializer(tf.trainable_variables(scope=self.scope_name))
            init_optim = tf.variables_initializer(self.train_fn.variables())
            #init_value = tf.global_variables_initializer()
            session.run([init_value, init_optim])

    def copy_weights(self, src, des):
        session = tf.get_default_session()
        sync_op = get_sync_op(from_list=src.var_list, to_list=des.var_list)
        session.run(sync_op)