import tensorflow as tf

class Model:
    def __init__(self, nchannels, imcropsize, reslearn):
        # --------------------------------------------------
        # parameters
        # --------------------------------------------------

        self.nchannels = nchannels
        self.imcropsize = imcropsize
        self.ks = 3
        self.nout0 = 16
        self.nout1 = 2*self.nout0
        self.nout2 = 2*self.nout1
        self.nout3 = 2*self.nout2
        self.noutFC1 = 1024
        self.noutFC2 = 128

        self.pixelIntensityNormalization = True
        self.localResponseNormalization = False
        self.networkInNetwork = False
        self.residualLearning = reslearn

        # --------------------------------------------------
        # shared variables
        # --------------------------------------------------

        # shared weights for convolution+fully_connected part of network
        # https://www.tensorflow.org/versions/r0.11/how_tos/variable_scope/index.html
        self.variables_dict = {
            "l0_bn_offset": tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nchannels]),name="l0_bn_offset"),
            "l0_bn_scale": tf.Variable(tf.constant(1.0, shape=[1,1,1,self.nchannels]),name="l0_bn_scale"),
            "l0_weights1": tf.Variable(tf.truncated_normal([self.ks, self.ks, 1, self.nout0], stddev=0.01), name="l0_weights1"),
            "l0_weights2": tf.Variable(tf.truncated_normal([self.ks, self.ks, self.nout0, self.nout0], stddev=0.01), name="l0_weights1"),
            "l0_biases": tf.Variable(tf.constant(0.01, shape=[self.nout0]), name="l0_biases"),
            "l0_shortcut": tf.Variable(tf.truncated_normal([1, 1, 1, self.nout0], stddev=0.01), name="l0_shortcut"),
            
            "l1_bn_offset": tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nout0]),name="l1_bn_offset"),
            "l1_bn_scale": tf.Variable(tf.constant(1.0, shape=[1,1,1,self.nout0]),name="l1_bn_scale"),
            "l1_1x1_weights": tf.Variable(tf.truncated_normal([1, 1, self.nout0, self.nout0], stddev=0.01), name="l1_1x1_weights"),
            "l1_weights1": tf.Variable(tf.truncated_normal([self.ks, self.ks, self.nout0, self.nout1], stddev=0.01), name="l1_weights1"),
            "l1_weights2": tf.Variable(tf.truncated_normal([self.ks, self.ks, self.nout1, self.nout1], stddev=0.01), name="l1_weights2"),
            "l1_biases": tf.Variable(tf.constant(0.01, shape=[self.nout1]), name="l1_biases"),
            "l1_shortcut": tf.Variable(tf.truncated_normal([1, 1, self.nout0, self.nout1], stddev=0.01), name="l1_shortcut"),
            
            "l2_bn_offset": tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nout1]),name="l2_bn_offset"),
            "l2_bn_scale": tf.Variable(tf.constant(1.0, shape=[1,1,1,self.nout1]),name="l2_bn_scale"),
            "l2_1x1_weights": tf.Variable(tf.truncated_normal([1, 1, self.nout1, self.nout1], stddev=0.01), name="l2_1x1_weights"),
            "l2_weights1": tf.Variable(tf.truncated_normal([self.ks, self.ks, self.nout1, self.nout2], stddev=0.01), name="l2_weights1"),
            "l2_weights2": tf.Variable(tf.truncated_normal([self.ks, self.ks, self.nout2, self.nout2], stddev=0.01), name="l2_weights2"),
            "l2_biases": tf.Variable(tf.constant(0.01, shape=[self.nout2]), name="l2_biases"),
            "l2_shortcut": tf.Variable(tf.truncated_normal([1, 1, self.nout1, self.nout2], stddev=0.01), name="l2_shortcut"),

            "l3_bn_offset": tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nout2]),name="l3_bn_offset"),
            "l3_bn_scale": tf.Variable(tf.constant(1.0, shape=[1,1,1,self.nout2]),name="l3_bn_scale"),
            "l3_1x1_weights": tf.Variable(tf.truncated_normal([1, 1, self.nout2, self.nout2], stddev=0.01), name="l3_1x1_weights"),
            "l3_weights1": tf.Variable(tf.truncated_normal([self.ks, self.ks, self.nout2, self.nout3], stddev=0.01), name="l3_weights1"),
            "l3_weights2": tf.Variable(tf.truncated_normal([self.ks, self.ks, self.nout3, self.nout3], stddev=0.01), name="l3_weights2"),
            "l3_biases": tf.Variable(tf.constant(0.01, shape=[self.nout3]), name="l3_biases"),
            "l3_shortcut": tf.Variable(tf.truncated_normal([1, 1, self.nout2, self.nout3], stddev=0.01), name="l3_shortcut"),
            
            "l4_bn_offset": tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nout3]),name="l4_bn_offset"),
            "l4_bn_scale": tf.Variable(tf.constant(1.0, shape=[1,1,1,self.nout3]),name="l4_bn_scale"),
            "l4_1x1_weights": tf.Variable(tf.truncated_normal([1, 1, self.nout3, self.nout3], stddev=0.01), name="l4_1x1_weights"),
            "l4_weights": tf.Variable(tf.truncated_normal([int((self.imcropsize/16)*(self.imcropsize/16)*self.nout3), self.noutFC1], stddev=0.01), name="l4_weights"),
            "l4_biases": tf.Variable(tf.constant(0.01, shape=[self.noutFC1]), name="l4_biases"),
            
            "l5_bn_offset": tf.Variable(tf.constant(0.0, shape=[1,self.noutFC1]),name="l5_bn_offset"),
            "l5_bn_scale": tf.Variable(tf.constant(1.0, shape=[1,self.noutFC1]),name="l5_bn_scale"),
            "l5_weights": tf.Variable(tf.truncated_normal([self.noutFC1, self.noutFC2], stddev=0.01), name="l5_weights"),
            "l5_biases": tf.Variable(tf.constant(0.01, shape=[self.noutFC2]), name="l5_biases"),

            "alpha": tf.Variable(tf.truncated_normal([self.noutFC2, 1], stddev=0.01), name="alpha")
        }

        # these variables are to store running averages for batch normalization
        for branch in range(3):
            self.variables_dict['mmts%d0'  % (branch)] = tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nchannels]), name='mmts%d0'  % (branch),trainable=False)
            self.variables_dict['mmts%d1'  % (branch)] = tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nchannels]), name='mmts%d1'  % (branch),trainable=False)
            self.variables_dict['mmts%d2'  % (branch)] = tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nout0]),     name='mmts%d2'  % (branch),trainable=False)
            self.variables_dict['mmts%d3'  % (branch)] = tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nout0]),     name='mmts%d3'  % (branch),trainable=False)
            self.variables_dict['mmts%d4'  % (branch)] = tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nout1]),     name='mmts%d4'  % (branch),trainable=False)
            self.variables_dict['mmts%d5'  % (branch)] = tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nout1]),     name='mmts%d5'  % (branch),trainable=False)
            self.variables_dict['mmts%d6'  % (branch)] = tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nout2]),     name='mmts%d6'  % (branch),trainable=False)
            self.variables_dict['mmts%d7'  % (branch)] = tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nout2]),     name='mmts%d7'  % (branch),trainable=False)
            self.variables_dict['mmts%d8'  % (branch)] = tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nout3]),     name='mmts%d8'  % (branch),trainable=False)
            self.variables_dict['mmts%d9'  % (branch)] = tf.Variable(tf.constant(0.0, shape=[1,1,1,self.nout3]),     name='mmts%d9'  % (branch),trainable=False)
            self.variables_dict['mmts%d10' % (branch)] = tf.Variable(tf.constant(0.0, shape=[1,self.noutFC1]),       name='mmts%d10' % (branch),trainable=False)
            self.variables_dict['mmts%d11' % (branch)] = tf.Variable(tf.constant(0.0, shape=[1,self.noutFC1]),       name='mmts%d11' % (branch),trainable=False)

        # --------------------------------------------------
        # model
        # --------------------------------------------------

        # tf_data1 = tf.placeholder("float", shape=[None,imcropsize,imcropsize,nchannels])
        # tf_data2 = tf.placeholder("float", shape=[None,imcropsize,imcropsize,nchannels])
        # tf_data3 = tf.placeholder("float", shape=[None,imcropsize,imcropsize,nchannels])

        tf_data1 = tf.placeholder("float", shape=[None,None,None,nchannels])
        tf_data2 = tf.placeholder("float", shape=[None,None,None,nchannels])
        tf_data3 = tf.placeholder("float", shape=[None,None,None,nchannels])

        self.tf_data = [];
        self.tf_data.append(tf_data1)
        self.tf_data.append(tf_data2)
        self.tf_data.append(tf_data3)

        self.ms = [] # moments (means and variances)
        self.cnn = []
        for branch in range(3):
            m, v = self.moments(self.tf_data[branch],0)
            self.ms.append(m)
            self.ms.append(v)
            l0 = self.l0(self.tf_data[branch],m,v)
            m, v = self.moments(l0,1)
            self.ms.append(m)
            self.ms.append(v)
            l1 = self.l1(l0,m,v)
            m, v = self.moments(l1,2)
            self.ms.append(m)
            self.ms.append(v)
            l2 = self.l2(l1,m,v)
            m, v = self.moments(l2,3)
            self.ms.append(m)
            self.ms.append(v)
            l3 = self.l3(l2,m,v)
            m, v = self.moments(l3,4)
            self.ms.append(m)
            self.ms.append(v)
            l4 = self.l4(l3,m,v)
            m, v = self.moments(l4,5)
            self.ms.append(m)
            self.ms.append(v)
            l5 = self.l5(l4,m,v)

            self.cnn.append(l5)

        self.sgm_same = tf.sigmoid(1-tf.matmul(tf.abs(self.cnn[0]-self.cnn[1]),self.variables_dict["alpha"]))
        self.sgm_diff = tf.sigmoid(1-tf.matmul(tf.abs(self.cnn[0]-self.cnn[2]),self.variables_dict["alpha"]))
        self.triplet_loss = -tf.reduce_sum(tf.log(self.sgm_same)+tf.log(1-self.sgm_diff))

        # test cnn
        # will use the saved running averages of moments, instead of computing them
        self.cnn_test = []
        for branch in range(3):
            m, v = self.moments(self.tf_data[branch],0,False,branch)
            l0 = self.l0(self.tf_data[branch],m,v)
            m, v = self.moments(l0,1,False,branch)
            l1 = self.l1(l0,m,v)
            m, v = self.moments(l1,2,False,branch)
            l2 = self.l2(l1,m,v)
            m, v = self.moments(l2,3,False,branch)
            l3 = self.l3(l2,m,v)
            m, v = self.moments(l3,4,False,branch)
            l4 = self.l4(l3,m,v)
            m, v = self.moments(l4,5,False,branch)
            l5 = self.l5(l4,m,v)
            self.cnn_test.append(l5)
            if branch == 0:
                self.b0l0_test = l0
                self.b0l1_test = l1
                self.b0l2_test = l2
                self.b0l3_test = l3

        self.sgm_same_test = tf.sigmoid(1-tf.matmul(tf.abs(self.cnn_test[0]-self.cnn_test[1]),self.variables_dict["alpha"]))
        self.sgm_diff_test = tf.sigmoid(1-tf.matmul(tf.abs(self.cnn_test[0]-self.cnn_test[2]),self.variables_dict["alpha"]))
        self.tf_denominator = tf.placeholder('float')
        self.tf_acrc_step = (tf.reduce_sum(tf.cast(tf.greater(self.sgm_same_test,0.5),'float'))+tf.reduce_sum(tf.cast(tf.less_equal(self.sgm_diff_test,0.5),'float')))/self.tf_denominator

        # --------------------------------------------------
        # summaries for tensorboard
        # --------------------------------------------------
        
        with tf.name_scope('sgm'):
            msgmsame = tf.reduce_mean(self.sgm_same)
            msgmdiff = tf.reduce_mean(self.sgm_diff)
            tf.summary.scalar('sgm_same', msgmsame)
            tf.summary.scalar('sgm_diff', msgmdiff)
        with tf.name_scope('loss'):
            tf.summary.scalar('triplet_loss', self.triplet_loss)
            tf.summary.scalar('accuracy', self.tf_acrc_step)
        self.merged = tf.summary.merge_all()

        # --------------------------------------------------
        # batch normalization management
        # --------------------------------------------------

        # list of moment variables (we'll store the running averages here after training)
        mmts_list = []
        for branch in range(3):
            for i in range(12):
                mmts_list.append(self.variables_dict['mmts%d%d' % (branch,i)])

        # moving average object; we'll maintain averages of moments independently for each branch of the triplet network
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        maintain_averages = ema.apply(self.ms)

        # ops to assign running averages to appropriate variables to be saved
        ops = []
        for i in range(len(mmts_list)):
            ops.append(mmts_list[i].assign(ema.average(self.ms[i])))
        self.asgn_run_avgs = tf.tuple(ops)

        # --------------------------------------------------
        # optimization ops
        # --------------------------------------------------

        opt_op = tf.train.MomentumOptimizer(1e-5,0.9).minimize(self.triplet_loss)

        # this creates an operation to maintain the averages that depends on opt_op, so that
        # when we call opt_op_with_ema it calls opt_op before
        with tf.control_dependencies([opt_op]):
            self.opt_op_with_ema = tf.group(maintain_averages)

    # --------------------------------------------------
    # cnn layers
    # --------------------------------------------------

    def pixel_intensity_normalization(self, imagebatch):
        m,v = tf.nn.moments(imagebatch,axes=[1,2],keep_dims=True)
        # return tf.nn.batch_normalization(imagebatch,m,v,0.0,1.0,0.000001)
        bn = tf.nn.batch_normalization(imagebatch,m,v,0.0,1.0,0.000001)
        rb = tf.image.random_brightness(bn,0.1)
        rc = tf.image.random_contrast(rb,0.99,1.01)
        return rc

    def l0(self, data, m, v):
        if self.pixelIntensityNormalization:
            bn_data = tf.nn.batch_normalization(self.pixel_intensity_normalization(data), m, v,
                                                self.variables_dict["l0_bn_offset"], self.variables_dict["l0_bn_scale"], 0.000001)
        else:
            bn_data = tf.nn.batch_normalization(data, m, v, self.variables_dict["l0_bn_offset"], self.variables_dict["l0_bn_scale"], 0.000001)
        c_00 = tf.nn.relu(tf.nn.conv2d(bn_data, self.variables_dict["l0_weights1"], strides=[1, 1, 1, 1], padding='SAME'))
        c_0 = tf.nn.relu(tf.nn.conv2d(c_00, self.variables_dict["l0_weights2"], strides=[1, 1, 1, 1], padding='SAME') + self.variables_dict["l0_biases"])
        if self.residualLearning:
            c_0 = c_0+tf.nn.conv2d(bn_data, self.variables_dict["l0_shortcut"], strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.max_pool(c_0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def l1(self, hidden, m, v):
        bn_h_0 = tf.nn.batch_normalization(hidden, m, v,
            self.variables_dict["l1_bn_offset"], self.variables_dict["l1_bn_scale"], 0.000001)
        if self.localResponseNormalization:
            n_1 = tf.nn.local_response_normalization(bn_h_0)
        else:
            n_1 = bn_h_0
        if self.networkInNetwork:
            c_1_1x1 = tf.nn.relu(tf.nn.conv2d(n_1, self.variables_dict["l1_1x1_weights"], strides=[1, 1, 1, 1], padding='SAME'))
        else:
            c_1_1x1 = n_1
        c_10 = tf.nn.relu(tf.nn.conv2d(c_1_1x1, self.variables_dict["l1_weights1"], strides=[1, 1, 1, 1], padding='SAME'))
        c_1 = tf.nn.relu(tf.nn.conv2d(c_10, self.variables_dict["l1_weights2"], strides=[1, 1, 1, 1], padding='SAME') + self.variables_dict["l1_biases"])
        if self.residualLearning:
            c_1 = c_1+tf.nn.conv2d(c_1_1x1, self.variables_dict["l1_shortcut"], strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.max_pool(c_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def l2(self, hidden, m, v):
        bn_h_1 = tf.nn.batch_normalization(hidden, m, v,
            self.variables_dict["l2_bn_offset"], self.variables_dict["l2_bn_scale"], 0.000001)
        if self.localResponseNormalization:
            n_2 = tf.nn.local_response_normalization(bn_h_1)
        else:
            n_2 = bn_h_1
        if self.networkInNetwork:
            c_2_1x1 = tf.nn.relu(tf.nn.conv2d(n_2, self.variables_dict["l2_1x1_weights"], strides=[1, 1, 1, 1], padding='SAME'))
        else:
            c_2_1x1 = n_2
        c_20 = tf.nn.relu(tf.nn.conv2d(c_2_1x1, self.variables_dict["l2_weights1"], strides=[1, 1, 1, 1], padding='SAME'))
        c_2 = tf.nn.relu(tf.nn.conv2d(c_20, self.variables_dict["l2_weights2"], strides=[1, 1, 1, 1], padding='SAME') + self.variables_dict["l2_biases"])
        if self.residualLearning:
            c_2 = c_2+tf.nn.conv2d(c_2_1x1, self.variables_dict["l2_shortcut"], strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.max_pool(c_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def l3(self, hidden, m, v):
        bn_h_2 = tf.nn.batch_normalization(hidden, m, v,
            self.variables_dict["l3_bn_offset"], self.variables_dict["l3_bn_scale"], 0.000001)
        if self.localResponseNormalization:
            n_3 = tf.nn.local_response_normalization(bn_h_2)
        else:
            n_3 = bn_h_2
        if self.networkInNetwork:
            c_3_1x1 = tf.nn.relu(tf.nn.conv2d(n_3, self.variables_dict["l3_1x1_weights"], strides=[1, 1, 1, 1], padding='SAME'))
        else:
            c_3_1x1 = n_3
        c_30 = tf.nn.relu(tf.nn.conv2d(c_3_1x1, self.variables_dict["l3_weights1"], strides=[1, 1, 1, 1], padding='SAME'))
        c_3 = tf.nn.relu(tf.nn.conv2d(c_30, self.variables_dict["l3_weights2"], strides=[1, 1, 1, 1], padding='SAME') + self.variables_dict["l3_biases"])
        if self.residualLearning:
            c_3 = c_3+tf.nn.conv2d(c_3_1x1, self.variables_dict["l3_shortcut"], strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.max_pool(c_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def l4(self, hidden, m, v):
        bn_h_3 = tf.nn.batch_normalization(hidden, m, v,
            self.variables_dict["l4_bn_offset"], self.variables_dict["l4_bn_scale"], 0.000001)
        if self.localResponseNormalization:
            n_4 = tf.nn.local_response_normalization(bn_h_3)
        else:
            n_4 = bn_h_3
        if self.networkInNetwork:
            c_4_1x1 = tf.nn.relu(tf.nn.conv2d(n_4, self.variables_dict["l4_1x1_weights"], strides=[1, 1, 1, 1], padding='SAME'))
        else:
            c_4_1x1 = n_4
        r_4 = tf.reshape(c_4_1x1, [-1, int((self.imcropsize/16)*(self.imcropsize/16)*self.nout3)])
        return tf.matmul(r_4, self.variables_dict["l4_weights"]) + self.variables_dict["l4_biases"]

    def l5(self, hidden, m, v):
        bn_h_4 = tf.nn.batch_normalization(hidden, m, v,
            self.variables_dict["l5_bn_offset"], self.variables_dict["l5_bn_scale"], 0.000001)
        return tf.matmul(bn_h_4, self.variables_dict["l5_weights"]) + self.variables_dict["l5_biases"]

    # --------------------------------------------------
    # means, variances for batch normalization
    # --------------------------------------------------

    def moments(self,data,layerindex,training=True,branch=None): # if training, branch variable not used
        if training:
            if layerindex < 5:
                return tf.nn.moments(data, axes=[0,1,2], keep_dims=True)
            else:
                return tf.nn.moments(data, axes=[0], keep_dims=True)
        else:
            return self.variables_dict['mmts%d%d' % (branch,layerindex*2)], self.variables_dict['mmts%d%d' % (branch,layerindex*2+1)]

    # --------------------------------------------------
    # train/test
    # --------------------------------------------------

    def assign_running_averages(self,sess):
        sess.run(self.asgn_run_avgs)

    def valid_step_with_summary(self,sess,batch_data1_Valid,batch_data2_Valid,batch_data3_Valid,npairs):
        return sess.run([self.merged, self.triplet_loss, self.tf_acrc_step], feed_dict={self.tf_data[0]: batch_data1_Valid,
                                                                                        self.tf_data[1]: batch_data2_Valid,
                                                                                        self.tf_data[2]: batch_data3_Valid,
                                                                                        self.tf_denominator: npairs})

    def train_step_with_summary(self,sess,batch_data1,batch_data2,batch_data3,npairs):
        return sess.run([self.merged, self.tf_acrc_step, self.opt_op_with_ema],feed_dict={self.tf_data[0]: batch_data1,
                                                                                          self.tf_data[1]: batch_data2,
                                                                                          self.tf_data[2]: batch_data3,
                                                                                          self.tf_denominator: npairs})

    def train_step_without_summary(self,sess,batch_data1,batch_data2,batch_data3):
        sess.run(self.opt_op_with_ema,feed_dict={self.tf_data[0]: batch_data1,
                                                 self.tf_data[1]: batch_data2,
                                                 self.tf_data[2]: batch_data3})

    def test(self,sess,batch_data1,batch_data2):
        return sess.run(self.sgm_same_test,feed_dict={self.tf_data[0]: batch_data1, self.tf_data[1]: batch_data2})

    def test2(self,sess,batch_data1,batch_data2):
        return sess.run(self.sgm_diff_test,feed_dict={self.tf_data[0]: batch_data1, self.tf_data[2]: batch_data2})

    def test_triplet(self,sess,batch_data1,batch_data2,batch_data3):
        return sess.run([self.sgm_same, self.sgm_diff],feed_dict={self.tf_data[0]: batch_data1, self.tf_data[1]: batch_data2, self.tf_data[2]: batch_data3})