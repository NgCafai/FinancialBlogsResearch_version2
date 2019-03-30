# -*- coding: utf-8 -*-

import tensorflow as tf


class Model(object):
    """
    用于分类的模型
    """
    def __init__(self, config, wordEmbedding):
        self.config = config
        self.wordEmbedding = wordEmbedding

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """
        CNN模型
        :return:
        """
        # Embedding layer
        with tf.device('/gpu:0'):
            with tf.name_scope('embedding_layer'):
                # 利用预训练的词向量初始化词嵌入矩阵
                W = tf.Variable(tf.cast(self.wordEmbedding, dtype=tf.float32, name="word2vec"), name='W')
                # 利用词嵌入矩阵，将input_x中的id转化成词向量
                embedding_input = tf.nn.embedding_lookup(W, self.input_x)

            with tf.name_scope('convolution_and_max_pooling'):
                # 这里分了几个不同size的filter
                pooled_outputs = []
                len_kernel_size = len(self.config.kernel_size)  # 一共有几种filter
                for i, filter_size in enumerate(self.config.kernel_size):
                    with tf.name_scope('filter_%d' % filter_size):
                        num_filters_each = int(self.config.num_filters / len_kernel_size)
                        # Convolution layer；conv.shape = [None, seq_length - kernel_size + 1, num_filters_each]
                        conv = tf.layers.conv1d(embedding_input, num_filters_each, filter_size,
                                                activation='relu', use_bias=True)
                        # global max pooling layer；gmp.shape = [None, num_filters_each]
                        gmp = tf.reduce_max(conv, reduction_indices=[1])
                        pooled_outputs.append(gmp)
                pooled_total = tf.concat(pooled_outputs, 1)  # [batch_size, num_filters]
                pooled_total_flat = tf.reshape(pooled_total, [-1, self.config.num_filters])

            with tf.name_scope('fully_connected'):
                fc = tf.layers.dense(pooled_total_flat, self.config.hidden_dim, name='fc1')
                fc = tf.contrib.layers.dropout(fc, self.keep_prob)
                fc = tf.nn.relu(fc)

            with tf.name_scope('predict'):
                self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
                # 模型预测的类别
                self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

            with tf.name_scope('optimize'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
                self.loss = tf.reduce_mean(cross_entropy)

                # 添加L2正则化：
                l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                      if 'bias' not in v.name and ('fc1' in v.name or 'fc2' in v.name)]) \
                            * self.config.l2_lambda
                self.loss = self.loss + l2_losses

                # 优化器
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

            with tf.name_scope('accuracy'):
                correct_pred = tf.equal(self.y_pred_cls, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
