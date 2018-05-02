import tensorflow as tf
import numpy as np

learning_rate=0.005

x = np.vstack([np.random.normal(0.1,1,(500,100)), np.random.normal(-0.1,1,(500,100))])
y = np.hstack([1.*np.ones(500),-1.*np.ones(500)])

X = tf.placeholder(tf.float32,[1000,100])
Y = tf.placeholder(tf.float32,[1000])
W = tf.Variable(tf.random_uniform([100,1], -2.0,2.0))
Z = tf.maximum(tf.zeros([1000],dtype=tf.float32),tf.ones([1000],dtype=tf.float32)-tf.matmul(X,W)*tf.reshape(Y,[1000,1]))
loss=tf.reduce_mean(Z)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        sess.run(train,feed_dict={X: x, Y: y})
        if i%200==0:
            print(sess.run(loss,feed_dict={X: x, Y: y}))
