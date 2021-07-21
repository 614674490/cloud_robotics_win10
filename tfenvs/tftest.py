import tensorflow as if
message = tf.constant('Welcome to the exciting world of Deep Neural Networks!')
#执行计算图
with tf.Session() as sess:
    print(sess.run(message).decode())
