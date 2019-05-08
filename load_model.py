import tensorflow as tf
import numpy as np
import load


data, label = load.load_data_and_label('C:/Users/niu/IdeaProjects/IdentitySample/data/sample1557283074375.csv')
saver = tf.train.import_meta_graph("Models/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "Models/model.ckpt")
    result = tf.get_default_graph().get_tensor_by_name("output/BiasAdd:0")
    X = tf.get_default_graph().get_tensor_by_name("X:0")
    P = tf.get_default_graph().get_tensor_by_name("P:0")
    r = sess.run(result, feed_dict={
        X: data[:, :],
        P: 1.0})
    for l in r:
        print(l)
