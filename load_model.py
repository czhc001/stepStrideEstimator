import tensorflow as tf
import numpy as np
import load


data, label = load.load_data_and_label('C:/Users/niu/IdeaProjects/IdentitySample/data/sample1557389138187.csv')
saver = tf.train.import_meta_graph("Models/model.ckpt.meta")

output_node_names = "output/BiasAdd,X,P"

with tf.Session() as sess:
    saver.restore(sess, "Models/model.ckpt")

    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
    result = graph.get_tensor_by_name("output/BiasAdd:0")
    X = graph.get_tensor_by_name("X:0")
    P = graph.get_tensor_by_name("P:0")
    r = sess.run(result, feed_dict={
        X: data[:, :],
        P: 1.0})
    for l in r:
        print(l)
    output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
        sess=sess,
        input_graph_def=input_graph_def,  # 等于:sess.graph_def
        output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

    with tf.gfile.GFile("Models/model.pb", "wb") as f:  # 保存模型
        f.write(output_graph_def.SerializeToString())  # 序列化输出
    print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

