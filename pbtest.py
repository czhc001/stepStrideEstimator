from tensorflow.python.platform import gfile
import tensorflow as tf
import load

sess = tf.Session()
with gfile.FastGFile('Models/model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图

# 需要有一个初始化的过程
sess.run(tf.global_variables_initializer())


# 输入
X = sess.graph.get_tensor_by_name('X:0')
P = sess.graph.get_tensor_by_name('P:0')

op = sess.graph.get_tensor_by_name('output/BiasAdd:0')

data, label = load.load_data_and_label('C:/Users/niu/IdeaProjects/IdentitySample/data/sample1557283074375.csv')
r = sess.run(op, feed_dict={
        X: data[:, :],
        P: 1.0})
for l in r:
    print(l)
