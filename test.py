import tensorflow as tf
w = tf.Variable(1.0)
ema = tf.train.ExponentialMovingAverage(0.9)
update = tf.assign_add(w, 1.0)

with tf.control_dependencies([update]):
    #返回一个op,这个op用来更新moving_average,i.e. shadow value
    ema_op = ema.apply([w])#这句和下面那句不能调换顺序
# 以 w 当作 key， 获取 shadow value 的值
ema_val = ema.average(w)#参数不能是list，有点蛋疼

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(5):
        sess.run(ema_op)
        print(sess.run(ema_val))
# 创建一个时间序列 1 2 3 4
#输出：
#1.1      =0.9*1 + 0.1*2
#1.29     =0.9*1.1+0.1*3
#1.561    =0.9*1.29+0.1*4
