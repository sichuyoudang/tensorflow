import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*1.0+3.0

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 创建变量Weight是，范围是 -1.0~1.0
biases = tf.Variable(tf.zeros([1]))                      # 创建偏置，初始值为0
y = Weights*x_data+biases                                # 定义方程
loss = tf.reduce_mean(tf.square(y-y_data))               # 定义损失，为真实值减去我们每一步计算的值
optimizer = tf.train.GradientDescentOptimizer(0.5)       # 0.5 是学习率
train = optimizer.minimize(loss)                         # 使用梯度下降优化
init = tf.initialize_all_variables()                     # 初始化所有变量

sess = tf.Session()
sess.run(init)

#for i in range(201):
#   sess.run(train)
#   if i%20 == 0:
#       print(i,sess.run(Weights),sess.run(biases))

input1 = tf.placeholder(tf.float32) #在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1,input2)  # 乘法运算

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:7.,input2:2.}))
