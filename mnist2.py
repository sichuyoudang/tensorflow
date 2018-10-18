# -*- coding: utf-8 -*-
# 识别模糊手写数字-构建模型

import tensorflow as tf 
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import pylab
tf.reset_default_graph()
# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))
pred = tf.nn.softmax(tf.matmul(x,W)+b)
# 损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# 定义参数
learning_rate= 0.01
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
training_epochs = 25
batch_size = 100
display_Step = 1

saver = tf.train.Saver()
model_path = "log/521model.ckpt"

# # 启动 session
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())

# 	# 启动循环训练
# 	for epoch in range(training_epochs):
# 		avg_cost = 0
# 		total_batch = int(mnist.train.num_examples/batch_size)
# 		for i in range(total_batch):
# 			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
# 			_, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
# 			avg_cost += c/ total_batch
# 		if (epoch+1)%display_Step ==0:
# 			print("epoch:", '%04d' %(epoch+1), "cost=","{:.9f}".format(avg_cost))
# 	print("finish")
# 	# 测试 model
# 	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
# 	# 计算准确率
# 	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 	print("Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
# 	# 保存模型
# 	save_path = saver.save(sess, model_path)
# 	print("MODEL SAVED IN FILE: %s" %save_path)

print("Starting 2nd session")
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, model_path)
	# 测试 Model
	correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	# 计算准确率
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("Accuracy:", accuracy.eval({x:mnist.test.images, y: mnist.test.labels}))

	output = tf.argmax(pred, 1)
	batch_xs, batch_ys = mnist.train.next_batch(2)
	outputval, predv = sess.run([output, pred], feed_dict={x:batch_xs})
	print(outputval, pred, batch_ys)

	im = batch_xs[0]
	im = im.reshape(-1,28)
	pylab.imshow(im)
	pylab.show()

	im = batch_xs[1]
	im = im.reshape(-1,28)
	pylab.imshow(im)
	pylab.show()