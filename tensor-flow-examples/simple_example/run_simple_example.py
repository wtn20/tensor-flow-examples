from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

print('Loading data...')
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


with tf.Session() as sess:
    print('Initializing graph...')
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    print('Initializing weights and biases...')
    sess.run(tf.global_variables_initializer())

    print('Defining model and loss function... ')
    y = tf.matmul(x,W) + b
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels= y_, logits=y)
    )

    print('Starting training...')

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    max_steps = 1000

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for i in range(max_steps):
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
        if i % 100 == 0:
            print('Training step {} (of {}): Training accuracy = {}'.format(
                i,
                max_steps,
                sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1]})
            ))

    print('Final test accuracy = {}'.format(accuracy.eval(
        feed_dict={x: mnist.test.images, y_: mnist.test.labels}
    )))

