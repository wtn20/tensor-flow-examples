from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def load_data(data_dir, fake_data):
    print('Loading data...')
    return input_data.read_data_sets(data_dir,
                                     one_hot=True,
                                     fake_data=fake_data
                                     )


def initialize_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def initialize_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stdev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stdev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#          TODO
# def conv_layer(input_tensor,
#                input_dim,
#                output_dim,
#                layer_name):
#     w_conv = initialize_weight_variable()

def define_place_holders():
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
    with tf.name_scope('dropout'):
        # Set up dropout probability
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
    return x, y_, keep_prob


def build_and_run_network(sess, FLAGS, x, y_, mnist, keep_prob):

    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 10)

    # Convolutional layer 1
    with tf.name_scope('convolution_layer_1'):
        with tf.name_scope('weights'):
            W_conv1 = initialize_weight_variable([5, 5, 1, 32])
            variable_summaries(W_conv1)

        with tf.name_scope('biases'):
            b_conv1 = initialize_bias_variable([32])
            variable_summaries(b_conv1)

        # Convolve and max pool
        with tf.name_scope('convolution'):
            preactivate = conv2d(x_image, W_conv1) + b_conv1
            tf.summary.histogram('pre-activation', preactivate)

            h_conv1 = tf.nn.relu(preactivate, name='activation')
            # variable_summaries(h_conv1)

        with tf.name_scope('pooling'):
            h_pool1 = max_pool_2x2(h_conv1)
            # variable_summaries(h_pool1)

        # Prep for output layer
        with tf.name_scope('sending_to_output_layer'):
            with tf.name_scope('weights'):
                W_mem_conv1 = initialize_weight_variable([14 * 14 * 32, 10])
                variable_summaries(W_mem_conv1)

            with tf.name_scope('biases'):
                b_mem_conv1 = initialize_bias_variable([10])
                variable_summaries(b_mem_conv1)

            with tf.name_scope('tensor_reshape'):
                h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*32])
                # variable_summaries(h_pool1_flat)

            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(h_pool1_flat, W_mem_conv1) + b_mem_conv1
                tf.summary.histogram('pre-activation', preactivate)

                y_conv_1_mem_full = tf.nn.relu(preactivate, name='activation')

            with tf.name_scope('dropout'):
                y_conv_1_mem_drop = tf.nn.dropout(y_conv_1_mem_full, keep_prob)

    # Convolutional layer 2
    with tf.name_scope('convolution_layer_2'):
        with tf.name_scope('weights'):
            W_conv2 = initialize_weight_variable([5, 5, 32, 64])
            variable_summaries(W_conv2)

        with tf.name_scope('biases'):
            b_conv2 = initialize_bias_variable([64])
            variable_summaries(b_conv1)

        # Convolve and max pool
        with tf.name_scope('convolution'):
            preactivate = conv2d(h_pool1, W_conv2) + b_conv2
            tf.summary.histogram('pre-activation', preactivate)

            h_conv2 = tf.nn.relu(preactivate)
            # variable_summaries(h_conv2)

        with tf.name_scope('pooling'):
            h_pool2 = max_pool_2x2(h_conv2)
            # variable_summaries(h_pool2)

        # Prep for output layer
        with tf.name_scope('sending_to_output_layer'):
            with tf.name_scope('weights'):
                W_mem_conv2 = initialize_weight_variable([7 * 7 * 64, 10])
                variable_summaries(W_mem_conv2)

            with tf.name_scope('biases'):
                b_mem_conv2 = initialize_bias_variable([10])
                variable_summaries(b_mem_conv2)

            with tf.name_scope('tensor_reshape'):
                h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
                # variable_summaries(h_pool2_flat)

            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(h_pool2_flat, W_mem_conv2) + b_mem_conv2
                tf.summary.histogram('pre-activation', preactivate)

                y_conv_2_mem_full = tf.nn.relu(preactivate, name='activation')

            with tf.name_scope('dropout'):
                y_conv_2_mem_drop = tf.nn.dropout(y_conv_2_mem_full, keep_prob)

        # Densely connected layer
    with tf.name_scope('dense_layer_1'):
        with tf.name_scope('weights'):
            W_fc1 = initialize_weight_variable([7 * 7 * 64, 1024])
            variable_summaries(W_fc1)

        with tf.name_scope('biases'):
            b_fc1 = initialize_bias_variable([1024])
            variable_summaries(b_fc1)

        with tf.name_scope('tensor_reshape'):
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            # variable_summaries(h_pool2_flat)

        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
            tf.summary.histogram('pre-activation', preactivate)

        h_fc1 = tf.nn.relu(preactivate, name='activation')

        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Prep for readout layer
        with tf.name_scope('sending_to_output_layer'):
            with tf.name_scope('weights'):
                W_fc2 = initialize_weight_variable([1024, 10])
                variable_summaries(W_fc2)

            with tf.name_scope('biases'):
                b_fc2 = initialize_bias_variable([10])
                variable_summaries(b_fc2)

            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
                tf.summary.histogram('pre-activation', preactivate)

            y_conv_org = preactivate

    # Create final output layer
    with tf.name_scope('output_layer'):
        with tf.name_scope('weights'):
            W_out = initialize_weight_variable([1, 3])  # number of layers feeding into output layer
            variable_summaries(W_out)

        with tf.name_scope('biases'):
            b_out = initialize_bias_variable([1])
            variable_summaries(b_out)

        # Merge outputs from each layer into a single tensor
        with tf.name_scope('stack_outputs'):
            h_inputs_from_layers = tf.stack([y_conv_org, y_conv_1_mem_drop, y_conv_2_mem_drop], 0)

        # Dot product along single tensor dimension
        with tf.name_scope('Wx_plus_b'):
            temp = tf.matmul(W_out, tf.reshape(h_inputs_from_layers, [3,-1])) + b_out
            y_conv = tf.reshape(temp, [-1,10])


    # Define the loss and accuracy functions
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate)\
            .minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1),
                                          tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    # Start the log writers
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

    # Initialize the weights and biases
    tf.global_variables_initializer().run()

    def feed_dict(train):
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    # Train the model
    for i in range(FLAGS.max_steps):

        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy],
                                    feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Testing run {} (of {}): current test accuracy = {}'.format(i, FLAGS.max_steps, acc))
        else:
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)

                _, acc = sess.run([merged, accuracy],
                                        feed_dict=feed_dict(True))

                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Training step {} (of {}): Training accuracy = {}'.format(i, FLAGS.max_steps, acc))
            else:
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()
