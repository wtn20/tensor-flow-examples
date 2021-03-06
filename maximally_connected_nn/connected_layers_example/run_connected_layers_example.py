import tensorflow as tf
import build_and_train

import argparse
import sys

FLAGS = None

with tf.Session() as sess:
    def main(_):
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)

        # Load the MNIST data
        mnist = build_and_train.load_data(FLAGS.data_dir, FLAGS.fake_data)

        # define the placeholder variables for the inputs & ouputs
        x, y_, keep_prob = build_and_train.define_place_holders()

        # Train the model
        build_and_train.build_and_run_network(sess, FLAGS, x, y_, mnist, keep_prob)


    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--fake_data',
                            nargs='?',
                            const=True,
                            type=bool,
                            default=False,
                            help='If true, uses fake data for unit testing.')

        parser.add_argument('--max_steps',
                            type=int,
                            default=1000,
                            help='Number of step to run trainer.')

        parser.add_argument('--learning_rate',
                            type=float,
                            default=0.01,
                            help='Initial learning rate')

        parser.add_argument('--dropout',
                            type=float,
                            default=0.9,
                            help='Keep probability for training dropout')

        parser.add_argument('--data_dir',
                            type=str,
                            default='/tmp/tensorflow/mnist/input_data',
                            help='Directory for storing input data')

        parser.add_argument('--log_dir',
                            type=str,
                            default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                            help='Summaries log directory')
        FLAGS, unparsed = parser.parse_known_args()

        print('Running with: ' + str(FLAGS))

        tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)