# A selection of TensorFlow examples

## Installation
1. Clone the repo
2. Install virtualenv
  * `$ sudo easy_install pip`
  * `$ sudo pip install --upgrade virtualenv`
3. Create a virtual environment in the folder you want to work in
  * `virtualenv --system-site-packages -p python3 targetDirectory`
4. Activate the virtual environment (from within the target directory)
  * `$ source ~bin/activate`
5. You should now have a clean python3 install. Use pip to install TensorFlow
  * `pip3 install --upgrade tensorflow`
6. When finished, deactivate the virtual environment using:
  * `deactivate`


## Running the examples
All of the examples here use the MNIST data set.

The pure TensorFlow examples are in the `tensor-flow-examples` folder
and follow the examples given in this [tensor flow tutorial](https://www.tensorflow.org/get_started/mnist/pros)

The example in the `tensor_board_example` folder follows this [tensor board tutorial](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

Finally, the example given in `maximally_connected_nn` is an example that mimics
the `tensor-flow-example/advanced_example` with the addition of memory
augmented connections between the hidden layers and the output layer.

To run any of these:

`python <path_to_file>/run_<example_name>.py --args`

### Options
The TensorBoard examples have various options, which can be found using:
`python run_tensor_board_example.py --help`

## TensorBoard
The tensor-board-example and the maximally_connected_nn both have events logged
for use by TensorBoard. To view the output after the model has run, use:

`tensorboard --logdir=<path_to_logs>`

where `<path_to_logs>` is the relative path to the logs created by the model
(specified by the --log_dir option when running the model)

If this runs sucessfully, you should be prompted to view the TensorBoard output on a browser
(e.g. `localhost:6006`)

