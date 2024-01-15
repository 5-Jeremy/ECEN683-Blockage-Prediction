# Blockage Prediction for Wireless Communication Using mmWave Power Signatures
This project is an attempt to solve the task of Wireless Signature-Based Blockage Prediction proposed by the creators of the [DeepSense 6G dataset](https://www.deepsense6g.net/). The task considers a wireless transmitter sending a constant stream of symbols to a receiver, which measures the received power for a codebook of 64 beams. Over time, obstacles briefly enter the space between the transmitter and receiver, resulting in a blockage which prevents the symbols from being received. A deep learning model is provided a sequence of mmWave samples collected in this way, and must output a prediction for whether a blockage will occur over a certain number of future time steps.

The use of a convolutional neural network (CNN) or a recurrent neural network (RNN) individually to solve this task was explored in [a paper by the creators of the DeepSense 6G dataset](https://www.wi-lab.net/research/wireless-signature-blockage-prediction-paper/). The code presented here combines a convolutional layer with an RNN and applies it to the same task.

For a complete presentation of the project, see [this video](https://www.youtube.com/watch?v=uh5NG3WydI0).

To replicate my results or experiment with other settings, you only need to use the file train_and_test.ipynb. The file prep_data.py contains functions used to load and preprocess the data.

