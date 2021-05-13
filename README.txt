Hello, here are instructions for running inference on our project: 

File Content: 
README.txt - This file
requirements.txt - The python requirments necessary to run this project
train_sample.py - A sample training file which trains a simple model on the data, and save the checkpoint to be loaded
                  in the test_submission.py file.
conform_validation_set.py - python script needed to reformat the validation set to match what the training script is expecting. 
test_submission.py - The file which will return an output for every input in the eval.csv
eval.csv - An example test file
data/get_data.sh - A script which will download the tiny-imagenet data into the data/tiny-imagenet-200 file

Note: You should be using Python 3 to run this code.


To run inferences first install dependencies through `pip3 install -r requirements.txt`, and then run script: `python3 test_submission.py eval.csv`.

TODO
add clip dependencies
