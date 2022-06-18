# Tracking Mobility

An application designed for tracking mobility (TM). The application provides image detection model trainer, inference server and RC car controller.

## Image Detection Model Training

### Environment Setup

Image training is designed to be executed at the environment relevant to deep learning such as a computer with GPU.

Install dependencies by
```sh
$ pip install -r requirements-train.txt
```

If cuda is supported on the computer, install also:
```sh
$ pip install -r requirements-train-cuda.txt
```

### Image Preprocessing

* Run src/preprocessor.py to preprocess the dataset
* Resizes each image in `src_dir` directory to `size * size`, then converts to numpy array, which is saved under `dst_dir` directory
* If `rewrite` set to `True`, every images are preprocessed

### Training Model

* Run src/train.py to train model
* Pytorch library and ResNet architecture is used to train image classification model
* The model determines whether the image is *part* of the target or not
* Some training options (e.g. batch size, learning rate) can be provided from command line
* Trained model is exported under `export_dir` directory if `export` is set to `True`

### Training Results

* **Dataset Creation**: Orange vest was chosen as a target for the TM. Dataset with about 14K images were collected by crawling from Google, slicing, and classifying manually. Images containing part of the target were labeled as "true", and others were labeled as "false".
* **Image Preprocessing**: Images were preprocessed with the size of 32 * 32.
* **Model Training**: Image classification model was trained for 5 epochs, with other options default. 6/7 of the dataset were used for training and 1/7 were used for test

```sh
$ python src/preprocessor.py --size 32
$ python src/train.py --epoch 5
```

## Inference Server

### Environment Setup

Inference server is designed to be executed at the environment relevant to deploy servers, such as AWS server.

Install dependencies by
```sh
$ pip install -r requirements-server.txt
```

### Description

* Run src/inference_server.py to deploy server
* The server only accepts POST / route, with a numpy-converted image passed in the format of JSON
* The image is divided into `split_count * split_count` images evenly with the size of `train_size * train_size`
* Then, the centroid of the target is computed based on the inference result of each image
* Sends reponse to the server with existence and normalized centroid of the target

### Centroid Computing Example

* If `train_size = 32` and `split_count = 4`, image from the client should be `128 * 128`
* Probability of each image being part of the target is inferred from the model.
* Centroid of the target is computed by the weighted average of probability and center point of each image
* Target is determined to be absent when the probability of all images is less than 0.5

## Tracking Mobility Controller

### Environment Setup

TM controller is designed to be executed at the Raspberry Pi computer with loaded on the TM. Raspberry Pi should have access to camera, motor controller, and wireless network.

Install dependencies by
```sh
$ pip install -r requirements-pi.txt
```

### Description

* Run src/car_controller.py on the Raspberry Pi to control the TM
* Captures image of the front of the TM with a camera
* Obtains centroid of the target by sending request to the inference server with the image
* Stops the car if the centroid does not exist
* Changes direction of the TM by applying relevant duty cycle to each wheel according to the normalized position of the centroid
* Repeats the whole process infinitely

## Results
