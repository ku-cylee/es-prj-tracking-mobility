# Tracking Mobility

![tm-overview](./images/tm-overview.png)

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
* PyTorch library and ResNet architecture is used to train image classification model
* The model determines whether the image is *part* of the target or not
* Some training options (e.g. batch size, learning rate) can be provided from command line
* Trained model is exported under `export_dir` directory if `export` is set to `True`

### Training Results

* **Dataset Creation**: Orange vest was chosen as a target for the TM. Dataset with about 14.6K images were collected by crawling from Google, slicing, and classifying manually. Images containing part of the target were labeled as "true", and others were labeled as "false".
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

### Image Detection Model

The accuracy over test dataset after training is:
* Overall dataset: 97.94%
* "false" dataset: 97.74%
* "true" dataset: 98.11%

### Inference and Centroid Computation

* Image captured from the camera is sliced into 4 * 4 images.
* Since the image model was trained for 32 * 32 size, the captured image is resized to 128 * 128.

![centroid-example-light](./images/centroid-example-light.png#gh-light-mode-only)
![centroid-example-dark](./images/centroid-example-dark.png#gh-dark-mode-only)

* Figure (a): A sample image with the target
* Figure (b): Inference result for each image with its true probability and center point position
* Figure (c): Computed centroid position
* Let $c_i$ and $p_i$ be the center point and true probability of $i$-th image. Then, the centroid of the target is obtained by

$$ C=\frac{\sum_{i=1}^{16}c_ip_i}{\sum_{i=1}^{16}p_i}=\frac{(2.65\times10^{-3})\cdot(16,16)+(2.16\times10^{-3})\cdot(48,16)+\cdots+(3.27\times10^{-2})\cdot(112,112)}{2.65\times10^{-3}+2.16\times10^{-3}+\cdots+3.27\times10^{-2}}=(83.89,51.62) $$

* Figure (c) shows that the obtained centroid quite accurate
* The inference server responds the following data to the client in the JSON format
    - `exists`: Existence of the target (in this case, true)
    - `offset`: Noramlized horizontal centroid position in the format of JSON. Normalized to be between -1 and 1: -1 the leftmost, 0 the center, and 1 the rightmost. (in this case, -0.19)

### Tracking Mobility Controller

[![tm-demo](https://img.youtube.com/vi/mu4dMLrJ6V8/0.jpg)](https://www.youtube.com/watch?v=mu4dMLrJ6V8)

Demo video can be seen above. The controller manipulates the RC car well to follow the target. It also stops when it fails to detect the target.

## Limitations

* Hardware Problems
    - Some hardwares had problems, causing major problems.
    - Performance of provided Raspberry Pi was poor. Low voltage problem and kernel panic were commonly issued problems and the computing speed was too low. This problem delayed the development schedule too much.
    - Wheels performed differently despite the same change duty cycle value. Specifically, the rotation speed of the right wheel was much slower than that of the left wheel. This caused the TM to change its direction inaccurately, which can be seen in the demo video above.
* PyTorch cannot be installed on Raspberry Pi
    - Pre-built packages of PyTorch later than version 1.0.0 for Raspberry Pi was not provided officially.
    - This caused the inference to be performed on the remote server.
* Ultrasonic wasn't able to be exploited due to delayed development schedule
