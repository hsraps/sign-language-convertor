# SignLanguage-Digits-to-Text

A real time American Sign Language(ASL) to Text convertor.

## Getting Started
--------------------------------------------------------------------------------------------------------------

### Install

This project requires **Python 3** and following Python libraries installed:
* [Numpy](http://www.numpy.org/)
* [OpenCV](https://opencv.org/)
* [Keras](https://keras.io/)
* [h5py](https://www.h5py.org/) (for loading and saving pre-trained model)

### Code
The 'Imagetest.py' runs the trained model that predicts input sign language to its text. 'classifier_neww.json' and 'classifier.h5' hdf5 file contains model architecture and pre-trained model network weights. Deep Learning model uses Keras and Tensorflow backend to train model. 'imgprocessing.py' contains the python code to perform the image processing tasks on the dataset images using OpenCV.

### Run

In a terminal or a Command Prompt, navigate to the folder containg datasets and codes and run the following command:
```bash
python3 Imagetest.py
``` 

This will start your webcam and a window for a real-time sign to on-screen text conversion. The model is trained with an accuracy of **94%**.

### Data

Link to dataset: https://github.com/ardamavi/Sign-Language-Digits-Dataset






