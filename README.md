# End-to-End Self-Driving Car using CNN and OOP in Python

## Overview

This project implements an end-to-end deep learning model for autonomous driving using a Convolutional Neural Network (CNN). Inspired by NVIDIA's research, the model learns to map raw front-facing camera images directly to steering commands — without relying on separate modules for lane detection, object recognition, or path planning.

It also emphasizes Object-Oriented Programming (OOP) practices in Python to structure and manage the system components, making the project a practical application of OOP principles in the field of Mechatronics Engineering.

## Key Features

- End-to-end learning approach based on *Bojarski et al., 2016*
- Modular design using **Python OOP** (classes for augmentation, preprocessing, data pipeline)
- Image preprocessing: cropping, color conversion, normalization
- Data balancing using binning to reduce steering bias
- Real-time data augmentation: zoom, pan, brightness, and flipping
- CNN model trained on **Udacity’s driving dataset** (center/left/right cameras)
- Performance evaluation using training/validation loss
- Model export for deployment in simulators or real-world applications


## Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras**
- **NumPy / Pandas / Matplotlib**
- **Scikit-learn**
- **imgaug** for data augmentation

## Project Structure

```bash
Copy
Edit
├── data/
│   ├── IMG/                   # Image folder from simulator
│   └── driving_log.csv        # Steering angle logs
├── model/
│   └── model.h5               # Trained CNN model
├── main.py                    # Backend connection for the udacity simulator
├── Augmentation_Techniques.py # A Class containing different Augmentation techniques used in this project
├── self-driving-car.ipynb     # Notebook where we implmented the project
├── README.md                  # Project documentation
```
## Project setup

1. Download Udacity simulator and its requirements from [here.](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

2. Setup the project from this repo in the step that follows:

``` bash
git clone https://github.com/Zahir-Seid/Autonomous-Driving-with-python

cd Autonomous-Driving-with-python

pip install -r requirements.txt

uvicorn main:socket_app --host 0.0.0.0 --port 4567 --ws websockets

```
3. Start the simulator in Autonomous mode

## Model Architecture

The CNN model includes:

- 4 Convolutional layers with **ELU activation**
- 3 Dense layers followed by a final steering output neuron
- Optimized using **Adam optimizer** and **Mean Squared Error (MSE)** loss
- **Total parameters:** ~264,000  
- **Input size:** 200×66×3 (normalized images)



## Training Highlights

- **Dataset:** ~4,000 samples (balanced to ~1,460 after downsampling)
- **Train/Validation Split:** 80/20
- **Epochs:** 10  
- **Batch Size:** 100  
- **Augmentation:** Enabled during training only  
- **Final Validation Loss:** ~0.027



## Results

- Effective learning without significant overfitting
- Smooth convergence of training and validation loss
- Model successfully generalizes steering predictions from augmented images
- Ready for simulation-based or hardware deployment


## Educational Impact

This project serves as a hands-on demonstration of how **Object-Oriented Programming (OOP)** in Python is crucial for building intelligent **mechatronic systems**.  
It bridges theoretical software design with real-world applications like:

- Robotics  
- Embedded systems  
- Autonomous navigation; all key areas in **Mechatronics Engineering**.


## Reference

Bojarski, M. et al. (2016). *End to End Learning for Self-Driving Cars*. [arXiv:1604.07316](https://arxiv.org/abs/1604.07316)
