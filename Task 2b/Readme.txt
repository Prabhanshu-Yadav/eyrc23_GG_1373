 # Task 2B README

## Team Information
- **Team ID:** GG_1373
- **Team Members:** Bhagyesh Dhamandekar, Om Rajput, Prabhanshu Yadav, Ayush Tiwari

## Table of Contents
- [Introduction](#introduction)
- [Task Description](#task-description)
- [Project Structure](#project-structure)
- [Running the Code](#running-the-code)
- [Steps Taken](#steps-taken)
- [Output](#output)
- [File Descriptions](#file-descriptions)
- [Acknowledgments](#acknowledgments)
- [Conclusion](#conclusion)

## Introduction
This README provides an overview of our work on Task 2B of the GeoGuide(GG) Theme (eYRC 2023-24). Task 2B involves classifying images into different events related to alien attacks on the planet.

## Task Description
In Task 2B, our goal was to classify images into five event categories caused by alien attacks. The event categories are as follows:
- Fire
- Destroyed buildings
- Humanitarian Aid and rehabilitation
- Military Vehicles
- Combat

We have trained a model to classify images into these event categories.

## Project Structure
Our project structure includes the following main components:

- **task_2b.py**: This Python script contains the code for classifying events using the trained model.
- **gg_1373.h5**: This file stores our trained deep learning model.
- **events**: A directory where temporary image files are stored during classification.

## Running the Code
To run the code, please follow these steps:

1. Ensure you have Python and required libraries installed.
2. Activate your preferred Python environment.
3. Execute the `task_2b.py` script.
4. The script will classify the events from the test images and display the results on the console.

## Steps Taken
Our approach to completing Task 2B involved the following key steps:

1. **Data Preprocessing**: We prepared the dataset, which includes resizing images to a consistent size (180x180) and standardizing the data.

2. **Model Selection**: We selected a deep learning model architecture for event classification. In our case, we used a ResNet-based model.

3. **Model Training**: We trained the deep learning model using the provided dataset. The model was trained to classify images into the predefined event categories.

4. **Challenges Faced**: During the task, we faced challenges such as data preprocessing and optimizing the model for accuracy. We addressed these issues through experimentation.

## Output
The detected events will be printed on the console and also saved in the `detected_events.txt` file. Each detected event represents a classification of the corresponding input image into one of the predefined event classes: Combat, Humanitarian Aid and rehabilitation, Military Vehicles, Fire, and Destroyed buildings.

## File Descriptions
- **task_2b.py**: This file contains the code for classifying events using the trained model.
- **gg_1373.h5**: This file contains the trained deep learning model.
- **detected_events.txt**: A text file where detected events are saved.

## Acknowledgments
We would like to acknowledge the following:

- Python, TensorFlow, and other essential libraries used in this project.

This README file is part of our submission for Task 2B of the GeoGuide(GG) Theme (eYRC 2023-24).

 
