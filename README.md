# Magic Wand: Rock, Paper, Scissors

> **Note:** I worked on this project at a summer school in Hangzhou. In the first week, we made a line-follower car, and in the second week, we developed this rock, paper, scissors gesture recognition model.

## Project Description

This project implements a "Magic Wand" using an **ESP32-S3 microcontroller** and an **MPU6050 IMU**. The device recognizes hand gestures for the game "Rock, Paper, Scissors" using a machine learning model. It captures motion data, processes it through a Kalman filter to stabilize the signal, and runs a **TensorFlow Lite** model on-device to classify the gesture and display the result via LEDs.
<img width="1106" height="453" alt="image" src="https://github.com/user-attachments/assets/acdcc970-ecdd-4c66-b32b-1b8396e60ab2" />

## Languages and Usage

The project utilizes the following languages for specific purposes:

### C++ (Embedded System & Device Interaction)
* **Device Interaction:** Interfaces directly with the hardware, initializing and reading motion data from the MPU6050 sensor via I2C.
* **Signal Processing:** Implements a Kalman filter to clean raw sensor data, remove gravity components, and transform coordinates before inference.
* **Model Inference:** Loads and runs the TensorFlow Lite Micro interpreter to classify gestures directly on the device.
* **IO Control:** Manages physical inputs (buttons) and outputs (LEDs) to trigger recordings and show results.

### Python (Model Training)
* **Machine Learning:** Utilizes TensorFlow and Keras to define a Convolutional Neural Network (CNN) optimized for sensor data.
* **Data Processing:** Uses Pandas to read and structure the training and testing datasets from CSV files.
* **Conversion:** Converts the trained model into the TensorFlow Lite format (`.tflite`) required for the microcontroller.

### JSON (IDE Configuration)
* **Workspace Setup:** Configures the VS Code workspace folders and settings for a consistent development environment.
* **Recommendations:** Suggests necessary extensions (like PlatformIO IDE) to ensure all tools are available.

### INI (Configuration)
* **Project Setup:** Configures the PlatformIO build environment, specifying the board, framework, and library dependencies.

## Hardware Setup

The project is configured for the **4D Systems ESP32-S3 Gen4 R8N16** board.

**Pin Configuration:**
* **MPU6050 (I2C):** SDA on Pin `4`, SCL on Pin `5`
* **Input Button:** Pin `41` (Input Pullup)
* **Status LED:** Pin `42`
* **Gesture Output LEDs:**
    * 🪨 **Rock:** Pin `37`
    * ✂️ **Scissors:** Pin `36`
    * 📄 **Paper:** Pin `35`

<img width="500" height="673" alt="image" src="https://github.com/user-attachments/assets/07eb429d-8872-428e-8ffa-37c2ba133922" />

### Driver Installation (CH34x)
This development board uses the **CH34x USB-to-UART bridge** to communicate with the computer.
* **Purpose:** The driver allows your operating system to recognize the ESP32-S3 as a serial COM port (e.g., `COM3` on Windows, `/dev/ttyUSB0` on Linux).
* **Usage:** You must install this driver **before** attempting to upload code or use the Serial Monitor. Without it, the IDE will not be able to find the board.
* **Installation:** Download and install the specific **CH340/CH341** driver for your operating system (Windows/macOS/Linux).

## Software Dependencies

This project is built using **PlatformIO** with the **Arduino** framework. The following libraries are required and automatically handled by `platformio.ini`:

* `electroniccats/MPU6050` (v1.4.4)
* `tanakamasayuki/TensorFlowLite_ESP32` (v1.0.0)

## How to Play

1.  **Start Recording:** Press and release the button (Pin `41`). The status LED (Pin `42`) will turn **HIGH** to indicate recording has started.
2.  **Perform Gesture:** Make a "Rock," "Paper," or "Scissors" gesture with the device. The system records data for **2 seconds** at a sampling frequency of **100Hz**.
3.  **See Result:** The device will run the inference and light up the corresponding LED:
    * **Rock:** Pin `37` blinks.
    * **Scissors:** Pin `36` blinks.
    * **Paper:** Pin `35` blinks.

## Directory Structure

* `.vscode/`: VS Code configuration files for extensions and workspace settings.
* `src/`: Contains the main C++ source code (`main.cpp`) and model headers.
* `train/`: Contains the Python scripts (`train.py`) for training the neural network and the dataset CSV files (`data_x.csv`, `data_y.csv`).
* `DatasetPreparation/`: Collecting the data for training and testing the model.
* `lib/`: Directory for private libraries.
* `test/`: Unit testing files.
