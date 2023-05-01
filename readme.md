# Uno Card Processing using Machine Learning 

This is a python that uses machine learning algorithms to detect and process Uno cards. The user can select from several options, including reading Uno card images from a file, capturing Uno card images from a live camera stream, and capturing custom Uno card images. 

# Video Display

https://youtu.be/pnI4YcnS-kA

## Get Started
Prerequisites

To run this project, you will need:

- Python 3.7 or higher
- OpenCV
- TensorFlow 2.x
- Keras
- Numpy
- Matplotlib
- Scikit-learn

## Github / Gitlab 
- https://labcode.mdx.ac.uk/ea1119/uno-card-detector
- https://github.com/toluwaanimi/uno-card-detector-machine-learning


### Installation

1. Install Python 3.7 or higher.
2. Install OpenCV, Matplotlib, Numpy, TensorFlow, and Keras using pip:

```
pip install opencv-python matplotlib numpy tensorflow keras
```

### Running the Code

To run the code, open the terminal and navigate to the project directory. Then, enter the following command:

```
python uno_card_processing.py
```

## Usage
1. Clone the repository to your local machine.
2. Install the required packages by running the command `pip install -r requirements.txt`.
3. Run the `uno_card_processing.py` file with `python uno_card_processing.py` command in your terminal.
4. Select one of the options from the menu to start processing the Uno cards.

### Menu options
1. Read Uno card images from file: This option allows the user to select a specific card to read, read all data from file, or return to the main menu.
2. Capture Uno card images from live camera stream: This option opens the live camera stream to capture the Uno cards.
3. Capture custom Uno card images: This option allows the user to capture custom Uno card images.
4. Exit: This option terminates the program.

### File structure
- The `img` directory contains the images of Uno cards.
- The `uno_card_processing.py` file contains the main code for processing the Uno cards.

### Guidelines
- When reading the cards from a file, provide the color and number of the card to read.
- When capturing custom Uno card images, ensure that the card is properly positioned and centered in the frame.
- When capturing Uno card images from the live camera stream, ensure that the camera is positioned and focused properly.
- Press the `Esc` key to stop the live camera stream.
The program will start, and the user will be presented with several options. The user can select from the options by entering the corresponding number and pressing Enter.

