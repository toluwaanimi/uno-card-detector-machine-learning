import random
import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from keras.utils.image_utils import load_img as load, img_to_array
from sklearn.utils import Bunch
from time import sleep
import os
import sys

card_colors = ["r", "y", "g", "b"]
colors = ["RED", "GREEN", "BLUE", "YELLOW"]
card_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "d", "a", "n"]
main_src_folder = './img/'
src_folder = './train/1/'
color_clf = KNeighborsClassifier(2)
number_clf = KNeighborsClassifier(2)
resX = 600
resY = 800


def add_noise_to_image(img):
    """
    Add Gaussian noise to an image.

    Parameters:
        img (numpy.ndarray): An image as a numpy array.

    Returns:
        numpy.ndarray: The image with Gaussian noise added.

    """
    # Set the maximum variability of the noise
    max_variability = 10

    # Generate a random deviation for the noise
    deviation = max_variability * random.random()

    # Create Gaussian noise with the deviation and image shape
    noise = np.random.normal(0, deviation, img.shape)

    # Add the noise to the image
    img_with_noise = img + noise

    # Clip the image pixel values to lie between 0 and 255
    np.clip(img_with_noise, 0., 255.)

    return img_with_noise


def get_image_data_generator():
    """
    Creates and returns an instance of the ImageDataGenerator class with specific augmentations and a preprocessing function.

    Returns:
        keras.preprocessing.image.ImageDataGenerator: An instance of the ImageDataGenerator class.
    """
    # Create an instance of the ImageDataGenerator class with the following parameters:
    # - rescale the pixel values to lie between 0 and 1
    # - rotate images randomly by up to 20 degrees
    # - shift the image width and height by up to 10%
    # - zoom the image by up to 10%
    # - apply the addNoise function to add Gaussian noise to the images
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        preprocessing_function=add_noise_to_image,
    )

    return datagen


def get_colored_image_augmentation(img):
    """
    Apply color augmentation to an image by randomly changing the saturation and value of the image in the HSV color space.

    Parameters:
        img (numpy.ndarray): An image as a numpy array.

    Returns:
        numpy.ndarray: The color augmented image as a numpy array.

    """
    # Generate a random value to multiply the saturation and value by
    value = random.uniform(0.6, 1.3)

    # Convert the image from RGB to HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    # Convert the HSV array to float64 for calculations
    hsv = np.array(hsv, dtype=np.float64)

    # Multiply the saturation and value of the image by the random value
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

    # Convert the HSV array back to uint8 format
    hsv = np.array(hsv, dtype=np.uint8)

    # Convert the image from HSV to RGB color space
    img = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

    return img


def convert_image_to_numpy_array(img):
    """
    Convert an image to a numpy array.

    Parameters:
        img (PIL.Image): An image as a PIL.Image object.

    Returns:
        numpy.ndarray: The image as a numpy array.
    """
    # Convert the image to a numpy array
    return np.array(img)


def convert_numpy_array_to_image(arr):
    """
    Convert a numpy array to an image.

    Parameters:
        arr (numpy.ndarray): An image as a numpy array.

    Returns:
        PIL.Image: The image as a PIL.Image object.
    """
    # If the array data type is not uint8, convert it to uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # Convert the numpy array to an image using the PIL library
    return Image.fromarray(arr)


def generate_noisy_images(img, num_duplicates):
    """
    Generates a list of noisy images using data augmentation.

    Parameters:
        img (numpy.ndarray): An image as a numpy array.
        num_duplicates (int): The number of duplicate images to generate.

    Returns:
        list: A list of noisy images as numpy arrays.
    """
    # Initialize an empty list to hold the augmented images.
    augmented_images = []

    # Reshape the image to match the expected input shape of the data generator.
    img = img.reshape((1,) + img.shape + (1,))

    # Get a data generator with noise augmentation.
    datagen = get_image_data_generator()

    # Generate new augmentations until the desired number has been reached.
    for batch in datagen.flow(img, batch_size=1):
        # Extract the augmented image from the batch and remove the batch dimension.
        augmented_image = np.squeeze(batch, axis=0)

        # Reshape the augmented image to match the original image shape.
        augmented_image = augmented_image.reshape((150, 150))

        # Add the augmented image to the list of augmented images.
        augmented_images.append(augmented_image)

        # Stop generating new augmentations once the desired number has been reached.
        if len(augmented_images) >= num_duplicates:
            break

    # Return the list of augmented images.
    return augmented_images


def capture_cards(vc):
    """
    Captures images of cards using a video capture device.

    Parameters:
        vc (cv2.VideoCapture): A video capture device.

    Returns:
        None
    """
    # Loop through each card color and number combination and capture an image of the card.
    for color in card_colors:
        for number in card_numbers:
            # Prompt the user to show the card.
            print('Please show: ' + color + str(number))
            card = ""
            while card == "":
                card = input()

            # Capture an image from the video capture device.
            rval, frame = vc.read()

            # Save the captured image to the specified folder.
            cv.imwrite(src_folder + color + str(number) + '.jpg', frame)

            # Print a message to indicate that the image has been saved.
            print('Saved: ' + color + str(number))

            # Display the captured image.
            plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            plt.pause(0.001)

    # Release the video capture device.
    vc.release()


def collect_card_data(cam_index):
    """
    Collects image data of cards from a video stream captured from the camera with the specified index.

    Args:
    - cam_index: An integer representing the index of the camera to be used for data collection.

    Returns:
    - None
    """
    print("Opening camera with index " + str(cam_index) + "...")
    vc = cv.VideoCapture(cam_index)
    vc.set(3, 960)
    vc.set(4, 1280)
    while vc.isOpened():
        # capture data for each card color and number
        capture_cards(vc)
        # break out of the loop if the 'Esc' key is pressed
        k = cv.waitKey(1)
        if k == 27:
            break
    # close all windows and release the camera
    cv.destroyAllWindows()
    cv.VideoCapture(0).release()


def findContour(frame):
    # Convert RGB image to grayscale
    mono = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    # Apply blurring to the grayscale image
    blur = cv.blur(mono, (10, 10))

    # Apply adaptive thresholding to the blurred image
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 35, 2)

    # Apply morphological opening to the thresholded image
    kernel = np.ones((5, 5), np.uint8)
    close = cv.morphologyEx(th, cv.MORPH_OPEN, kernel)

    # Find contours in the morphologically opened image
    cont, t = cv.findContours(close, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Select the contour with the most number of points that is smaller than a specified size
    cnt = ()
    maxCnt = 1600
    for c in cont:
        if len(cnt) < len(c) < maxCnt:
            cnt = c

    return cnt


def getBinaryImage(frame):
    """
    Takes an RGB image as input, converts it to grayscale, applies a Gaussian blur, and applies adaptive thresholding
    to produce a binary image.

    Parameters:
    frame (numpy.ndarray): The input RGB image as a numpy array.

    Returns:
    numpy.ndarray: The resulting binary image as a numpy array.
    """
    # Convert the RGB image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    # Apply a Gaussian blur to the grayscale image
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to the blurred grayscale image
    binary = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 35, 2)

    return binary


def getCardCrops(frame):
    """Extract card, color, and number crops from an input frame based on a contour.

    Args:
        frame (numpy.ndarray): Input frame as a NumPy array.

    Returns:
        list: List of NumPy arrays containing the card crop, color crop, and number crop, respectively.
    """
    contour = findContour(frame)
    (x, y), (MA, ma), angle = cv.fitEllipse(contour)
    card_height = int(MA / 1.2)
    card_width = int(ma / 1.28)
    card_x = int((x - card_height / 2))
    card_y = int((y - card_width / 2))
    card_crop = frame[card_x:card_x + card_width, card_y:card_y + card_height]

    color_x = int(ma / 4.7)
    color_y = int(MA / 2.5)
    color_width = color_height = 2
    color_crop = card_crop[color_x:color_x + color_width, color_y:color_y + color_height]

    number_x = int(ma / 4.5)
    number_y = int(MA / 5.8)
    number_width = number_height = 150
    number_crop = card_crop[number_x:number_x + number_width, number_y:number_y + number_height]

    # number_crop_resized = cv2.resize(number_crop, (150, 150))

    return [card_crop, color_crop, number_crop]


def rotateFrame(frame):
    """
    Rotates a playing card image based on the orientation of its contour.

    Args:
    - frame: a numpy array representing a playing card image.

    Returns:
    - A numpy array representing the rotated playing card image.
    """
    # Get the contour and fit an ellipse to it
    cnt = findContour(frame)
    (x, y), (MA, ma), angle = cv.fitEllipse(cnt)

    # Get the height and width of the frame
    (h, w) = frame.shape[:2]

    # Set the center of rotation and adjust angle if necessary
    center = (x, y)
    if angle > 90:
        angle = 180 + abs(angle)

    # Set the scaling factor and rotation matrix
    scale = 1
    M = cv.getRotationMatrix2D(center, angle, scale)

    # Apply the rotation and return the rotated frame
    rFrame = cv.warpAffine(frame, M, (w, h))
    return rFrame


def displayCardData(frame, rFrame, cardCrop, colorCrop, numberCrop):
    """
    Displays the input frame and different crops of a playing card, namely the rotated frame, the entire card crop,
    the color region crop, and the number/face region crop.

    Args:
    - frame: a numpy array representing a playing card image.
    - rFrame: a numpy array representing the rotated frame of the playing card.
    - cardCrop: a numpy array representing the cropped image of the entire playing card.
    - colorCrop: a numpy array representing the cropped image of the color region of the playing card.
    - numberCrop: a numpy array representing the cropped image of the number/face region of the playing card.

    Returns:
    - None
    """
    print(plt.imshow(frame))
    plt.pause(0.001)
    print(plt.imshow(rFrame))
    plt.pause(0.001)
    print(plt.imshow(cardCrop))
    plt.pause(0.001)
    print(plt.imshow(colorCrop))
    plt.pause(0.001)
    print(plt.imshow(numberCrop, cmap='Greys_r'))
    plt.pause(0.001)


def predictCardColor(classifier, card_image):
    """
    Predicts the color of a playing card image using a trained classifier.

    Args:
    - classifier: a trained machine learning classifier
    - card_image: a numpy array representing a playing card image

    Returns:
    - A string representing the predicted color of the card, either "red" or "black".
    """
    color = ""
    prediction = classifier.predict(card_image.data)  # predict the color of the card
    color += colors[prediction[0]]  # get the predicted color
    return color


def predictCardNumber(clf, card):
    """
    Predicts the number or face value of a playing card image.

    Args:
    - clf: a trained machine learning model used for prediction.
    - card: a Card object representing the playing card image.

    Returns:
    - A string representing the predicted number/face value of the playing card.
    """
    number = ""
    prediction = clf.predict(card.data)
    number += str(card_numbers[prediction[0]])  # get the predicted number/face value from the cardNumbers dictionary
    return number


def extractCardData(isLive, frame):
    """
    Extracts the card data (number and color) from a playing card image.

    Args:
    - isLive: a boolean indicating if the image is being captured live.
    - frame: a numpy array representing a playing card image.

    Returns:
    - A list of two elements: [outputFrame, cardData], where:
        - outputFrame is the original image with bounding box and text overlayed on it.
        - cardData is a string containing the card's number and color in the format "number color".
    """
    cardData = ""
    contour = findContour(frame)
    rotatedFrame = rotateFrame(frame)
    rotatedContour = findContour(rotatedFrame)
    rCardCrop, colorCrop, numberCrop = getCardCrops(rotatedFrame)  # extract card crops
    if numberCrop.shape == (150, 150, 3):
        numberCrop = getBinaryImage(numberCrop)
        displayCardData(frame, rotatedFrame, rCardCrop, colorCrop, numberCrop)  # log data
        # predict color and number
        pcol = Bunch(data=colorCrop)
        pcol = (pcol.data).reshape(1, 2 * 2 * 3)
        col = predictCardColor(color_clf, pcol)
        pnum = Bunch(data=numberCrop)
        pnum = (pnum.data).reshape(1, 150 * 150)
        num = str(predictCardNumber(number_clf, pnum))
        cardData = num + " " + col
    # draw bounding box and text on original image
    x, y, w, h = cv.boundingRect(np.asarray(contour))
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv.putText(frame, cardData, (x + 5, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (38, 38, 38), 2)
    cv.drawContours(frame, contour, -1, (255, 0, 0), 1)
    cv.drawContours(frame, rotatedContour, -1, (0, 0, 255), 1)
    return [frame, cardData]


def train_card_recognition_models():
    """
    This function trains the color and number recognition models for card detection using machine learning.
    """
    # Initialize global variables for the models
    global color_clf
    global number_clf

    # Load and process color images for training the color model
    print("")
    print("Loading 5200 color images...")
    color_data = []
    color_target = []
    for number in card_numbers:
        target = []
        for color in card_colors:
            try:
                # Load and preprocess image
                image = np.asarray(load('img/' + color + str(number) + '.jpg', target_size=(resX, resY)))
                card_crop, color_crop, number_crop = getCardCrops(rotateFrame(image))

                # Augment data for better training
                for i in range(0, 100):
                    color_data.append(get_colored_image_augmentation(color_crop))
                    target.append(card_colors.index(color))
            except IOError:
                break
        color_target += target

    # Create data set for color model
    color_dataset = Bunch(data=np.array(color_data), target=np.array(color_target))
    X = color_dataset.data.reshape(len(color_data), 2 * 2 * 3)
    y = color_dataset.target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, train_size=.75, random_state=42)
    print(np.shape(X_train), np.shape(X_test), len(y_train), len(y_test))

    # Train the color model and get accuracy scores
    color_clf.fit(X_train, y_train)
    color_train_acc = color_clf.score(X_train, y_train)
    color_test_acc = color_clf.score(X_test, y_test)

    # Load and process number images for training the number model
    print("")
    print("Loading 5200 real + augmented numbers images...")
    number_data = []
    number_target = []
    for color in card_colors:
        target = []
        for number in card_numbers:
            try:
                # Load and preprocess image
                image = np.asarray(load('img/' + color + str(number) + '.jpg', target_size=(resX, resY)))
                card_crop, color_crop, number_crop = getCardCrops(rotateFrame(image))
                number_crop = getBinaryImage(number_crop)
                card_crop = image[225:590, 225:470]
                number_crop = getBinaryImage(card_crop[110:260, 50:200])

                # Augment data for better training
                number_data += generate_noisy_images(number_crop, 100)
                for i in range(0, 100):
                    target.append(card_numbers.index(number))
            except IOError:
                break
        number_target += target

    # Create data set for number model
    number_dataset = Bunch(data=np.array(number_data), target=np.array(number_target))
    X = number_dataset.data.reshape(len(number_data), 150 * 150)
    y = number_dataset.target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, train_size=.75, random_state=42)
    print(np.shape(X_train), np.shape(X_test), len(y_train), len(y_test))

    # Train the number model and get accuracy scores
    print("")
    print("Displaying real and augmented samples...")
    number_clf.fit(X_train, y_train)
    number_train_acc = number_clf.score(X_train, y_train)
    number_test_acc = number_clf.score(X_test, y_test)
    i = 0
    for i in range(0, 400):
        if (-1 < i < 10) or (100 < i < 110) or (200 < i < 210) or (300 < i < 310):
            print(plt.imshow(color_data[i]))
            plt.pause(0.001)
        i += 1
    i = 0
    for i in range(0, 1600):
        if (-1 < i < 10) or (100 < i < 110) or (200 < i < 210) or (300 < i < 310):
            print(plt.imshow(number_data[i], cmap='Greys_r'))
            plt.pause(0.001)
        i += 1
    print("")
    print("Train and Test color and number models...")
    print("100%")
    sleep(2)
    print("")
    print("ML results...")
    print("COLORS    train-acc " + ('{:.2f}'.format(color_train_acc)) + "       test-acc " + (
        '{:.2f}'.format(color_test_acc)))
    print("NUMBERS   train-acc " + ('{:.2f}'.format(number_train_acc)) + "       test-acc " + (
        '{:.2f}'.format(number_test_acc)))


def get_user_option():
    """
    Get user option for Uno card processing.

    Returns:
        str: User-selected option.
    """
    print()
    print("----------------------------")
    print("*** Uno Card Processing ***")
    print("----------------------------")
    print("[1] Read Uno card images from file")
    print("[2] Capture Uno card images from live camera stream")
    print("[3] Capture custom Uno card images")
    print("[0] Exit")
    option = input("Enter your choice (1, 2, 3, or 0): ")
    return option


def get_read_file_submenu_option():
    print("\n--- File Reading Options ---")
    print("What would you like to do?")
    print("[1] Select a specific card to read")
    print("[2] Read all data from file")
    print("[9] Return to main menu")
    option = input("Enter the number corresponding to your choice: ")
    return option


def get_camera_index():
    print("Please enter the index number of your camera:")
    index = input("Example: 0, 1, 2, etc. : ")
    return int(index)


def get_user_card():
    print("Please choose a card:")
    print("Colors: Red (r), Green (g), Blue (b), Yellow (y)")
    print("Numbers: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, Draw Two (d), Skip (a), Reverse (n)")
    color = input("Color: ")
    number = input("Number: ")
    return color + number


def live_stream(camera_index):
    print("Opening camera with index " + str(camera_index) + "...")
    video_capture = cv.VideoCapture(camera_index)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 800)
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
    frame_counter = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if frame_counter > 20:
            frame = np.asarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            frame, detected_card = extractCardData(True, frame)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow("frame", frame)
        frame_counter += 1
        key = cv.waitKey(1)
        if key == 27:
            break
    cv.destroyWindow("stream")
    video_capture.release()


def readFromFile(imgName):
    # Check if file exists
    if os.path.isfile("img/" + imgName + ".jpg"):
        # Load image and convert to numpy array
        frame = np.asarray(load('img/' + imgName + '.jpg', target_size=(600, 800)))
        # Print card info
        print("Card Info: " + str(imgName))
        # Get detected card information
        frame, detectedCard = extractCardData(False, frame)

        # Display image and detected card information
        print(plt.imshow(frame))
        print("Detected as: " + str(detectedCard))
        plt.pause(0.001)


print("")
print("Initializing Uno Machine Learning models..")
train_card_recognition_models()

opt = get_user_option()
while opt != "0":
    if opt == "1":
        readOpt = get_read_file_submenu_option()
        while readOpt != "0":
            if readOpt == "1":
                readFromFile(get_user_card())
            elif readOpt == "2":
                for c in card_colors: [readFromFile(c + str(n)) for n in card_numbers]
            elif readOpt == "9":
                break
            elif readOpt == "0":
                sys.exit("Terminated")
            else:
                print("Invalid option !")
            readOpt = get_read_file_submenu_option()
    elif opt == "2":
        live_stream(get_camera_index())
    elif opt == "3":
        collect_card_data(get_camera_index())
    elif opt == "0":
        sys.exit("Terminated")
    else:
        print("Invalid option !")
    opt = get_user_option()
