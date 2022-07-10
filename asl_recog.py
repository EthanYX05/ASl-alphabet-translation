import tensorflow as tf
import numpy as np
import jetson.utils as jsu
from time import sleep




# My constants
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_TYPE = "/dev/video0"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
LAZIZ_LABELS = ["b", "bg", "c","d","delete","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","a","t","u","v","w","x","y","z"]



# Function to convert RGBA to RGB, published on Stackoverflow by Feng Wang
# https://stackoverflow.com/questions/50331463/convert-rgba-to-rgb-in-python
def rgba2rgb(rgba, background=(255,255,255)):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, "RGBA image has 4 channels"
    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    a = np.asarray(a, dtype='float32') / 255.0
    R, G, B = background
    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B
    return np.asarray(rgb, dtype="uint8")

# Load savedmodel from the Google's Teachable Machine Web site
saved_model = tf.keras.models.load_model("model.savedmodel")
nano_model = saved_model.signatures["serving_default"]

# Create the camera
camera = jsu.gstCamera(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_TYPE)

temp_output =["init"]


# Process frames and recognise objects until user exits
while True:
    if len(temp_output) >13:
         temp_output =["init"]
    # Capture the image
    img, width, height = camera.CaptureRGBA(zeroCopy=1)

    # Resize the image to the ML input requirements
    nano_image = rgba2rgb(jsu.cudaToNumpy(img, width, height, 4)) # Convert PyCapsule into NP array
    nano_image = tf.image.resize_with_pad(nano_image, IMAGE_HEIGHT, IMAGE_WIDTH)
    nano_image = np.expand_dims(nano_image, axis=0)

    # Classify the image
    prediction = saved_model.predict(nano_image)

    # Find the prediction confidence and the object description
    confidence = max(prediction[0])
    conf_index = np.argmax(prediction[0])
    class_desc = LAZIZ_LABELS[conf_index]

    if class_desc != temp_output[-1]:
        if class_desc == "bg":
            print("no input detected")
            sleep(2)
            continue
        elif class_desc == "delete":
            del temp_output[-1]
            print("recent input has been deleted")
            print(temp_output)
            sleep(2)
        else:
            print("input " + class_desc + " has been added")
            temp_output.append(class_desc)
            print(temp_output)
            sleep(2)

