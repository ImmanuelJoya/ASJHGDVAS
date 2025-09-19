# Neural Networks

#TenserFlow and Keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# the dataset that we're gonna use is the fashion mnist dataset
fashion_mnist = keras.datasets.fashion_mnist #importing the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() #loading the dataset into training and testing sets

train_images.shape # (60000, 28, 28)
train_images[0, 23, 23] # pixel value of the first image
#print(train_images) # prints all the pixel values of all the images

train_labels # labels for the training set

# there are 10 different classes in the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# mapping the labels to the class names
plt.figure()``
plt.imshow(train_images[5]) # display the first image in the training set
plt.colorbar() # display the color bar
plt.grid(False) # remove the grid
plt.show() # show the image

# Preprocessing the data
# scaling the images to a range of 0 to 1 before feeding them to the neural network model
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # input layer, flattens the 28x28 images into 1D array of 784 pixels
    keras.layers.Dense(128, activation='relu'), # hidden layer with 128 neurons and ReLU activation function
    keras.layers.Dense(10, activation='softmax') # output layer with 10 neurons (one for each class) and softmax activation function
])

# Compile the model
model.compile(optimizer='adam', # optimizer
                loss='sparse_categorical_crossentropy', # loss function
                metrics=['accuracy']) # metric to monitor

# Training the model
model.fit(train_images, train_labels, epochs=10) # training the model for 10 epochs

# Evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2) # evaluating the model on the test set
print('\nTest accuracy:', test_acc) # printing the test accuracy

# Making predictions
predictions = model.predict(test_images) # making predictions on the test set
print(class_names[np.argmax(predictions[0])]) # printing the predicted class for the first test image
plt.figure()
plt.imshow(test_images[0]) # display the first test image
plt.colorbar() # display the color bar
plt.grid(False) # remove the grid
plt.show() # show the image

#Varifying the prediction
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = model.predict(np.arry([image])) 
    predicted_class = class_names[np.argmax(prediction)]

    show_image(image, class_names[correct_label], predicted_class)

def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Actual: " + label + " Guess: " + guess)
    plt.axis('off')
    plt.show()

def get_number():
    while True:
        num = input("Enter a number between 0 and 9999: ")
        if num.isdigit() and 0 <= int(num) <= 9999:
            return int(num)
        else:
            print("Invalid input. Please try again.")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
   