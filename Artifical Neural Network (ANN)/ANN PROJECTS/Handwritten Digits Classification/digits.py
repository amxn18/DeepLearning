import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from keras import datasets


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scaling the data
#------------------------------------------------
x_train = x_train / 255.0
x_test = x_test / 255.0
# print(len(x_train))
# print(len(x_test))

#28X28 Cross grid
# print(x_train[0].shape)
# print(x_test[0].shape)
# print(x_train[0])
# print(x_test[0])

# Checking the training data
#----------------------------
# plt.matshow(x_train[5])
# plt.title("First Image")
# # plt.show()
# print(y_train[5])

# FLattening the data into 1D array
#------------------------------------------------
# print(x_train.shape)  --> (60000, 28, 28)
x_trainRevised =x_train.reshape((len(x_train), 28 * 28))
x_testRevised =x_test.reshape((len(x_test), 28 * 28))
# print(x_trainRevised.shape)  --> (60000, 784)  (28*28 =784 )
# print(x_testRevised.shape)  --> (10000, 784)

# Creating the layers and model
#------------------------------------------------
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),  # Hidden layer
    keras.layers.Dense(10, activation='softmax')                      # Output layer
])

tbCallBack = tf.keras.callbacks.TensorBoard(log_dir = './logs', histogram_freq=1)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_trainRevised, y_train, epochs=5, callbacks = [tbCallBack]) # 5 epochs for training(no of iterations)
print("Model trained")

# Without Adding Hidden Layer Accuracy and  without scaling~ 88% and high loss
# With Adding Hidden Layer Accuracy and without scaling ~ 94% and low loss
# -->  With Adding Hidden Layer Accuracy and with scaling and 5 epochs ~ 98% and low loss

Evaluation = model.evaluate(x_testRevised, y_test)  # Evaluating the model on test data
print("Evaluation: ", Evaluation)  # Printing the evaluation results

# Model Prediction
#------------------------------------------------
predictions = model.predict(x_testRevised)  # Making predictions on test data
plt.matshow(x_test[0])  # Displaying the first test image
print("Raw Prediction Vector:", predictions[0])
plt.show()  # Showing the image
print("Predicted Number:",np.argmax(predictions[0]))  # Printing the predicted class for the first test image
print("Actual Number:",y_test[0])  # Printing the actual class for the first test image

# Use the different index to check the other images
#------------------------------------------------
# Model Saving
#------------------------------------------------
model.save('digits_model.h5')  # Saving the trained model to a file
# Loading the model
#------------------------------------------------
loaded_model = keras.models.load_model('digits_model.h5')  # Loading the saved model
loaded_model.summary()  # Displaying the model summary
