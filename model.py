import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# emnist_train, info = tfds.load('emnist', split='train', as_supervised=True, with_info=True)
# emnist_test = tfds.load('emnist', split='test', as_supervised=True)
#
# emnist_train = emnist_train.map(lambda img, label: (tf.image.convert_image_dtype(img, tf.float32), label))
# emnist_test = emnist_test.map(lambda img, label: (tf.image.convert_image_dtype(img, tf.float32), label))
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
# model.add(tf.keras.layers.MaxPooling2D((2,2)))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(info.features['label'].num_classes, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(emnist_train.batch(32), epochs=10)
#
# model.save('handwritten.model')

# Load the saved model
model = tf.keras.models.load_model('handwritten.model')

# Load the image
img = cv2.imread('testSample/img_50.jpg')

# Preprocess the image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))
img = img.reshape((1, 28, 28, 1))
img = img.astype('float32') / 255.0

# Predict the class probabilities
probs = model.predict(img)[0]

# Get the predicted class index
class_idx = np.argmax(probs)

# Map the index to the actual character/digit
char_digit_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a',
    37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r',
    46: 't'
}
predicted_char_digit = char_digit_map[class_idx]

# Print the predicted character/digit
print('The predicted character/digit is:', predicted_char_digit)

