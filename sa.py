# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from PIL import Image 
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
print(tf.__version__)
import cv2
import os
b=[""]
def clothes_detector(c):
  b[0]=c
  print(b[0])
  fashion_mnist = keras.datasets.fashion_mnist

  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'T-shirt/top', 'T-shirt/top', 'Ankle boot']

  train_images.shape


  len(train_labels)

  train_labels

  test_images.shape

  len(test_labels)

  train_images = train_images / 255.0

  test_images = test_images / 255.0

  plt.figure(figsize=(10,10))




  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
  ])


  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  model.fit(train_images, train_labels, epochs=10)


  test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

  print('\nTest accuracy:', test_acc)


  probability_model = tf.keras.Sequential([model, 
                                          tf.keras.layers.Softmax()])

  predictions = probability_model.predict(test_images)

  predictions[0]
  np.argmax(predictions[0])
  test_labels[0]
  def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

  def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')





  # Plot the first X test images, their predicted labels, and the true labels.
  # Color correct predictions in blue and incorrect predictions in red.




  # Grab an image from the test dataset.
  #img = test_images[1]
  a=os.getcwd()
  print(b[0])
  img = Image.open(b[0])
  img = img.resize((28, 28))
  img.save('image1.jpg')
  b[0]='image1.jpg'
  gray_image = cv2.cvtColor(np.array(Image.open(b[0])), cv2.COLOR_BGR2GRAY)
    #b[0]= np.array(Image.open(gray_image))
  img=gray_image
  img = img / 255.0
    #print(img.shape)

    # Add the image to a batch where it's the only member.
  img = (np.expand_dims(img,0))

  print(img.shape)

  predictions_single = probability_model.predict(img)

  print(predictions_single)
  print(class_names)
  i=0
  print(np.argmax(predictions_single[i]))
  return(class_names[np.argmax(predictions_single[i])])
  
    #plot_value_array(i, predictions_single[i],  test_labels)
    #plt.show()

    #plot_value_array(1, predictions_single[0], test_labels)
    #_ = plt.xticks(range(10), class_names, rotation=45)


    #np.argmax(predictions_single[0])
