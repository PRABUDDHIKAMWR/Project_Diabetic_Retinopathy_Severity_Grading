import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Specify image size
IMG_WIDTH = 128
IMG_HEIGHT = 128
CHANNELS = 3

def crop_image_from_gray(img, tol=7):
    """
    Applies masks to the orignal image and 
    returns the a preprocessed image with 
    3 channels
    
    :param img: A NumPy Array that will be cropped
    :param tol: The tolerance used for masking
    
    :return: A NumPy array containing the cropped image
    """
    # If for some reason we only have two channels
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    # If we have a normal RGB images
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_image(image, sigmaX=10):
    """
    The whole preprocessing pipeline:
    1. Read in image
    2. Apply masks
    3. Resize image to desired size
    4. Add Gaussian noise to increase Robustness
    
    :param img: A NumPy Array that will be cropped
    :param sigmaX: Value used for add GaussianBlur to the image
    
    :return: A NumPy array containing the preprocessed image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Display the image
    # plt.imshow(image)
    # plt.title('Test Image')
    # plt.show()
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)
    # plt.imshow(image)
    # plt.title('Test Image')
    # plt.show()
    
    return image

data = pd.read_csv("E:/trainingset/archive/trainLabels.csv")

images = []
labels = []

for index, row in data.iterrows():
    image = cv2.imread('E:/trainingset/archive/resized_train/' + row['image'] + '.jpeg')
    images.append(preprocess_image(image))
    labels.append(row['level'])

fig, ax = plt.subplots(1, 5, figsize=(15, 6))
for i in range(5):
    sample = data[data['level'] == i].sample(1)
    image_name = sample['image'].item()
    img = cv2.imread('E:/trainingset/archive/resized_train/' + image_name + '.jpeg')
    X = preprocess_image(img)
    ax[i].set_title(f"Image: {image_name}\n Label = {sample['level'].item()}", 
                    weight='bold', fontsize=10)
    ax[i].axis('off')
    ax[i].imshow(X);
plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

""" model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(np.array(X_train), np.array(y_train), batch_size=128, epochs=10, verbose=1)

score = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) """

# Convert lists to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Normalize the data
X_train = X_train / 255
X_test = X_test / 255

# One-hot encode labels
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print("1")
# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))  # 5 classes for diabetic retinopathy grading

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

# Evaluate the model
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Predict on a single image
image_index = 0  # Index of the image in the test dataset
single_image = np.expand_dims(X_test[image_index], axis=0)
predicted_label = model.predict(single_image)
print('Predicted label:', np.argmax(predicted_label))

# Display the image
plt.imshow(X_test[image_index])
plt.title('Test Image')
plt.show()
