import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, Sequential
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import gc

train_path = '/kaggle/input/tusimple-dataset-preprocessed/'
img_generator = keras.preprocessing.image.ImageDataGenerator()
images_set = img_generator.flow_from_directory(
    train_path,
    shuffle=False,
    batch_size=64,
    class_mode='binary',
    target_size=(256, 320)
)

num_images = 7252
num_batches = num_images // 64 + 1
X, Y = [], []

for i in range(num_batches):
    batch = next(images_set)
    batch_images, batch_labels = batch[0], batch[1]
    for ind, lb in enumerate(batch_labels):
        if lb == 0: 
            X.append(batch_images[ind])
        else:
            Y.append(np.mean(batch_images[ind], axis=2))
    if i % 10 == 0:
        print(f'Batch {i}')

X, Y = np.array(X), np.array(Y)
X, Y = shuffle(X, Y, random_state=100)
Y = (Y >= 100).astype('int').reshape(-1, 256, 320, 1)
X, Y = np.array(X[:2000]), np.array(Y[:2000])
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=.1, random_state=100)

del X, Y, images_set
gc.collect()

plt.figure(figsize=(10, 40))
s, e = 80, 84
index = 1

for i, j in zip(X_train[s:e], Y_train[s:e]):
    plt.subplot(10, 2, index)
    plt.imshow(i/255.)
    plt.subplot(10, 2, index+1)
    plt.imshow(j, cmap='gray')
    index += 2

def unet(input_size=(256, 320, 3)):
    inputs = layers.Input(input_size)
    rescale = keras.layers.Rescaling(1./255)(inputs)

    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(rescale)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)
    
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2,2))(drop5))
    merge6 = layers.Concatenate(axis=3)([conv4, up6])
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    
    up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2,2))(conv6))
    merge7 = layers.Concatenate(axis=3)([conv3, up7])
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2,2))(conv7))
    merge8 = layers.Concatenate(axis=3)([conv2, up8])
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    up9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2,2))(conv8))
    merge9 = layers.Concatenate(axis=3)([conv1, up9])
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    return models.Model(inputs=inputs, outputs=outputs)

model = unet()
model.compile(optimizer='adam', loss=keras.losses.BinaryFocalCrossentropy(), metrics=['accuracy'])
model.summary()

epochs = 32
batch_size = 8
callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")]

model.fit(X_train, Y_train, epochs=epochs, callbacks=callbacks, validation_data=(X_val, Y_val), batch_size=batch_size)

!mkdir out
plt.figure(figsize=(10, 45))
s, e = 90, 98
index = 1

preds = model.predict(X_val)
preds = (preds >= .5).astype('int')

for i, j, k in zip(X_val[s:e], preds[s:e], Y_val[s:e]):
    cv2.imwrite(f'./out/img-{index}.jpg', i)
    cv2.imwrite(f'./out/pred-{index}.jpg', j*255.)
    cv2.imwrite(f'./out/ground-{index}.jpg', k*255.)
    
    plt.subplot(10, 2, index)
    plt.imshow(i/255.)
    
    plt.subplot(10, 2, index+1)
    plt.imshow(j, cmap='gray')
    index += 2

accuracy = tf.keras.metrics.Accuracy()
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
iou = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])

accuracy.update_state(Y_val, preds)
accuracy_value = accuracy.result().numpy()

precision.update_state(Y_val, preds)
precision_value = precision.result().numpy()

recall.update_state(Y_val, preds)
recall_value = recall.result().numpy()

f1_score = 2 / ((1 / precision_value) + (1 / recall_value))

iou.update_state(Y_val, preds)
iou_value = iou.result().numpy()

print("Accuracy:", accuracy_value)
print("Precision:", precision_value)
print("Recall:", recall_value)
print('F1 Score:', f1_score)
print('IoU:', iou_value)
