import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

from keras import layers
from keras import models
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
 
import tensorflow as tf
import cv2                  
import numpy as np  
from tqdm import tqdm
import os

X=[]
Z=[]
IMG_SIZE=75
BLANK_DIR= r'chess-dataset\figures\o'
BISHOP_DIR= r'chess-dataset\figures\b'
KING_DIR= r'chess-dataset\figures\k'
KNIGHT_DIR= r'chess-dataset\figures\n'
PAWN_DIR= r'chess-dataset\figures\p'
QUEEN_DIR= r'chess-dataset\figures\q'
ROOK_DIR= r'chess-dataset\figures\r'

def make_train_data(figure_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        X.append(np.array(img))
        Z.append(str(figure_type))

make_train_data('o', BLANK_DIR)
make_train_data('b', BISHOP_DIR)
make_train_data('k', KING_DIR)
make_train_data('n', KNIGHT_DIR)
make_train_data('p', PAWN_DIR)
make_train_data('q', QUEEN_DIR)
make_train_data('r', ROOK_DIR)

print(len(X))

le = LabelEncoder()
Y = le.fit_transform(Z)
Y = tf.keras.utils.to_categorical(Y, 7)
X = np.array(X)
X = X/255.
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.25, random_state=42)

train_datagen = ImageDataGenerator(
      rotation_range=40,
      shear_range=0.2,
      zoom_range=0.2,
      vertical_flip=True,
      horizontal_flip=True)

train_datagen.fit(x_train)

base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation ='relu')(x)
predictions = layers.Dense(7, activation ='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

adam = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()

History = model.fit_generator(train_datagen.flow(x_train,y_train, batch_size=500),
                              epochs = 15, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // 500)

model.save('model.h5')