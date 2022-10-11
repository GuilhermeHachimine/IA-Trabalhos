from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import os

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 100, activation = 'relu'))
classificador.add(Dense(units = 100, activation = 'relu'))
classificador.add(Dense(units = 2, activation = 'softmax'))

classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

dados_treinamento = ImageDataGenerator(rescale = 1./255)
dados_teste = ImageDataGenerator(rescale = 1./255)

os.chdir(r'C:\Users\leoca\Documents\Python Scripts\CATS')

amostras_treinamento = dados_treinamento.flow_from_directory('treinamento',target_size = (64, 64),
 batch_size = 32, 
 class_mode = 'categorical')

amostras_teste = dados_treinamento.flow_from_directory('teste',target_size = (64, 64),
 batch_size = 32, 
 class_mode = 'categorical')

classificador.fit(amostras_treinamento,
steps_per_epoch = 4,
epochs = 7,
validation_data = amostras_teste,
validation_steps = 1000)
