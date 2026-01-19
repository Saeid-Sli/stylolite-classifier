import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


epochs = 500

#    
from google.colab import drive
drive.mount('/content/Payan/DataAug')

#   
num_classes = 5
batch_size = 32
learning_rate = 0.0001

tf.random.set_seed(42)

# DATAGEN
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

#        
base_path = '/content/Payan/DataSplit'

#   
train_generator = datagen.flow_from_directory(
    base_path + '/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)
#  
test_generator = datagen.flow_from_directory(
    base_path + '/validation',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the model (using pre-trained ResNext-50)
base_model = tf.keras.applications.ResNeXt50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
#         
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Traim
history = model.fit(train_generator, epochs=500, validation_data=test_generator)

#  
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#                  
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
"""
           ******
"""
# Confusion Matrix for the validation set
test_preds = model.predict(test_generator)
test_preds = np.argmax(test_preds, axis=1)
test_labels = test_generator.classes

conf_matrix = confusion_matrix(test_labels, test_preds)
class_names = ['intact Rock', 'stylolite', 'horizontal plug', 'vertical plug', 'Crack']
plt.figure(figsize=(8, 8))
"""      """
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Test Set)')
plt.show()


