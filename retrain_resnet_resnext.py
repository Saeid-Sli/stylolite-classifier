import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from google.colab import drive

# Mount Google Drive
drive.mount('/content/Payan/DataAug')

# Constants
epochs = 500
num_classes = 5
batch_size = 32
learning_rate = 0.001
initial_learning_rate = 0.001

# Set random seed for reproducibility
tf.random.set_seed(42)

# Data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Path to the split data in Google Drive
base_path = '/content/Payan/DataSplit'

# Training data
train_generator = datagen.flow_from_directory(
    base_path + '/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Validation (test) data
test_generator = datagen.flow_from_directory(
    base_path + '/validation',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# base_model = tf.keras.applications.ResNeXt50(
#     include_top=False,
#     weights='imagenet',
#     input_shape=(224, 224, 3)
# )

    
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),  # Adding dropout to prevent overfitting
    layers.Dense(num_classes, activation='softmax')
])

#  
def lr_scheduler(epoch, lr):
    if epoch % 100 == 0 and epoch > 0:
        return lr * 0.1  # Reduce learning rate by a factor of 0.1 every 100 epochs
    return lr

optimizer = SGD(learning_rate=initial_learning_rate, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

lr_schedule = LearningRateScheduler(lr_scheduler, verbose=1)

history_fine_tune = model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[lr_schedule])


base_model.set_weights(model.layers[0].get_weights())


# Plot
plt.plot(history_fine_tune.history['accuracy'], label='Train Acc (Fine-tune)')

plt.plot(history_fine_tune.history['val_accuracy'], label='Val Acc (Fine-tune)')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history_fine_tune.history['loss'], label='Train Loss (Fine-tune)')
plt.plot(history_fine_tune.history['val_loss'], label='Val Loss (Fine-tune)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


test_loss_fine_tune, test_accuracy_fine_tune = model.evaluate(test_generator)
print(f'Test Loss (Fine-tune): {test_loss_fine_tune:.4f}, Test Accuracy (Fine-tune): {test_accuracy_fine_tune:.4f}')

# Confusion
test_preds_fine_tune = model.predict(test_generator)
test_preds_fine_tune = np.argmax(test_preds_fine_tune, axis=1)
test_labels = test_generator.classes

conf_matrix_fine_tune = confusion_matrix(test_labels, test_preds_fine_tune)
class_names = ['intact Rock', 'stylolite', 'horizontal plug', 'vertical plug' , 'Crack']
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix_fine_tune, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label (Fine-tune)')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Test Set - Fine-tune)')
plt.show()

#     
"""
# Option 1: Fine-tune more layers
for layer in base_model.layers[:100]:
    layer.trainable = False
"""

"""
# Option 2: Use a different optimizer (e.g., Adam)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
"""

"""
# Option 3: Fine-tune with different data augmentation settings
datagen_augmented = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator_augmented = datagen_augmented.flow_from_directory(
    base_path + '/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

history_fine_tune_augmented = model.fit(train_generator_augmented, epochs=epochs, validation_data=test_generator, callbacks=[lr_schedule])
"""

"""
# Option 4: Add more dense layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),  # Additional dense layer
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])
"""




