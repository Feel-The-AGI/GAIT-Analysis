import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

# Check GPU availability and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set the GPU to use
        tf.config.set_visible_devices(gpus[1], 'GPU')  # Replace with gpus[0] if you want to use the first GPU

        # Set memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, using CPU instead.")

# Parameters
batch_size = 256
learning_rate = 0.001
num_epochs = 10
image_size = (256, 256)
num_classes = 25  # Adjust this as needed

# Define the training and testing directories
train_dir = "C:/Users/gaitl/OneDrive/Desktop/gait/dataset/2D_Silhouettes/0000/training"
test_dir = "C:/Users/gaitl/OneDrive/Desktop/gait/dataset/2D_Silhouettes/0000/testing"

# Preprocessing function
def preprocess_image(image, label):
    image = tf.image.resize(image, image_size)  # Resize images
    image = tf.cast(image, tf.float32) / 255.0  # Rescale pixel values
    # Normalization
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image, label

# Load and preprocess the training and testing dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=123).map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=False).map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# CNN Model Definition
model = models.Sequential([
    layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training the model
history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)

# Testing the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Accuracy of the network on the test images: {test_acc * 100:.2f}%')
