import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Check GPU availability and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set the GPU to use
        tf.config.set_visible_devices(gpus[0], 'GPU')  # Replace with gpus[0] if you want to use the first GPU

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
batch_size = 64
learning_rate = 0.001
num_epochs = 5
image_size = (256, 256)
num_classes = 3  # Adjust this as needed

# Define the base directory and training/testing directories based on base directory
base_dir = "C:/Users/gaitl/OneDrive/Desktop/gait/dataset/2D_Silhouettes/0000"
train_dir = os.path.join(base_dir, 'training')
test_dir = os.path.join(base_dir, 'testing')

# List of camera IDs for training and testing
train_camids = ['camid0_videoid2', 'camid3_videoid2', 'camid9_videoid2']
test_camids = ['camid11_videoid2']

# Ensure the directories for training and testing exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to copy the specified camid folders into the training/testing directories
def copy_folders(camid_list, destination):
    for camid in camid_list:
        src_path = os.path.join(base_dir, camid)
        dst_path = os.path.join(destination, camid)
        if os.path.exists(dst_path):
            print(f"Directory {dst_path} already exists. Skipping copy to prevent duplication.")
            continue
        try:
            shutil.copytree(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")
        except Exception as e:
            print(f"Failed to copy {src_path} to {dst_path}: {str(e)}")

# Copy the folders
copy_folders(train_camids, train_dir)
copy_folders(test_camids, test_dir)


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

# Adding Dropout in the CNN Model
model = models.Sequential([
    layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),  # Dropout layer after pooling
    layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),  # Another Dropout layer
    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dropout(0.5),  # Dropout before the final layer
    layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Callbacks
checkpoint_callback = ModelCheckpoint(filepath='best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

# Training the model with callbacks
history = model.fit(
    train_dataset, 
    epochs=num_epochs, 
    validation_data=test_dataset, 
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Testing the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Accuracy of the network on the test images: {test_acc * 100:.2f}%')
