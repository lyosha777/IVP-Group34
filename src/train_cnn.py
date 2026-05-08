# imports
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

# paths
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR/"data"/"train"
MODEL_DIR = BASE_DIR/"models"
MODEL_DIR.mkdir(exist_ok=True)

# training parameters
IMG_SIZE = (32, 32) #resize images to 32x32
BATCH_SIZE = 32 #images per batch
EPOCHS = 15 #training epochs

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred", #names are used as class labels
    label_mode="categorical",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,  #reserves 20% for validation
    subset="training",
    seed=42
)

# validation parameters
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)
#cache() stores data.
#shuffle() randomizes order.
#prefetch() prepares the next batch while the model is training on the current batch.
train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

#building the sequential CNN model
#convolutional layers extract visual features from images.
#maxPooling layers reduce size and computation.
#dense layers do the final classification.
model = models.Sequential([
    layers.Input(shape=(32, 32, 1)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(10, activation="softmax")
])
# compiling
model.compile(
    optimizer="adam", #adaptive optimizer
    loss="categorical_crossentropy", #multi-class classification
    metrics=["accuracy"] #evaluation metric
)

model.summary()
# early stopping to monitor validation loss
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
# model learns from train_ds and is evaluated on val_ds after each epoch.
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

model.save(MODEL_DIR / "hindi_digit_cnn.keras") #saving the model in Keras format

print("Model saved to:", MODEL_DIR / "hindi_digit_cnn.keras")