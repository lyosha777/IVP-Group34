from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR/"data"/"test"
TEST_CSV = BASE_DIR/"data"/"test.csv"
MODEL_PATH = BASE_DIR/"models"/"hindi_digit_cnn.keras"
OUTPUT_CSV = BASE_DIR/"predictions.csv"

IMG_SIZE = (32,32)
model = tf.keras.models.load_model(MODEL_PATH)
test_df = pd.read_csv(TEST_CSV)
ID_COL = "Id"
images = []
ids = []

for image_id in test_df[ID_COL]:
    img_path = TEST_DIR/f"{image_id}.png"
    if not img_path.exists():
        print("not found:",img_path)
        continue
    img = Image.open(img_path).convert("L")
    img = img.resize(IMG_SIZE)
    img = np.array(img,dtype=np.float32)
    img = np.expand_dims(img,axis=-1)
    images.append(img)
    ids.append(image_id)

images = np.array(images,dtype=np.float32)
predictions = model.predict(images)
predicted_digits = np.argmax(predictions,axis=1)

output_df = pd.DataFrame({
    "Id": ids,
    "Category": predicted_digits
})
output_df.to_csv(OUTPUT_CSV, index=False)
print("dr:", OUTPUT_CSV)