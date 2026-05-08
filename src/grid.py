from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "data" / "train"

fig, axes = plt.subplots(2, 5, figsize=(7, 3))

for digit, ax in enumerate(axes.flat):
    class_dir = TRAIN_DIR / str(digit)

    image_files = list(class_dir.glob("*"))

    img_path = random.choice(image_files)

    img = Image.open(img_path).convert("L")

    ax.imshow(img, cmap="gray")
    ax.set_title(str(digit))
    ax.axis("off")

plt.tight_layout()
plt.savefig("sample_digits.png", dpi=300, bbox_inches="tight")

print("Saved sample_digits.png")