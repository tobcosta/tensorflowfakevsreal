from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import urllib.request

MODEL_URL = "https://huggingface.co/tb-costa23/keras-image-model-baseline/resolve/main/image_detection_v1.keras"
MODEL_PATH = "image_detection_v1.keras"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = load_model(MODEL_PATH)

img_folder = "test_images"
img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg'))]

for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)

img = image.load_img(img_path, target_size=(160, 160))  
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)  

predictions = model.predict(x)
predicted_class = np.argmax(predictions, axis=1)

print(f"Image: {img_file}")
print("Prediction vector:", predictions)
print("Predicted class index:", predicted_class)
print("-" * 40)