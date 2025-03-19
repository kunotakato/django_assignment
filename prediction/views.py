from django.shortcuts import render
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import save_model  

# VGG16モデルのロード
model = VGG16(weights='imagenet')
save_model(model, 'vgg16.h5')  

import numpy as np
from .forms import ImageUploadForm
from .models import UploadedImage



def predict_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            img_path = uploaded_image.image.path

            # 画像の前処理
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # 画像の判定
            predictions = model.predict(img_array)
            results = decode_predictions(predictions, top=5)[0]

            return render(request, 'prediction/result.html', {'results': results, 'image_url': uploaded_image.image.url})

    else:
        form = ImageUploadForm()

    return render(request, 'prediction/upload.html', {'form': form})