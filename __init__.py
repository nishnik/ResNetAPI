from flask import Flask, render_template, request
from werkzeug import secure_filename
from keras.preprocessing import image as kimage
from keras.applications.resnet50 import preprocess_input, ResNet50
import numpy as np

app = Flask(__name__)

model = ResNet50()

@app.route('/')
def index():
   return render_template('index.html')
    
@app.route('/uploader', methods = ['POST'])
def uploader():
   if request.method == 'POST':
      f = request.files['file']
      f.save('images/' + secure_filename(f.filename))
      img = kimage.load_img('images/' + f.filename, target_size=(224, 224))
      img_loaded = kimage.img_to_array(img)
      img_loaded = np.expand_dims(img_loaded, axis=0)
      img_loaded = preprocess_input(img_loaded)
      return str(model.predict(img_loaded))
        
if __name__ == '__main__':
   app.run(debug = True)