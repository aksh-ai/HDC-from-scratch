import os
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras import Sequential
from flask import Flask, flash, request, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = '12345678'

# model = pickle.load(open('models/MNIST.dat', 'rb'))

def ANN():
    model = Sequential()
    model.add(Dense(128, input_shape=(784, ), activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(256, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = ANN()
model.load_weights('models/MNIST_tf_weights.h5')

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_copy = img.copy()
    img = cv2.resize(img, (28, 28))
    img = img/255.0
    img = img.reshape(-1, 784)
    return img, cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)

@app.route('/empty')
def empty():
    filename = session.get('filename', None)
    os.remove(os.path.join(UPLOAD_FOLDER, filename))
    return redirect(url_for('index'))

@app.route('/results')
def results():
    pred = session.get('label', None)
    filename = session.get('filename', None)
    return render_template('results.html', pred=pred, f_name=filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            f = request.files['MNIST']
            filename = str(f.filename)

            if filename is not None:
                ext = filename.split('.')
                if ext[1] in ALLOWED_EXTENSIONS:
                    filename = secure_filename(f.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    f.save(file_path)

                    image, original = preprocess_image(file_path)

                    prediction = model.predict(image)[0].argmax()

                    print(f"Predicted Digit: {prediction}")

                    session['label'] = str(prediction)
                    session['filename'] = filename

                    return redirect(url_for('results'))

    except:
        pass

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)                    