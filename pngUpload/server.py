import os
from flask import Flask, flash, render_template, redirect, request, url_for, send_file
import random
from werkzeug.utils import secure_filename

app = Flask(__name__)
SECRET_KEY="supertopsecretprivatekey"
UPLOAD_FOLDER="/Users/nimesh/PycharmProjects/webApp/pngUpload/tmp/storedImgs"
LETTER_SET = list(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

def generate_random_name(filename):
    """ Generate a random name for an uploaded file. """
    ext = filename.split('.')[-1]
    rns = [random.randint(0, len(LETTER_SET) - 1) for _ in range(3)]
    chars = ''.join([LETTER_SET[rn] for rn in rns])

    new_name = "{new_fn}.{ext}".format(new_fn=chars, ext=ext)
    new_name = secure_filename(new_name)

    return new_name



@app.route('/images/<filename>', methods=['GET'])
def images(filename):
    return send_file(os.path.join(app.config['/Users/nimesh/PycharmProjects/webApp/pngUpload/tmp/storedImgs'], filename))


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('home.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        image_file = request.files['image']

        # if filename is empty, then assume no upload
        if image_file.filename == '':
            flash('No file was uploaded.')
            return redirect(request.url)

        # if the file is "legit"
        if image_file:
            passed = False
            try:
                filename = generate_random_name(image_file.filename)
                filepath = os.path.join('/Users/nimesh/PycharmProjects/webApp/pngUpload/tmp/storedImgs', filename)
                image_file.save(filepath)
                passed = True
            except Exception:
                passed = False

            if passed:
                return redirect(url_for('predict', filename=filename))
            else:
                flash('An error occurred, try again.')
                return redirect(request.url)


@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    image_url = url_for('images', filename=filename)
    return render_template(
        'predict.html',
        image_url=image_url
    )


@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500

if __name__ == "__main__":
    app.run('127.0.0.1')