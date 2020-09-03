#from flask import Flask, request, jsonify
from flask import Flask, render_template, url_for, request
#from urllib import request
from werkzeug.utils import redirect, secure_filename
import os
from model import Model
#import model

app = Flask(__name__)
model = Model()
model.train_classifiers()
model.train_hierarchical()

app.config['UPLOAD_FOLDER'] = "audio/"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
def index():
    return render_template('base.html', isComplete=False)


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if request.files:
            f = request.files['audio']
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
            fileName=os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
            predict, stat, stat20 = model.newAudio(fileName)
            return render_template('base.html', audio=predict, isComplete=True, stat=stat, stat20=stat20)
    return render_template('base.html',isComplete=False)


@app.route("/about")
def about():
    return render_template('about.html', title='About')


if __name__ == "__main__":
    app.run(debug=True)

