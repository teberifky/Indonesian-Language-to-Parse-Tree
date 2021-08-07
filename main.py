from flask import Flask, render_template, request
from program import *

app = Flask(__name__)


@app.route('/')
def index():
    input_data = request.args.get('input', "")
    hasil = "Enter the sentence above!"
    if input_data:
        parsed = parsing(input_data)
        if parsed:
            hasil = parsed
        else:
            hasil = "Salah satu atau seluruh kata dalam kalimat tidak terdapat pada grammars."

    return render_template('index.html', hasil=hasil, input_data=input_data)


app.run(host="0.0.0.0", port=5000, debug=True)