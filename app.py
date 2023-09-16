# Sample app: capitalize text and return it



from flask import Flask, render_template, request, jsonify
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)

csrf = CSRFProtect(app)

@app.route('/process_text', methods=['POST'])
def process_text():
    input_text = request.json.get('input_text')
    capitalized_text = input_text.upper()
    return jsonify({'capitalized_text': capitalized_text})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)