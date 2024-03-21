from flask import Flask, render_template, request, send_file
from SSAF import generate_inf_graph

app = Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('upload.html', graph_path=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', graph_path=None, error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', graph_path=None, error="No selected file")
        if file:
            filename = 'uploads/' + file.filename
            file.save(filename)
            N = int(request.form['N'])
            L = int(request.form['L'])
            M = int(request.form['M'])
            r = int(request.form['r'])
            graph_path = generate_inf_graph(filename, r, M, N, L)
            return render_template('upload.html', graph_path=graph_path)

@app.route('/plot/<path:path>')
def serve_plot(path):
    return send_file(path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
