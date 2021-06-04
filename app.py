from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('CovidDataset.pkl','rb'))
app = Flask(__name__ , static_url_path='/static')

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods = ['POST'])
def output():
    a = request.form['a']
    b = request.form['b']
    c = request.form['c']
    d = request.form['d']
    e = request.form['e']
    f = request.form['f']
    g = request.form['g']
    h = request.form['h']
    i = request.form['i']
    j = request.form['j']
    k = request.form['k']
    l = request.form['l']
    m = request.form['m']
    n = request.form['n']
    o = request.form['o']
    p = request.form['p']
    q = request.form['q']
    r = request.form['r']

    arr = np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r]])
    prediction = model.predict(arr)
    return render_template('output.html', data = prediction)

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/information')
def information():
	return render_template('information.html')
@app.route('/data')
def data():
	return render_template('data.html')

if __name__ == '__main__':
    app.run(debug=True)