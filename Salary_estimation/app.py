from flask import  Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)

model = pickle.load(open('sal.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('sal.html')
@app.route('/predict',methods=['POST'])
def sal():
     data1 = request.form['year'] or 1
     arr = np.array([[data1]])
     pre=model.predict(arr)
     import math 
     for i in range(0,len(pre)):

          pre[i]=math.floor(pre[i])
     return render_template('after.html',d=pre)  

if __name__=="__main__":
    app.run(port=5000,debug=True)


