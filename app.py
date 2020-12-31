import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model_heart.pkl','rb'))
@app.route('/')
def home():
    return render_template('Heart_Disease_Classifier.html')

@app.route('/predict',methods=['POST'])
def predict():
 features = [float(i) for i in request.form.values()]
 ar=np.array([features])
 st = StandardScaler()
 f= st.fit_transform(ar)
 predictions = model.predict(f)
 if predictions[0] == 1:
  return render_template('Heart_Disease_Classifier.html', result = 'The persion is not likely to have heart disease')
 else:
  return render_template('Heart_Disease_Classifier.html', result='The persion is not likely to have heart disease')
if __name__ == '__main__':
  app.debug = True
  app.run()
