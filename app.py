from flask import Flask, request, render_template
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
modelcv=pickle.load(open('modelcv.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template('fake.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    text=[]
    for x in request.form.values():
        text.append(x)
    line =' '.join(text)
    token = word_tokenize(line.lower())
    tokens = []
    stopwordsList = stopwords.words("english")
    stopwordsList.extend([',','.','-','!'])
    for word in token:
        if word not in stopwordsList:
            tokens.append(word)  
    wnet=WordNetLemmatizer()
    for i in range(len(tokens)):
        tokens[i] = wnet.lemmatize(tokens[i],pos='v')
    sent = ' '.join(tokens)
    vect = modelcv.transform([sent])
    prediction=model.predict_proba(vect)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.65):
        return render_template('fake.html',pred='News is Authentic.')
    else:
        return render_template('fake.html',pred='News is Spurious.')


if __name__ == '__main__':
    app.run(debug=True)