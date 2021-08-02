from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


## Definitions des fonction pour le traitement du texte
#Grace à cette fonction on pourra enlever les @devant chaque commentaire
def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt
# cette fonction permet de compter le nombre de ponctuation sur chaque ligne en %
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100


app = Flask(__name__)

# le fichier sentiments.csv a deux colonnes label', 'body_text'
# on cherche à faire de la prédiction pour savoir si un sentiment exprimer sur tweeter est positive ou negative
#Apprentissage supervisé : pour cet apprentissage, nous avons des données en entrée (Features 'body_text') et le résultat attendu (Label)
data = pd.read_csv("sentiment.tsv",sep = '\t')
data.columns = ["label","body_text"]
# Feature et Labels
data['label'] = data['label'].map({'pos': 0, 'neg': 1})
# avec la foncition remove_pattern() on vas supprimer les ponctuation sur chaque ligne
data['tidy_tweet'] = np.vectorize(remove_pattern)(data['body_text'],"@[\w]*")
tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split())
stemmer = PorterStemmer()
# Tokeniser les tweets
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
data['tidy_tweet'] = tokenized_tweet
data['body_len'] = data['body_text'].apply(lambda x:len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))
X = data['tidy_tweet']
y = data['label']
# Extraire données en entrée avec la fonction CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X.toarray())],axis = 1)
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
## Utilisant un modele de claissification
clf = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
clf.fit(X,y)

# route qui permet de voir la page d'accueil où se trouve le formulaire
@app.route('/')
def home():
    return render_template('home.html')

# route qui permet de poster l'information saisie, on analyse l'information et on fait une prédiction 
# pour savoir si l'information saisie est un sentiment positive ou négative
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = pd.DataFrame(cv.transform(data).toarray())
        body_len = pd.DataFrame([len(data) - data.count(" ")])
        punct = pd.DataFrame([count_punct(data)])
        total_data = pd.concat([body_len,punct,vect],axis = 1)
        my_prediction = clf.predict(total_data)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4000)
