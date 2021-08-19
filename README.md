# Test technique Api Python/Flask 

# source fichier : https://www.kaggle.com/karanarya4196/twitter-sentiment-analysis?select=sentiment.tsv

# API serveur Rest(Python/Flask) 

# Installer Docker pour windows

# Construire l'image docker  :

docker build -t mynlpmodelflask:v1 .

# lancer l'image docker  :

docker run -d -p 5000:4000 --name nlpmodel mynlpmodelflask:v1

# pour le serveur web simple nous avons utilisé une image hub: frolvlad/alpine-python-machinelearning:latest
# en lançant la commande : docker pull frolvlad/alpine-python-machinelearning à partir de Dockerfile


# API initialization dans app.py
# Dans un terminale on peut installer les dependence :

pip install -r requirements.txt

# Lancer l'API :

flask run --port=4000


# pour améliorer l'API on peut rajouter une image mongoDB pour gérer une base de données des tweets
# On peut même ajouter des utilisateurs à la base de données et améliorer les performance du modèle
# On peut ajouter aussi un volume qui va permettre d'echanger des données entre conteneur 

