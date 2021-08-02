# Test technique Api Python/Flask 

# source fichier : https://www.kaggle.com/karanarya4196/twitter-sentiment-analysis?select=sentiment.tsv

# API serveur Rest(Python/Flask) 

# Installer Docker pour windows

# Construire l'image docker  :

docker build -t mynlpmodelflask:v1 .

# Construire l'image docker  :

docker run -d -p 5000:4000 --name nlpmodel mynlpmodelflask:v1


# API initialization dans app.py
# Dans un terminale on peut installer les dependence :

pip install -r requirements.txt

# Lancer l'API :

flask run --port=5000


# pour améliorer l'API on peut rajouter une image mongoDB pour gérer une base de données des tweets
# On peut même ajouter des utilisateurs à la base de données
# On peut ajouter aussi un volume qui va permettre d'echanger des données entre conteneur  

