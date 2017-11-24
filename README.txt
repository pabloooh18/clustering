K-means 50 clusters (centroid seeded)

Complete documents
purity : 0.876
f1 : 0.900622195747
entropy : 0.583970469182

Sum100
purity : 0.66
f1 : 0.498078380723
entropy : 1.20295201143

Sum10
purity : 0.1695
f1 : 0.0812850411753
entropy : 4.30810132307

# Mails
Lo que hay son unas pruebas con k-means de sklearn. Use para lematizar el stanford corenlp. Éste últimopPuedes instalarlo de aquí https://stanfordnlp.github.io/CoreNLP/ y correrlo como un servicio:

java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 -annotators lemma

Eso carga sólo el lematizador, que usa mi programa en tiempo real. En general, lo que hago es calcular los vectores para cada documento basado en sus terminos lematizados sin stopwords (baselines.py que usa duc.py y creo que processing.py). De ahí, calculé los centroides ideales (eso está en processing.py), y se los doy a sklearn para que ejecute k-means (eso está en kmeans.py) con una distancia coseno (use la de nltk). No está muy complicado, pero como no esperaba que nadie más lo leyera entonces no comenté nada. Echale un ojo y dime que dudas te surgen para que te explique lo que hay que hacer a continuación y si crees poder terminar antes de que me vaya, ¿qué opinas?

Ahora, lo importante primero, es que corras ese código. En la carpeta hay un requirements.txt para virtualenv. Crea un virtualenv, activalo e instala los requirements con pip (pip install -r requirements.txt). Cuando eso esté, ejecuta baselines.py (tiene que estar corriendo el servicio de corenlp en tu máquina)  y despues ejecuta __init__.py . Lo que tiene que darte es los numeros del readme y otros valores (estoy experimentando varias cosas).

Por cierto, es Python 3:

virtualenv --python=/usr/bin/python3 env

O la ruta equivalente si usas windows.

Los archivos principales entonces son baseline.py y __init__.py. Usalos como mapa para entender las funciones que llaman.
