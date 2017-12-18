K-means 50 clusters (given centroid)

Original (euclidean)

purity : 0.964
purity corr : 0.964
f1 : 0.964232516112
entropy : 0.147060146381

Original (cosine)

purity : 0.978
purity corr : 0.978
f1 : 0.977730655399
entropy : 0.0833692073919

Original (multiplication)

purity : 0.978
purity corr : 0.978
f1 : 0.977478130147
entropy : 0.0793688772336

Original (f1)

purity : 0.964
purity corr : 0.964
f1 : 0.963376173713
entropy : 0.134833049242

Summ100 (euclidean)

purity : 0.56
purity corr : 0.56
f1 : 0.492808524809
entropy : 1.65994825965

Summ100 (cosine)

purity : 0.985
purity corr : 0.985
f1 : 0.982545454545
entropy : 0.0344829847612

Summ100 (multiplication)

purity : 0.985
purity corr : 0.985
f1 : 0.984761904762
entropy : 0.0541446071166

Summ100 (f1)

purity : 0.88
purity corr : 0.88
f1 : 0.860519480519
entropy : 0.35612737131

Summ10 (euclidean)

purity : 0.1105
purity corr : 0.1105
f1 : 0.0734880325735
entropy : 4.92849825021

Summ10 (cosine)

purity : 0.9065
purity corr : 0.9065
f1 : 0.903781474497
entropy : 0.42595333366

Summ10 (multiplication)

purity : 0.871
purity corr : 0.871
f1 : 0.869433598676
entropy : 0.612386019069

Summ10 (f1)

purity : 0.6585
purity corr : 0.6585
f1 : 0.634326658522
entropy : 1.76004385039

# Mails
Lo que hay son unas pruebas con k-means de sklearn. Use para lematizar el stanford corenlp. Éste últimopPuedes instalarlo de aquí https://stanfordnlp.github.io/CoreNLP/ y correrlo como un servicio:

java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 -annotators lemma

Eso carga sólo el lematizador, que usa mi programa en tiempo real. En general, lo que hago es calcular los vectores para cada documento basado en sus terminos lematizados sin stopwords (baselines.py que usa duc.py y creo que processing.py). De ahí, calculé los centroides ideales (eso está en processing.py), y se los doy a sklearn para que ejecute k-means (eso está en kmeans.py) con una distancia coseno (use la de nltk). No está muy complicado, pero como no esperaba que nadie más lo leyera entonces no comenté nada. Echale un ojo y dime que dudas te surgen para que te explique lo que hay que hacer a continuación y si crees poder terminar antes de que me vaya, ¿qué opinas?

Ahora, lo importante primero, es que corras ese código. En la carpeta hay un requirements.txt para virtualenv. Crea un virtualenv, activalo e instala los requirements con pip (pip install -r requirements.txt). Cuando eso esté, ejecuta baselines.py (tiene que estar corriendo el servicio de corenlp en tu máquina)  y despues ejecuta __init__.py . Lo que tiene que darte es los numeros del readme y otros valores (estoy experimentando varias cosas).

Por cierto, es Python 3:

virtualenv --python=/usr/bin/python3 env

O la ruta equivalente si usas windows.

Los archivos principales entonces son baseline.py y __init__.py. Usalos como mapa para entender las funciones que llaman.
