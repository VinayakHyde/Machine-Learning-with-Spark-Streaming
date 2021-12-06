import numpy as np
from pyspark.ml.feature import Tokenizer, StringIndexer, StopWordsRemover, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
from nltk.corpus import stopwords
from pyspark.sql.functions import lower, regexp_replace
from pyspark.ml import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import pickle
import matplotlib.pyplot as plt

filename1 = 'multinomialnb.sav'
filename2 = 'sgdclassifier.sav'
filename3 = 'passiveaggressive.sav'
filename4 = 'minibatchkmeans.sav'
clf1 = pickle.load(open(filename1, 'rb'))
clf2 = pickle.load(open(filename2, 'rb'))
clf3 = pickle.load(open(filename3, 'rb'))
clu1 = pickle.load(open(filename4, 'rb'))


def sender(df):
	global clf1
	global clf2
	global clf3
	global clu1


	df_clean = df.select("sentiment", (lower(regexp_replace("tweet", "[^a-zA-Z\s]", "")).alias("tweet")))
	tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
	remover = StopWordsRemover(inputCol="words", outputCol="nostop")
	#cv = CountVectorizer(inputCol="nostop", outputCol="vectors")
	label_stringIdx = StringIndexer(inputCol = "sentiment", outputCol = "label")
	pipeline = Pipeline(stages=[tokenizer, remover, label_stringIdx])

	pipelineFit = pipeline.fit(df_clean)
	df_clean = pipelineFit.transform(df_clean)

	X = df_clean.select("nostop").collect()
	    
	finalstlist = []
	    
	for i in range(len(X)):
		st=''
		listofwords = X[i]['nostop']
		st = ' '.join(listofwords)
		finalstlist.append(st)
	    	
	tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
	Xnew = tfidfconverter.fit_transform(finalstlist).toarray()
	
	
	
	Y = df.select('sentiment').rdd.flatMap(lambda x: x).collect()
	Ynew = [int(i) for i in Y]
	Ynew = np.array(Ynew)
	
	#MULTINOMIALNB
	
	y_pred = clf1.predict(Xnew)
	print(accuracy_score(Ynew, y_pred))

	#SGDCLASSIFIER
	
	y_pred = clf2.predict(Xnew)
	print(accuracy_score(Ynew, y_pred))
	
	#BERNOULLINB
	
	y_pred = clf3.predict(Xnew)
	print(accuracy_score(Ynew, y_pred))
	
	#MINIMIZEDKMEANS
	"""pca = PCA(2)
	df = pca.fit_transform(Xnew)
	label = clu1.predict(df)
	print(label)

	uniqueLabels = np.unique(label)
	
	for j in uniqueLabels:
		plt.scatter(df[label == j, 0], df[label == j, 1], label = j)
		plt.legend()
		
	plt.show() """
	
