import numpy as np
from pyspark.ml.feature import Tokenizer, StringIndexer, StopWordsRemover, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
from nltk.corpus import stopwords
from pyspark.sql.functions import lower, regexp_replace
from pyspark.ml import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans


clf1 = MultinomialNB()
clf2 = SGDClassifier()
clf3 = PassiveAggressiveClassifier()
clu1 = MiniBatchKMeans(n_clusters=2)



def sender(df):

	
	global clf1
	global clf2
	global clf3
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
	filename1 = 'multinomialnb.sav'
	clf1.partial_fit(Xnew, Ynew, [0, 4])
	#y_pred = clf1.predict(x_test)
	#print("MNB: ", accuracy_score(y_test, y_pred))
	pickle.dump(clf1, open(filename1, 'wb'))
	
	
	#SGDClassifier
	filename2 = 'sgdclassifier.sav'
	clf2.partial_fit(Xnew, Ynew, [0, 4])
	#y_pred = clf2.predict(x_test)
	#print("SGDC: ", f1_score(y_test, y_pred, pos_label=0))
	pickle.dump(clf2, open(filename2, 'wb'))
	
	
	#PassiveAggressive
	filename3 = 'passiveaggressive.sav'
	clf3.partial_fit(Xnew, Ynew, [0, 4])
	#y_pred = clf3.predict(x_test)
	#print("BNB: ", accuracy_score(y_test, y_pred))
	pickle.dump(clf3, open(filename3, 'wb'))
	
	#MiniBatchKMeans
	pca = PCA(2)
	df = pca.fit_transform(Xnew)
	clu1.partial_fit(df)
	filename4 = 'minibatchkmeans.sav'
	#clu1.partial_fit(Xnew)
	pickle.dump(clu1, open(filename4, 'wb'))

