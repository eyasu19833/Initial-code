import csv
import datetime
import os
import numpy as np
import re
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import nltk
from nltk.corpus import stopwords


#Open a file and remove all quote and save it in different filename called 'output.txt'
with open("C:/Users/x190170/Documents/Data/Capstone/Dataset/bbchealth.txt", "r") as f, open('output.txt', 'w') as fo:
    for line in f:
        fo.write(line.replace('"', '').replace("'", ""))


	
#Open file from the directory in read mode
with open("C:/Users/x190170/Documents/Data/Capstone/Code/output.txt", "r") as ins:
    array = []
    for line in ins:
		 #split the line and take the last column, which is the tweet text and remove the URL from it, and append to "array"
         array.append(re.sub(r"http\S+", "", line).split("|")[-1:])

kmeans = KMeans(n_clusters=2)	
#stopwords = nltk.corpus.stopwords.words('english')


#vectorize the text i.e. convert the strings to numeric features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(array)
#cluster documents
true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
#print top terms per cluster clusters
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :10]:
        print ' %s' % terms[ind],
    print
print(array)