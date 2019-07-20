# This is the main file to run to do everything

#Add links

#Three links with different subreddit links (biggest link i could found, if you can find a bigger list let me know)

#https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits - main thing
#https://www.reddit.com/r/ListOfSubreddits/wiki/nsfw - nsfw thing

#https://www.reddit.com/r/ListOfSubreddits/wiki/banned - can be used to train for banned reddits but later

random_state = 215

from list_of_subreddit_parser import *
from space_sprinkler import *

def get_spaced_subreddits():
    links = ["https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits" , "https://www.reddit.com/r/ListOfSubreddits/wiki/nsfw"]

    subreddits, categories = get_linked_subreddits_from_pages_faster(links)

    sprinkled_subs = sprinkle_on_subreddits(subreddits)

    return sprinkled_subs, subreddits, categories

def get_categorized(categories):
    categories_dict = {}
    categories_dict_subs = {}
    for i,category in enumerate(categories):
        if not category[-1] in categories_dict:
            categories_dict[category[-1]]=1
            categories_dict_subs[category[-1]]=[i]
        else:
            categories_dict[category[-1]]+=1
            categories_dict_subs[category[-1]].append(i)
    return categories_dict, categories_dict_subs

def tokenize_categories(categories):
    #Tokenize categories
    categories_token = {}
    token_categories = {}
    i = 0
    for key in categories_dict:
        categories_token[key]=i
        token_categories[i]=key
        i+=1

    tokenized_categories = [0]*len(categories)
    for i, cat in enumerate(categories):
        tokenized_categories[i]=categories_token[cat[-1]]
    return tokenized_categories, token_categories

sprinkled_subs, subreddits, categories = get_spaced_subreddits()

#remove Ex50k+ category
cats_to_remove = ["Ex 50k+"]
filtered = (
    (cat, sub, sp_sub) 
        for cat, sub, sp_sub in zip(categories, subreddits, sprinkled_subs) 
            if not cat[-1] in cats_to_remove)

c, s, ss = zip(*filtered)

categories = list(c)
subreddits = list(s)
sprinkled_subs = list(ss)

categories_dict, categories_dict_subs = get_categorized(categories)

'''
#get detailed categories for other
for subreddit_i in categories_dict_subs['Other']:
    categories[subreddit_i]=categories[subreddit_i][0:-1]
categories_dict, categories_dict_subs = get_categorized(categories)
'''

# build all words dictonary for dataset
all_words = {}
for subreddit in sprinkled_subs:
    for word in subreddit.split(" "):
        if not word in all_words:
            all_words[word]=1
        else:
            all_words[word]+=1

# change it to array of arrays for word2vec
sprinkled_subs_array = sprinkled_subs.copy()
for i,sub in enumerate(sprinkled_subs_array):
    sprinkled_subs_array[i]=sub.split(" ")
            
tokenized_categories, token_categories = tokenize_categories(categories)

from sklearn.model_selection import train_test_split

train, test, Train_Y, Test_Y = train_test_split(sprinkled_subs, tokenized_categories, test_size=0.1, random_state=random_state, stratify=tokenized_categories)


#TF-IDF Scheme

from sklearn.feature_extraction.text import TfidfVectorizer

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(sprinkled_subs)
Train_X_Tfidf = Tfidf_vect.transform(train)
Test_X_Tfidf = Tfidf_vect.transform(test)

from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

#Word2Vec
from gensim.models import Word2Vec

model = Word2Vec(sprinkled_subs_array, min_count=1)
model.train(sprinkled_subs_array,epochs = 10,total_examples=len(sprinkled_subs_array))

#  https://engineering.reviewtrackers.com/using-word2vec-to-classify-review-keywords-a5fa50ce05dc
import spacy
import en_core_web_md

nlp = en_core_web_md.load()
#Split %50 and use everyword as keyword
train, test, Train_Y, Test_Y = train_test_split(sprinkled_subs, [c[-1] for c in categories], test_size=0.1, random_state=random_state, stratify=[c[-1] for c in categories])

import itertools
import numpy as np

#Train?
topic_docs = list(nlp.pipe(train,
  batch_size=10000,
  n_threads=3))

topic_vectors = np.array([doc.vector 
  if doc.has_vector else spacy.vocab[0].vector
  for doc in topic_docs])

#Test?
keyword_docs = list(nlp.pipe(test,
  batch_size=10000,
  n_threads=3))

keyword_vectors = np.array([doc.vector
  if doc.has_vector else spacy.vocab[0].vector
  for doc in keyword_docs])


from sklearn.metrics.pairwise import cosine_similarity
# use numpy and scikit-learn vectorized implementations for performance
simple_sim = cosine_similarity(keyword_vectors, topic_vectors)
topic_idx = simple_sim.argmax(axis=1)

print("Guesses:")
correct = 0
for i,idx in enumerate(topic_idx):
    #print(subreddits[i]+" - "+categories[idx][-1])
    if  categories[idx][-1]==Test_Y[i]:
        correct+=1
print("Accuracy on test:")
print(correct/len(test))