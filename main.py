# This is the main file to run to do everything

#Add links

#Three links with different subreddit links (biggest link i could found, if you can find a bigger list let me know)

#https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits - main thing
#https://www.reddit.com/r/ListOfSubreddits/wiki/nsfw - nsfw thing

#https://www.reddit.com/r/ListOfSubreddits/wiki/banned - can be used to train for banned reddits but later

random_state = 215

from preprocessing.data_processing import *
from preprocessing.tokenizer import *
from models.word2vec import *
from models.nb_classifier import *
from models.svm_classifier import *

links = ["https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits" , "https://www.reddit.com/r/ListOfSubreddits/wiki/nsfw"]
data_path = "sprinkled_subs"
stemmed_data_path = "sprinkled_subs_stemmed"

sprinkled_subs,categories = get_data(stemmed_data_path,data_path,links)
categories_dict, categories_dict_subs = get_categorized(categories)
tokenized_categories, token_categories = tokenize_categories(categories)

from sklearn.model_selection import train_test_split

train, test, Train_Y, Test_Y = train_test_split(sprinkled_subs, tokenized_categories, test_size=0.1, random_state=random_state, stratify=tokenized_categories)

tfidf = get_tfdif(sprinkled_subs)
train_tfidf = tfidf.transform(train)
test_tfidf = tfidf.transform(test)

from sklearn.metrics import accuracy_score
import numpy as np

nb = NB_classifier()
nb.fit(train_tfidf,Train_Y)
predictions_NB = nb.predict(test_tfidf)
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

svm = SVM_classifier()
svm.fit(train_tfidf,Train_Y)
predictions_SVM = svm.predict(test_tfidf)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

nbp = NB_pipelined()
gs_clf_nb = nbp.fit(train,Train_Y)
predictions_NB = nbp.predict(test)

print("gs_clf_nb scores:")
print(np.mean(predictions_NB == Test_Y))
print(gs_clf_nb.best_score_)
print(gs_clf_nb.best_params_)

svmp = SVM_pipelined()
gs_clf_svm = svmp.fit(train,Train_Y)
predicted_svm = svmp.predict_svm(test)
predicted_svm2 = svmp.predict(test)

print("gs_clf_svm scores:")
print(np.mean(predicted_svm == Test_Y))
print(np.mean(predicted_svm2 == Test_Y))
print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)