from sklearn import model_selection, naive_bayes

class NB_classifier:

    nb = naive_bayes.MultinomialNB()

    def fit(self,train_tfidf,y):
        # fit the training dataset on the NB classifier
        self.nb.fit(train_tfidf,y)# predict the labels on validation dataset

    def predict(self,tfidf):
        return self.nb.predict(tfidf)# Use accuracy_score function to get the accuracy

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle
# gs_clf_bayes
class NB_pipelined:

    p = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf', naive_bayes.MultinomialNB()),
                        ])

    gs_clf = None

    def fit(self,train,y):
        text_clf = self.p.fit(train, y)
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                        'tfidf__use_idf': (True, False),
                        'clf__alpha': (1e-2, 1e-3),
                    }
        self.gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
        self.gs_clf = self.gs_clf.fit(train, y)
        return self.gs_clf
    
    def predict(self,x):
        return self.gs_clf.predict(x)

    def save(self,p):
        if not '.pk1' in p:
            path = p+'.pk1'
        else:
            path = p
        with open(path, 'wb') as output:
            pickle.dump(self.gs_clf, output, pickle.HIGHEST_PROTOCOL)
    
    def load(self,p):
        if not '.pk1' in p:
            path = p+'.pk1'
        else:
            path = p
        with open(path, 'rb') as input:
            self.gs_clf = pickle.load(input)