
from sklearn import model_selection, svm

random_state = 42

class SVM_classifier:
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

    def fit(self,train,y):
        self.SVM.fit(train,y)
        return self.SVM

    def predict(self,x):
        return self.SVM.predict(x)# Use accuracy_score function to get the accuracy

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
class SVM_pipelined:
    #gs_clf_svm
    svm = Pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                alpha=1e-3, random_state=random_state)),
                            ])
    
    gs_clf_svm = None
    
    def fit(self,train,y):
        self.svm.fit(train, y)
        parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
                   'tfidf__use_idf': (True, False),
                   'clf-svm__alpha': (1e-2, 1e-3),
                 }
        self.gs_clf_svm = GridSearchCV(self.svm, parameters_svm, n_jobs=-1)
        self.gs_clf_svm = self.gs_clf_svm.fit(train, y)
        return self.gs_clf_svm

    def predict_svm(self,x):
        return self.svm.predict(x)

    def predict(self,x):
        return self.gs_clf_svm.predict(x)


