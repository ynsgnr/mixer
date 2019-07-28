
from sklearn.feature_extraction.text import TfidfVectorizer

def get_word_dictionary(sprinkled_subs):
    # build all words dictonary for dataset
    all_words = {}
    for subreddit in sprinkled_subs:
        for word in subreddit.split(" "):
            if not word in all_words:
                all_words[word]=1
            else:
                all_words[word]+=1

def get_tfdif(sprinkled_subs):
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(sprinkled_subs)
    return Tfidf_vect

