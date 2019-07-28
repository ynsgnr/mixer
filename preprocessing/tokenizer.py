
try:
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize
except:
    import nltk
    nltk.download()
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize


def tokenize_categories(categories,categories_dict=None):
    #Tokenize categories
    from preprocessing.data_processing import get_categorized
    if categories_dict is None:
        categories_dict , _ = get_categorized(categories)
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


def stem_subs(sprinkled_subs):
    ps = PorterStemmer()
    stemmed_subs = []
    for sub in sprinkled_subs:
        stemmed_sub = ""
        for word in word_tokenize(sub):
            stemmed_sub+=" "+ps.stem(word)
        stemmed_subs.append(stemmed_sub[1:])
    return stemmed_subs