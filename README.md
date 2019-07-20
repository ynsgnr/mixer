# Mixer
Mixed for reddit (In Progress)

## What Is This?
This is a repo of python code that aims to guess subreddit category using NLP and Machine Learning techniques by sorely by subreddit title. Ex: /r/gifs should categorized as gifs, and /r/cats should categorize as animals.

## Results (For Now)
- Naive Bayes Accuracy With TF-IDF Vectorizer Score ->  42.971887550200805
- SVM Accuracy With TF-IDF Vectorizer Score ->  46.98795180722892
- gs_clf With Naive Bayes Accuracy Score -> 45.149754135002235 {'clf__alpha': 0.01, 'tfidf__use_idf': False, 'vect__ngram_range': (1, 2)}
- Word2Vec with keywords and spacy en_core_web_md -> 09.236947791164658

### TO-DO
 - Add Stemming