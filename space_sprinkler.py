#This file has functions that helps determine which words are standalone in a word without spaces and separates them
# Each function on this file uses lower() and returns words in all non caps

#Converts unspaced words to spaced worts using Zipf's law,
## THIS PART IS TAKEN FROM https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words#11642687
from math import log


# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
words = open("words-by-frequency.txt").read().split()
wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out)).lower()

## UP PART IS TAKEN FROM https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words#11642687

from collections import Counter 
from heapq import merge

freq_list = []
counter = dict()

def merge_dict_by_adding(d1,d2):
    for k in d2:
        if k in d1:
            d1[k]+=d2[k]
        else:
            d1[k]=d2[k]
    return d1

def update_costs():
    global freq_list
    global wordcost
    global counter
    freq_list = [x[0] for x in sorted(counter.items(), key=lambda kv: kv[1], reverse=True)]
    for i,w in enumerate(freq_list):
        #TODO determine if we need to change it if exits
        wordcost[w]=log(i+3)*log(len(freq_list)+2)

def add_words(word_list):
    global counter
    merge_dict_by_adding(counter,Counter(word_list))
    update_costs()

def add_single_word(word):
    global counter
    merge_dict_by_adding(counter,{word:1})
    update_costs()

# Converts snake case to regular words (spaced)
def sprinkle_on_snake(word):
    return " ".join(word.split('_')).lower()

#converts camel case to regular words (spaced)
import re
def sprinkle_on_camel(word):
    return " ".join(re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', word)).split()).lower()
