#This file has functions that helps determine which words are standalone in a word without spaces and separates them

# Converts snake case to regular words (spaced)
def sprinkle_on_snake(word):
    return " ".join(words.split('_'))

#converts camel case to regular words (spaced)
import re
def sprinkle_on_camel(word):
    return " ".join(re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', word)).split())
