# This is the main file to run to do everything

#Add links

#Three links with different subreddit links (biggest link i could found, if you can find a bigger list let me know)

#https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits - main thing
#https://www.reddit.com/r/ListOfSubreddits/wiki/nsfw - nsfw thing

#https://www.reddit.com/r/ListOfSubreddits/wiki/banned - can be used to train for banned reddits but later

from list_of_subreddit_parser import *
from space_sprinkler import *

def get_spaced_subreddits():
    links = ["https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits" , "https://www.reddit.com/r/ListOfSubreddits/wiki/nsfw"]

    subreddits, categories = get_linked_subreddits_from_pages_faster(links)

    sprinkled_subs = sprinkle_on_subreddits(subreddits)

    return sprinkled_subs, subreddits, categories


sprinkled_subs, subreddits, categories = get_spaced_subreddits()

categories_dict = {}
categories_dict_subs = {}
for i,category in enumerate(categories):
    if not category[-1] in categories_dict:
        categories_dict[category[-1]]=1
        categories_dict_subs[category[-1]]=[subreddits[i]]
    else:
        categories_dict[category[-1]]+=1
        categories_dict_subs[category[-1]].append(subreddits[i])