# This is the main file to run to do everything

#Add links

#Three links with different subreddit links (biggest link i could found, if you can find a bigger list let me know)

#https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits - main thing
#https://www.reddit.com/r/ListOfSubreddits/wiki/nsfw - nsfw thing

#https://www.reddit.com/r/ListOfSubreddits/wiki/banned - can be used to train for banned reddits but later

links = ["https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits" , "https://www.reddit.com/r/ListOfSubreddits/wiki/nsfw"]

from list_of_subreddit_parser import *

subreddits = get_linked_subreddits_from_pages(links)
print(subreddits)