#This file parses list of subreddits to dataset of words
# Link: https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits#wiki_general_content

#Three links with different subreddit links (biggest link i could found, if you can find a bigger list let me know)

#https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits - main thing
#https://www.reddit.com/r/ListOfSubreddits/wiki/nsfw - nsfw thing

#https://www.reddit.com/r/ListOfSubreddits/wiki/banned - can be used to train for banned reddits but later



#Return subreddit name from the link
def get_subreddit_from_link(link):
    parts = link.split("reddit.com/r/")
    if len(parts) !=2:
        return ""
    parts = parts[1].split("/")
    return parts[0]