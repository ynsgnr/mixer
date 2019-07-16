#This file parses list of subreddits to dataset of words

subreddit_sep = "/r/"

#Return subreddit name from the link
def get_subreddit_from_link(link):
    parts = link.split(subreddit_sep)
    if len(parts) !=2:
        return ""
    parts = parts[1].split("/")
    return parts[0]

#Get all the links from given links
import httplib2
from bs4 import BeautifulSoup, SoupStrainer

#this function gets all linked(<a>) subreddits from given array of page links
def get_linked_subreddits_from_pages(links):
    subreddits = []
    for page in links:
        http = httplib2.Http()
        status, response = http.request(page)
        for link in BeautifulSoup(response, parse_only=SoupStrainer("a")):
            if link.has_attr("href") and subreddit_sep in link["href"]:
                subreddits.append(get_subreddit_from_link(link["href"]))
    return subreddits
