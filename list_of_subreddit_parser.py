#This file parses list of subreddits to dataset of words

subreddit_sep = "/r/"
user_link = "/user/"
message_link = "/message/"

#Return subreddit name from the link
def get_subreddit_from_link(link):
    if user_link in link or message_link in link: return ""
    parts = link.split(subreddit_sep)
    if len(parts) !=2:
        return ""
    parts = parts[1].split("/")
    return parts[0]

#Get all the links from given links
import httplib2
from bs4 import BeautifulSoup, SoupStrainer

def get_subreddit_categories(subreddit,response):
    if not subreddit_sep in subreddit: subreddit = subreddit_sep + subreddit
    bs_element = BeautifulSoup(response)
    categories = []
    for a in bs_element.find_all("a", text=subreddit):
        categories.append(a.parent.previous_element.previous_element)
    return categories

#this function gets all linked(<a>) subreddits from given array of page links as an array and with a map with categories
def get_linked_subreddits_from_pages(links):
    subreddits = []
    subreddits_map = {}
    for page in links:
        http = httplib2.Http()
        status, response = http.request(page)
        for link in BeautifulSoup(response, parse_only=SoupStrainer("a")):
            if link.has_attr("href") and subreddit_sep in link["href"]:
                sub = get_subreddit_from_link(link["href"])
                subreddits.append(sub)
                if not sub in subreddits_map:
                    subreddits_map[sub] = get_subreddit_categories(sub,response)
    return subreddits, subreddits_map
    

forbidden = [""," ",".  ","ListOfSubreddits","all"]

def get_linked_subreddits_from_pages_faster(links):
    subreddits = []
    categories = []
    for page in links:
        http = httplib2.Http()
        status, response = http.request(page)
        bs = BeautifulSoup(response)
        for link in bs.find_all("a"):
            if link.has_attr("href") and subreddit_sep in link["href"]:
                sub = get_subreddit_from_link(link["href"])
                cat = link.parent.previous_element.previous_element
                if hasattr(cat, 'text'): cat = cat.text
                if not "<" in sub and not sub in forbidden and not "<" in cat and not "\n" in cat and not cat in forbidden:
                    subreddits.append(sub)
                    categories.append(cat)
    return subreddits, categories