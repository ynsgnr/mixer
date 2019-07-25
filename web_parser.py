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

levels = ["h5","h4","h3","h2","h1"]

def get_linked_subreddits_from_pages_faster(links):
    subreddits = []
    categories = []
    big_categories = []
    for page in links:
        http = httplib2.Http()
        status, response = http.request(page)
        bs = BeautifulSoup(response)
        for link in bs.find_all("a"):
            if link.has_attr("href") and subreddit_sep in link["href"]:
                cats = []
                sub = get_subreddit_from_link(link["href"])

                #Get all the categories
                cat = link.parent.find_previous_sibling(levels)
                if cat is None:
                    cat = link.parent.previous_element.previous_element
                    bcat = cat.parent.parent
                else:
                    bcat = cat
                
                if hasattr(cat, 'text'): cat = cat.text

                for i,level in enumerate(levels[0:-1]):
                    if bcat.name == level:
                        cats.append(bcat.text)
                        bcat = bcat.find_previous_sibling(levels[i+1])
                if bcat.name in levels and not bcat.text == cat:
                    cats.append(bcat.text)
                
                if len(cats)==0:
                    cats.append(cat)

                if page == "https://www.reddit.com/r/ListOfSubreddits/wiki/nsfw":
                    cats.append("NSFW")

                if not "<" in sub and not sub in forbidden and not "<" in cat and not "\n" in cat and not cat in forbidden:
                    subreddits.append(sub)
                    categories.append(cats)
    return subreddits, categories

def clear_subreddit(sub):
    if "/" in sub :
        w = sub.split("/")
        if "r" in sub:
            return w[w.index("r")+1]
        else:
            for s in w:
                if len(s)>0: return s
    return sub

reddit_link = "https://www.reddit.com/r/"
title_prev = "r/"
def subreddit_to_link(sub):
    return reddit_link+sub

def add_cookie(cookies,cookie_to_add):
    splited = cookies.split("secure")
    splited[-2]+=cookie_to_add
    return "secure".join(splited)

first_request = True
headers = {}
# import requests
def get_description(sub):
    global headers
    global first_request
    sub_name = clear_subreddit(sub)
    link = subreddit_to_link(sub_name)
    http = httplib2.Http()
    if first_request:
        first_request = False
        status, response = http.request(link)
        headers = {'Cookie': add_cookie(status['set-cookie'],"over18=1; ") }
    status, response = http.request(link,'GET',headers=headers)
    bs = BeautifulSoup(response)
    detail_titles = bs.findAll("span")
    real_sub_name = sub_name
    detail_title = None
    desc = ""
    for dt in detail_titles:
        if dt.has_attr('title'):
            detail_title = dt.parent
            real_sub_name = dt.text
    try:
        desc = detail_title.parent.parent.find(attrs={"data-redditstyle":"true"}).text
    except:
        print(detail_title)
        print(sub)
        print(real_sub_name)
        if not (detail_title is None):
            print(detail_title.parent)
            if not (detail_title.parent is None):
                print(detail_title.parent.parent)
                if not (detail_title.parent.parent is None):
                    print(detail_title.parent.parent.find(attrs={"data-redditstyle":"true"}))
    return desc, real_sub_name
    

def write_to_file(path,str):
    with open(path, 'w') as the_file:
        the_file.write(str)