
from preprocessing.space_sprinkler import *
from preprocessing.data2disk import *
from preprocessing.tokenizer import *
from preprocessing.web_parser import *

def get_spaced_subreddits(links):
    subreddits, categories = get_linked_subreddits_from_pages_faster(links)

    sprinkled_subs = sprinkle_on_subreddits(subreddits)

    return sprinkled_subs, subreddits, categories

def get_categorized(categories):
    categories_dict = {}
    categories_dict_subs = {}
    for i,category in enumerate(categories):
        if not category[-1] in categories_dict:
            categories_dict[category[-1]]=1
            categories_dict_subs[category[-1]]=[i]
        else:
            categories_dict[category[-1]]+=1
            categories_dict_subs[category[-1]].append(i)
    return categories_dict, categories_dict_subs

def remove_categories(cats_to_remove,categories,subreddits,sprinkled_subs):
    #remove Ex50k+ category
    filtered = (
        (cat, sub, sp_sub) 
            for cat, sub, sp_sub in zip(categories, subreddits, sprinkled_subs) 
                if not cat[-1] in cats_to_remove)

    c, s, ss = zip(*filtered)
    return list(c),list(s),list(ss)

def get_descriptions(subreddits,sprinkled_subs):
    subs_sentences = [" "] * len(subreddits)
    for i,sentence in enumerate(sprinkled_subs):
        desc, real_sub_name = get_description(subreddits[i])
        if not subreddits[i] == real_sub_name:
            sentence = sprinkle_on_subreddits([real_sub_name.split("/")[-1]])[0]
        subs_sentences[i] = sentence + " " + desc
    return subs_sentences

def get_data(stemmed_data_path,data_path,links):
    #Get necessary data from file if possible
    #If file doesnt exists construct data from links
    sprinkled_subs,categories = load_data(stemmed_data_path)
    if sprinkled_subs is None:
        sprinkled_subs,categories = load_data(data_path)
        if sprinkled_subs is None:
            #Get Data
            sprinkled_subs, subreddits, categories = get_spaced_subreddits(links)
            categories, subreddits, sprinkled_subs = remove_categories(["Ex 50k+"],categories,subreddits,sprinkled_subs)
            sprinkled_subs = get_descriptions(subreddits,sprinkled_subs)
            save_data(data_path,sprinkled_subs,categories)

        #Stem sub
        sprinkled_subs = stem_subs(sprinkled_subs)
        save_data(stemmed_data_path,sprinkled_subs,categories)
    return sprinkled_subs,categories

def get_category_details(categories,categories_dict_subs=None):
    #get detailed categories for other
    if categories_dict_subs is None:
        _ , categories_dict_subs = get_categorized(categories)
    for subreddit_i in categories_dict_subs['Other']:
        categories[subreddit_i]=categories[subreddit_i][0:-1]
    categories_dict, categories_dict_subs = get_categorized(categories)
    return categories_dict, categories_dict_subs
