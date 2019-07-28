#space_sprinkle test
from space_sprinkler import *

test_dict1 = {'test':1,'test2':2,'test3':3}
test_dict2 = {'test':1,'test1':2,'test3':3}
expected = {'test':2,'test1':2,'test2':2,'test3':6}
result = merge_dict_by_adding(test_dict1,test_dict2)
assert set(expected) == set(result)

add_words(['test','lorem','ipsum','lorem','single'])
add_single_word('ipsum')
add_words(['test','lorem'])
add_single_word('extra')
print(freq_list)
print(counter)
#print(wordcost)