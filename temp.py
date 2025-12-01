import json
import string
# with open('rag-search-engine/data/movies.json', 'r') as f:
#     data  = json.load(f)
# print((data['movies'][0]))

# context = data['movies'][0]['title']
# query = 'Police'
# print(query in context)

punc_table = str.maketrans('','',string.punctuation)