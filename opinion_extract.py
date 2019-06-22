from nltk.parse.stanford import StanfordDependencyParser
import pickle
import sys
from collections import Counter
# filename = './pkl/'
from nltk.corpus import stopwords

data_file = './pkl/data_laptop_2014.pkl'

def get_parser(domain):
	if domain == 'laptop':
		suffix = '_2014.pkl'
	else:
		suffix = '_2016_2.pkl'
	filename = './pkl/data_'+domain+suffix
	save_filename = './pkl/dep_'+domain+suffix
	fr = open(filename, 'rb')
	data = pickle.load(fr)
	sents = data['raw_sentence']
	labels = data['labels']
	fr.close()

	new_data = {}
	dep = []


	eng_parser = StanfordDependencyParser(model_path=u'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
	for sent in sents:
		res = list(eng_parser.parse(sent))
		sent_dep = []
		# dep.append(res)
		for row in res[0].triples():
			print(row)
			sent_dep.append(row)
		dep.append(sent_dep)
		# break
	new_data['raw_sentence'] = sents
	new_data['dependency'] = dep
	new_data['labels'] = labels
	fr = open(save_filename,'wb')
	pickle.dump(new_data,fr)
	fr.close()
def get_pos(filename):
	fr = open(filename, 'rb')
	data = pickle.load(fr)
	fr.close()
	tags = data['tags']
	idx2tag = data['idx2tag']
	tags = [[idx2tag[t] for t in tag] for tag in tags]
	return tags


def extraction(sents, deps, tags):
	curr_opinion = ['good','high','easy','great']


def opinion(domain):
	if domain == 'laptop':
		suffix = '_2014.pkl'
	else:
		suffix = '_2016_2.pkl'

	pos_file = './pkl/data_'+domain+suffix
	dep_file = './pkl/dep_'+domain+suffix

	tags = get_pos(pos_file)

	fr = open(dep_file,'rb')
	data = pickle.load(fr)
	fr.close()
	sents = data['raw_sentence']
	deps = data['dependency']
	labels = data['labels']






	for sent,tag, dep in zip(sents, tags, deps):
		# print(sent)
		# print(tag)
		# print(len(sent), len(tag))
		# print('-----------------------------')
		# break
		print(' '.join(sent))
		st = ''
		for s,t in zip(sent,tag):
			st += s+':'+t+' '
		print(st)
		# print('\t'.join(sent))
		# print(tag)
		# print('\t'.join(tag))
		# print(dep)

		for d in dep:
			print(d)
		print('--------------------------------')
		assert len(sent) == len(tag)



def frequency_counter(domain):
	if domain == 'laptop':
		suffix = '_2014.pkl'
	else:
		suffix = '_2016_2.pkl'
	data_file = './pkl/data_'+domain+suffix
	fr = open(data_file,'rb')
	data = pickle.load(fr)
	fr.close()
	sents = data['raw_sentence']

	import nltk.stem as ns 
	lemmatizer = ns.WordNetLemmatizer()

	sents = [[lemmatizer.lemmatize(lemmatizer.lemmatize(word,'n'),'v') for word in sent] for sent in sents]

	word_c = Counter()
	for sent in sents:
		word_c += Counter(sent)

	stop_words = list(set(stopwords.words('english')))
	print (stop_words)

	common_words = word_c.most_common(200)
	for word, count in common_words:
		if word not in stop_words:
			print(word, count)
	# print(word_c.most_common(10))

if __name__ == '__main__':
	# get_parser(sys.argv[1])
	opinion('laptop')
	# frequency_counter('laptop')