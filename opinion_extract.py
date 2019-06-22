from nltk.parse.stanford import StanfordDependencyParser
import pickle
import sys
# filename = './pkl/'
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

if __name__ == '__main__':
	get_parser(sys.argv[1])