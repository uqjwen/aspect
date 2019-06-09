import gensim
import re 
import numpy as np 
import xml.etree.ElementTree as ET 
import io, json 
import nltk
import os
from nltk import word_tokenize
import pickle
from nltk.tag import StanfordPOSTagger
pos_tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')



class MySentence():
	def __init__(self,sentence):
		self.sentence = sentence
	def __iter__(self):
		for sentence in self.sentence:
			yield sentence





def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""

	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)

	string = re.sub(r" \'", " ", string)
	string = re.sub(r"\' ", " ",string)

	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip()

def get_loc(sent, term):
	locs = []
	for i,t in enumerate(sent):
		if t == term[0]:
			begin = i 
			j = i+1
			while j<len(sent) and j-i<len(term) and term[j-i]==sent[j]:
				j+=1
			if j-i>=len(term):
				end = j
				locs.append((begin,end))
	return locs



def get_label(sent, terms):
	sent_tokens = word_tokenize(sent)
	label = np.zeros(len(sent_tokens))
	for term in terms:
		term_tokens = word_tokenize(term)
		locs = get_loc(sent_tokens, term_tokens)
		for loc in locs:
			begin,end = loc
			for i in range(begin,end):
				label[i] = 1 if i==begin else 2
	return sent_tokens, label


def process_laptop_file(filename):

	xmlFilePath = os.path.abspath(filename)
	tree = ET.parse(xmlFilePath)
	root = tree.getroot()

	all_sent 	= []
	all_label 	= []
	all_tag		= []
	all_term 	= []
	all_cats 	= []
	for sent in root.iter("sentence"):
		sent_text = sent.find("text").text.lower()

		clean_sent = clean_str(sent_text)

		terms = []
		for aspect in sent.iter('aspectTerm'):
			term = aspect.attrib['term'].lower()
			term = clean_str(term)
			terms.append(term)
		if len(terms)==0:
			continue

		cats = []
		for category in sent.iter('aspectCategory'):
			cat = category.attrib['category']
			cats.append(cat)
		if(len(cats)) == 0:
			print('terms not none cat nont')
		# print(sent_text)
		print(clean_sent)

		tokens, label = get_label(clean_sent, terms)
		tags = pos_tagger.tag(tokens)
		tags = [tag[1] for tag in tags]
		# tags = []
		print(tokens)
		print(label)
		print(tags)
		print(terms)
		print(cats)
		print('---------------------------------')


		all_sent.append(tokens)
		all_label.append(label)
		all_tag.append(tags)
		all_term.append(terms)
		all_cats.append(cats)
		# break
	return all_sent, all_label, all_tag, all_term, all_cats
		# print(tags)

def process(domain):

	train  	= process_laptop_file(domain+'_train')
	test 	= process_laptop_file(domain+'_test')

	# for item1,item2 in zip(train,test):
	data = []
	for item1,item2 in zip(train,test):
		data.append(item1+item2)


	build_vocab(data, 'data_'+domain+'.pkl')


	sents = MySentence(data[0])
	model = gensim.models.Word2Vec(sents, size=100, window = 5, min_count=1, workers=4)
	model.save('new_gensim_'+domain)
		# print(sent_text)
		# print(clean_sent)
		# print('-----------------------------------')


def build_vocab(data, filename):
	sentences, labels, tags, terms, cats = data[0],data[1],data[2],data[3],data[4]
	dic = {}
	for sen in sentences:
		for word in sen:
			dic[word] = dic.get(word,0)+1
	tag_dic = {}
	for tag in tags:
		for t in tag:
			tag_dic[t] = dic.get(t,0)+1

	tag2idx = dict((tag,i+1) for i,tag in enumerate(tag_dic.keys()))
	idx2tag = dict((i+1,tag) for i,tag in enumerate(tag_dic.keys()))


	word2idx = dict((word, i+1) for i,word in enumerate(dic.keys()))
	idx2word = dict((i+1, word) for i,word in enumerate(dic.keys()))



	cat_labels_set = []
	for cat in cats:
		for c in cat:
			if c not in cat_labels_set:
				cat_labels_set.append(c)



	clabel2idx = dict((c,i) for i,c in enumerate(cat_labels_set))
	idx2clabel = dict((i,c) for i,c in enumerate(cat_labels_set))







	data = {}
	data['word2idx'] = word2idx
	data['idx2word'] = idx2word
	data['vocab_size'] = len(word2idx)+1
	sen = [ [word2idx[word] for word in sentence] for sentence in sentences]
	data['labels'] = labels
	data['processed_sentence'] = sen
	data['raw_sentence'] = sentences

	tag = [[tag2idx[t] for t in tag] for tag in tags]
	data['tags'] = tag
	data['tag2idx'] = tag2idx
	data['idx2tag'] = idx2tag

	data['terms'] = terms


	cat_labels = [[clabel2idx[c] for c in cat] for cat in cats]
	data['cat_labels'] = cat_labels
	data['clabel2idx'] = clabel2idx
	data['idx2clabel'] = idx2clabel
	data['num_cat'] = len(cat_labels_set)





	pickle.dump(data,open(filename, 'wb'))

def laptop_cat():
	filename = 'annot_cat.txt'
	fr = open(filename)
	data = fr.readlines()
	fr.close()
	fr = open('data_laptop.pkl','rb')
	laptop = pickle.load(fr)
	fr.close()

	cats = []
	cats_set = []
	for line in data:
		line = line.strip()
		listfromline = line.split()
		# cats.append(listfromline[-1].split(';'))
		cat = listfromline[-1].split(';')
		for c in cat:
			if c not in cats_set:
				cats_set.append(c)
		cats.append(cat)
	print(' '.join(cats_set))

	assert len(cats) == len(laptop['processed_sentence'])

	clabel2idx = dict((c,i) for i,c in enumerate(cats_set))
	idx2clabel = dict((i,c) for i,c in enumerate(cats_set))
	cat_labels = [[clabel2idx[c] for c in cat] for cat in cats]
	laptop['cat_labels'] = cat_labels
	laptop['clabel2idx'] = clabel2idx
	laptop['idx2clabel'] = idx2clabel
	laptop['num_cat'] = len(cats_set)

	fr = open('data_laptop.pkl','wb')
	pickle.dump(laptop,fr)
	fr.close()

# sentences, labels, tags, terms, cats = data[0],data[1],data[2],data[3],data[4]	
def laptop_sent():
	fr 		= open('data_laptop.pkl','rb')
	data 	= pickle.load(fr)
	fr.close()
	sent 	= data['raw_sentence']

	import nltk.stem as ns 
	lemmatizer = ns.WordNetLemmatizer()


	sent 	= [[lemmatizer.lemmatize(lemmatizer.lemmatize(word,'n'),'v') for word in tokens] for tokens in sent]
	data['raw_sentence'] = sent



	dic = {}
	for s in sent:
		for word in s:
			dic[word] = dic.get(word,0)+1

	word2idx = dict((word, i+1) for i,word in enumerate(dic.keys()))
	idx2word = dict((i+1, word) for i,word in enumerate(dic.keys()))

	data['word2idx'] = word2idx
	data['idx2word'] = idx2word
	data['vocab_size'] = len(word2idx)+1

	data['processed_sentence'] = [[word2idx[word] for word in s] for s in sent]

	pickle.dump(data,open('data_laptop_2014.pkl','wb'))


	model = gensim.models.Word2Vec(sent, size=100, window = 5, min_count=1, workers=4)
	model.save('gensim_laptop_2014')




def annot_cat():
	fr = open('/home/wenjh/Desktop/gitlab/aspect/data/sent_cat_laptop.txt')
	data = fr.readlines()
	fr.close()
	res = []
	for line in data:
		line = line.strip()
		listfromline = line.split()
		cats = listfromline[-1].split(';')
		for cat in cats:
			if cat not in res:
				res.append(cat)

	fr = open('data_laptop.pkl','rb')
	data = pickle.load(fr)
	fr.close()


	fr = open('annot_cat.txt', 'r')
	cat_data = fr.readlines()
	fr.close()
	begin = len(cat_data)


	fr = open('annot_cat.txt','a')


	all_cat = []
	for sent,terms in zip(data['raw_sentence'][begin:],data['terms'][begin:]):
		string = ''
		for i,cat in enumerate(res):
			string += str(i)+':'+cat+' '
		
		print(string)
		print(' '.join(sent))
		print('#'.join(terms))
		index = list(map(int,input().split('.')))
		cats = np.array(res)[index].tolist()
		cats = ';'.join(cats)
		fr.write(' '.join(sent)+' '+cats+'\n')



def change_pos():
	fr = open('data_laptop_2014.pkl','rb')
	data = pickle.load(fr)
	fr.close()

	tags = [nltk.pos_tag(sent) for sent in data['raw_sentence']]
	tags = [[t[1] for t in tag] for tag in tags]
	all_tags = []
	for tag in tags:
		for t in tag:
			if t not in all_tags:
				all_tags.append(t)
	tag2idx = dict((t,i+1) for i,t in enumerate(all_tags))
	idx2tag = dict((i+1,t) for i,t in enumerate(all_tags))

	tags = [[tag2idx[t] for t in tag] for tag in tags]
	data['tags'] = tags
	data['tag2idx'] = tag2idx
	data['idx2tag'] = idx2tag

	pickle.dump(data, open('data_laptop.pkl','wb'))



if __name__ == '__main__':
	# process_laptop_file('Laptop_Train_v2.xml')
	# process('rest')
	# process('laptop')
	# annot_cat()
	# laptop_cat()
	# laptop_sent()
	change_pos()