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


def get_label(sent, terms):
	sent_tokens = word_tokenize(sent)
	label = np.zeros(len(sent_tokens))
	for term in terms:
		term_tokens = word_tokenize(term)
		if term_tokens[0] not in sent_tokens:
			print (sent)
		index = sent_tokens.index(term_tokens[0])
		label[index] = 1
		for i in range(1,len(term_tokens)):
			label[index+i] = 2
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
			category.attrib['category']
			cats.append(category)
		if(len(cats)) == 0:
			print('terms not none cat nont')
		# print(sent_text)
		print(clean_sent)
		print('---------------------------------')

		tokens, label = get_label(clean_sent, terms)
		tags = pos_tagger.tag(tokens)
		tags = [tag[1] for tag in tags]
		tags = []


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







if __name__ == '__main__':
	# process_laptop_file('Laptop_Train_v2.xml')
	# process('laptop')
	# process('rest')
	annot_cat()