
import nltk
import numpy as np 
import re
import os
import xml.etree.ElementTree as ET 
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import pickle
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
import time


pos_tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
# ner_tagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
# dpdc_parser = StanfordDependencyParser(model_path=u'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

# def dpdc_parse(sent):
# 	res = list(dpdc_parser.parse(sent))
# 	for row in res[0].triples():
# 		print(row)


# def get_sentence_()
split_re = "[- ;!.,-?\"\'\)\()]"

def my_split(line):
	tokens = []
	token=''
	idx = 0
	line+="#"
	while idx<len(line):
		if line[idx].isalpha():
			token += line[idx].lower()
		else:
			if len(token)>0:
				tokens.append(token)
				token = ''
		idx+=1
	return tokens





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





class MySentence():
	def __init__(self,sentence):
		self.sentence = sentence
	def __iter__(self):
		for sentence in self.sentence:
			yield sentence


def parseSentence(line):
	lmtzr = WordNetLemmatizer()	
	stop = stopwords.words('english')
	text_token = CountVectorizer().build_tokenizer()(line.lower())
	text_rmstop = [i for i in text_token if i not in stop]
	text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
	return text_stem


def processSentence(tokens, labels):
	lmtzr = WordNetLemmatizer()
	new_tokens = []
	new_labels = []
	for token,label in zip(tokens, labels):
		# if token.lower() not in stopwords.words("english")+['']:
		if token is not '':
			# new_tokens.append(lmtzr.lemmatize(token.lower()))
			if token.isdigit():
				new_tokens.append("#num")
			else:
				new_tokens.append(token.lower())
			new_labels.append(label)
	return new_tokens, new_labels


def process_file(filename):
	# filename = './data/ABSA16_Restaurants_Train_SB1_v2.xml'



	xmlFilePath = os.path.abspath(filename)
	tree = ET.parse(xmlFilePath)
	root = tree.getroot()
	# print(dir(root))
	sentences = []
	labels = []
	pos = []

	all_sent = []
	all_label = []
	all_tag = []
	all_term = []
	all_cat = []


	import nltk.stem as ns 
	lemmatizer = ns.WordNetLemmatizer()


	for sent in root.iter("sentence"):
		# print(word_tokenize(sen.find("text").text))
		sent_text = sent.find("text").text.lower()

		clean_sent = clean_str(sent_text)
		terms 	= []
		cats 	= []
		for op in sent.iter('Opinion'):
			if op.attrib['target'] != 'NULL':
				# terms.append()
				term = op.attrib['target'].lower()
				term = clean_str(term)
				terms.append(term)

				cat = op.attrib['category'].split('#')[0]
				cats.append(cat)

		tokens, label = get_label(clean_sent, terms)
		tokens = [lemmatizer.lemmatize(lemmatizer.lemmatize(word,'n'),'v') for word in tokens]


		if len(cats)== 0:
			continue


		tags = pos_tagger.tag(tokens)
		tags = [tag[1] for tag in tags]
		# tags = []
		print(tokens)
		print(label)
		print(tags)
		print(terms)
		print(cats)
		print('---------------------------------')



		# if len(terms) == 0:
		# 	continue
		# if len(terms) == 0:
		# 	cats.append('unknown')

		all_sent.append(tokens)
		all_label.append(label)
		all_tag.append(tags)
		all_term.append(terms)
		all_cat.append(cats)



	# return sentences, labels, pos
	return all_sent, all_label, all_tag, all_term, all_cat

def process_domain(domain):
	train 	= process_file('./data/ABSA16_Restaurants_Train_SB1_v2.xml')
	print('train_size', len(train))

	test 	= process_file('./data/EN_REST_SB1_TEST_gold.xml')
	data = []
	for item1, item2 in zip(train, test):
		data.append(item1+item2)
	build_vocab(data, './pkl/data_'+domain+'_2016_2.pkl')

	sents = MySentence(data[0])
	model = gensim.models.Word2Vec(sents, size = 100, window = 5, min_count=1, workers = 4)
	model.save('./pkl/gensim_'+domain+"_2016_2")


		# break
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

def process_unsupervised_sent(tokens):
	new_tokens = []
	for token in tokens:
		if token is not '':
			new_tokens.append(token.lower())
	return new_tokens

def get_unsupervised_sent(filename):


	fr = open(filename)
	data = fr.readlines()
	sentences = []
	tags = []
	labels = []
	for line in data:
		tokens = re.split(split_re, line)
		new_tokens = process_unsupervised_sent(tokens)
		sentences.append(new_tokens)
		tag = pos_tagger.tag(new_tokens)
		tag = [tp[1] for tp in tag]
		# tags = [tp[1] for tp in tags]
		########################################## tags.append(pos_tagger.tag(new_tokens))
		tags.append(tag)
		labels.append([0]*len(new_tokens))
	return sentences, tags, labels



def processFile():
	sent1, label1, tag1 = get_sentence_labels('./data/ABSA16_Restaurants_Train_SB1_v2.xml')
	print("training 1 finished ...")
	# print('t')
	sent2, label2, tag2 = get_sentence_labels('./data/EN_REST_SB1_TEST_gold.xml')
	print('testing 1 finished ...')

	# sent3, label3, tag3 = get_sentence_labels('./data/ABSA16_Laptops_Train_SB1_v2.xml')
	# print("training 2 finished...")
	# sent4, label4, tag4 = get_sentence_labels('./data/EN_LAPT_SB1_TEST_.xml.gold')
	# print("testing 2 finished...")
	# # sent3, tag3 = get_unsupervised_sent('./data/train.txt')
	# label_mask = [1]*(len(sent1)+len(sent2)) +[0]*(len(sent3)+len(sent4))

	# sent = sent1+sent2+sent3+sent4
	# label = label1+label2+label3+label4
	# tag = tag1+tag2+tag3+tag4

	# print(len(sent1), len(sent2), len(sent3), len(sent4))  #2000 676 2500 808

	# supervise_size = len(sent1+sent2)
	# return sent, label, tag, label_mask, supervise_size


def processLaptop(filename):
	fr = open(filename)
	data = fr.readlines()
	fr.close()

	sents 	= []
	labels 	= []
	tags 	= []

	import nltk.stem as ns 
	lemmatizer = ns.WordNetLemmatizer()
	for i in range(0,len(data),2):
		sent = data[i].strip().split()

		sent = [lemmatizer.lemmatize(lemmatizer.lemmatize(word,'n'),'v') for word in sent]

		label = list(map(int,data[i+1].strip().split()))

		# tag = 

		tag = pos_tagger.tag(sent)
		# tags = nltk.pos_tag(tokens)

		tag = [tp[1] for tp in tag]

		sents.append(sent)
		labels.append(label)
		tags.append(tag)
		print(sent)
		print(tag)
		print("-----------------------------------------")

	sen = MySentence(sents)
	model = gensim.models.Word2Vec(sen, size=100, window=5, min_count=1, workers=4)
	model.save("gensim_laptop")

	train_size = len(sents)
	label_mask = [1]*len(sents)
	build_vocab(sents, labels, tags, train_size=train_size, emb_size=100, label_mask=label_mask)

def processCat(filename):
	filename_1 = 'data_cat_'+filename+'.pkl'
	filename_2 = './data/sent_cat_'+filename+'.txt'
	fr = open(filename_1,'rb')
	data = pickle.load(fr)
	fr.close()
	# fr = open('./data/')
	fr = open(filename_2)
	catinfo = fr.readlines()
	fr.close()
	cat_labels = []
	cat_labels_set = []
	for line in catinfo:
		line = line.strip()
		listfromline = line.split()
		cat = listfromline[-1].split(';')
		cat_labels.append(cat)
		for c in cat:
			if c not in cat_labels_set:
				cat_labels_set.append(c)
		# if len(cat) == 1:
		# 	cat_labels.append("unknown")
		# else:
		# 	if cat[0] == 'LAPTOP':
		# 		cat_labels.append(cat[1])
		# 	else:
		# 		cat_labels.append(cat[0])
	# print(len(cat_labels))
	# print(len(data['processed_sentence']))
	# print(data['raw_sentence'][2675])

	assert len(cat_labels) == len(data['processed_sentence'])

	clabel2idx = dict((c,i) for i,c in enumerate(cat_labels_set))
	idx2clabel = dict((i,c) for i,c in enumerate(cat_labels_set))

	cat_labels = [[clabel2idx[c] for c in cat_label] for cat_label in cat_labels]
	data['cat_labels'] = cat_labels
	data['clabel2idx'] = clabel2idx
	data['idx2clabel'] = idx2clabel
	data['num_cat'] = len(cat_labels_set)

	# pickle.dump('data_cat_laptop.pkl', open())
	pickle.dump(data,open(filename_1, 'wb'))


def processLaptop():
	fr = open('data_cat_laptop.pkl', 'rb')
	data = pickle.load(fr)
	fr.close()

	def get_label_from_file(filename):
		fr = open(filename)
		data = fr.readlines()
		fr.close()
		labels = []
		for i in range(1,len(data),2):
			line = data[i].strip()
			listfromline = line.split()
			label = list(map(int,listfromline))
			labels.append(label)
		return labels
	labels = get_label_from_file('./data/sent_annot.txt')	
	assert len(labels) == len(data['labels'])
	data['labels'] = labels
	pickle.dump(data, open('data_cat_laptop.pkl', 'wb'))

if __name__ == '__main__':
	# main()
	# parser = argparser.ArgumentParser()
	# sentences, labels, tags = get_sentence_labels('./data/ABSA16_Restaurants_Train_SB1_v2.xml')
	# test_sent, test_label, test_tag = get_sentence_labels('./data/EN_REST_SB1_TEST_gold.xml')


	# label_mask = [1]*(len)


	#------------------------------------------------------------
	# start = time.time()
	# sent, label, tag, label_mask, supervise_size = processFile()

	# sen = MySentence(sent)
	# model = gensim.models.Word2Vec(sen, size = 100, window = 5, min_count=0, workers = 4)
	# model.save("my_gensim_model")
	# build_vocab(sent, label, tag, train_size = supervise_size, emb_size = 100, label_mask = label_mask)
	# end = time.time()
	# print(end-start)
	#------------------------------------------------------------#

	# processLaptop('./data/sent_annot.txt')
	# processCat()
	# processRes()
	# processLaptop()
	# processCat('laptop')
	process_domain('rest')