
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
		if token.lower() not in stopwords.words("english")+['']:
		# if token is not '':
			new_tokens.append(lmtzr.lemmatize(token.lower()))
			# new_tokens.append(token.lower())
			new_labels.append(label)
	return new_tokens, new_labels


def get_sentence_labels(filename):
	# filename = './data/ABSA16_Restaurants_Train_SB1_v2.xml'

	xmlFilePath = os.path.abspath(filename)
	tree = ET.parse(xmlFilePath)
	root = tree.getroot()
	# print(dir(root))
	sentences = []
	labels = []
	pos = []


	for sen in root.iter("sentence"):
		# print(word_tokenize(sen.find("text").text))
		sentext = sen.find("text").text
		# tokens = word_tokenize(sentext)
		tokens = re.split(split_re, sentext)
		# tokens = my_split(sentext)
		y_labels = [0]*len(tokens)
		# print(sentext)
		# print(tokens)
		for op in sen.iter("Opinion"):
			# print(op.attrib['from'], op.attrib['to'])
			if 'target' not in op.attrib:
				continue
			begin = int(op.attrib['from'])
			end = int(op.attrib['to'])

			if end>begin and op.attrib["target"]!="NULL":
				# print(re.split(split_re, sentext[:begin]), tokens)

				begin_idx = 0 if begin==0 else len(re.split(split_re, sentext[:begin].strip()))
				end_idx = len(re.split(split_re, sentext[:end].strip()))

				y_labels[begin_idx] = 1
				for idx in range(begin_idx+1, end_idx):
					y_labels[idx] = 2
			temp = [tokens[i] for i,label in enumerate(y_labels) if label!=0]


		tokens, sen_labels = processSentence(tokens,y_labels)

		tags = pos_tagger.tag(tokens)

		tags = [tp[1] for tp in tags]


		# tags = [0]*len(tokens)


		sentences.append(tokens)
		labels.append(sen_labels)
		pos.append(tags)


	return sentences, labels, pos
		# break
def build_vocab(sentences, labels,tags, train_size, emb_size, label_mask):
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

	data = {}
	data['word2idx'] = word2idx
	data['idx2word'] = idx2word
	data['vocab_size'] = len(word2idx)+1
	sen = [ [word2idx[word] for word in sentence] for sentence in sentences]
	data['labels'] = labels
	data['processed_sentence'] = sen 
	data['raw_sentence'] = sentences
	data['emb_size'] = emb_size
	data['train_size'] = train_size

	tag = [[tag2idx[t] for t in tag] for tag in tags]
	data['tags'] = tag
	data['tag2idx'] = tag2idx
	data['idx2tag'] = idx2tag


	data['label_mask'] = np.array(label_mask).astype(np.float32)

	pickle.dump(data,open('data.pkl', 'wb'))

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
	sent2, label2, tag2 = get_sentence_labels('./data/EN_REST_SB1_TEST_gold.xml')
	print('testing 1 finished ...')

	sent3, label3, tag3 = get_sentence_labels('./data/ABSA16_Laptops_Train_SB1_v2.xml')
	print("training 2 finished...")
	sent4, label4, tag4 = get_sentence_labels('./data/EN_LAPT_SB1_TEST_.xml.gold')
	print("testing 2 finished...")
	# sent3, tag3 = get_unsupervised_sent('./data/train.txt')
	label_mask = [1]*(len(sent1)+len(sent2)) +[0]*(len(sent3)+len(sent4))

	sent = sent1+sent2+sent3+sent4
	label = label1+label2+label3+label4
	tag = tag1+tag2+tag3+tag4

	print(len(sent1), len(sent2), len(sent3), len(sent4))  #2000 676 2500 808

	supervise_size = len(sent1+sent2)
	return sent, label, tag, label_mask, supervise_size



if __name__ == '__main__':
	# main()
	# parser = argparser.ArgumentParser()
	# sentences, labels, tags = get_sentence_labels('./data/ABSA16_Restaurants_Train_SB1_v2.xml')
	# test_sent, test_label, test_tag = get_sentence_labels('./data/EN_REST_SB1_TEST_gold.xml')


	# label_mask = [1]*(len)

	sent, label, tag, label_mask, supervise_size = processFile()

	# print(len(sent))
	sen = MySentence(sent)
	model = gensim.models.Word2Vec(sen, size = 100, window = 5, min_count=0, workers = 4)
	model.save("my_gensim_model")
	build_vocab(sent, label, tag, train_size = supervise_size, emb_size = 100, label_mask = label_mask)
