
import re
import os
import xml.etree.ElementTree as ET 
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import pickle
# def get_sentence_()
split_re = "[- ;!.,-?\"\'\)\()]"

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
			new_tokens.append(lmtzr.lemmatize(token.lower()))
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


	for sen in root.iter("sentence"):
		# print(word_tokenize(sen.find("text").text))
		sentext = sen.find("text").text
		# tokens = word_tokenize(sentext)
		tokens = re.split(split_re, sentext)
		y_labels = [0]*len(tokens)
		for op in sen.iter("Opinion"):
			# print(op.attrib['from'], op.attrib['to'])
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
			# print(temp, sentext[begin:end])
		# print(tokens)
		# print(y_labels)
		tokens, sen_labels = processSentence(tokens,y_labels)
		# print(tokens)
		# print(labels)
		sentences.append(tokens)
		labels.append(sen_labels)
		# print(y_labels)
		# break

	return sentences, labels
		# break
def build_vocab(sentences, labels):
	dic = {}
	for sen in sentences:
		for word in sen:
			dic[word] = dic.get(word,0)+1
	word2idx = dict((word, i+1) for i,word in enumerate(dic.keys()))
	data = {}
	data['word2idx'] = word2idx
	sen = [ [word2idx[word] for word in sentence] for sentence in sentences]
	data['labels'] = labels
	data['processed_sentence'] = sen 
	data['raw_sentence'] = sentences

	pickle.dump(data,open('data.pkl', 'wb'))



if __name__ == '__main__':
	# main()
	# parser = argparser.ArgumentParser()
	sentences, labels = get_sentence_labels('./data/ABSA16_Restaurants_Train_SB1_v2.xml')
	print(len(sentences))
	sen = MySentence(sentences)
	model = gensim.models.Word2Vec(sen, size = 100, window = 5, min_count=0, workers = 4)
	model.save("my_gensim_model")
	build_vocab(sentences, labels)
	# for word in model.wv.vocab:
	# 	print(word)