
import re
import os
import xml.etree.ElementTree as ET 
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
# import gensim
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
			new_tokens.append(token.lower())
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
		print(tokens)
		print(sen_labels)
		print(pos_tagger.tag(tokens))
		print('----------------------------')
		tags = pos_tagger.tag(tokens)
		tags = [tp[1] for tp in tags]
		# print(ner_tagger.tag(tokens))
		# dpdc_parse(tokens)
		print("###########################################")



		sentences.append(tokens)
		labels.append(sen_labels)
		pos.append(tags)
		# print(y_labels)
		# break

	return sentences, labels, pos
		# break
def build_vocab(sentences, labels,tags, train_size, emb_size):
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

	pickle.dump(data,open('data.pkl', 'wb'))



if __name__ == '__main__':
	# main()
	# parser = argparser.ArgumentParser()
	sentences, labels, tags = get_sentence_labels('./data/ABSA16_Restaurants_Train_SB1_v2.xml')
	test_sent, test_label, test_tag = get_sentence_labels('./data/EN_REST_SB1_TEST_gold.xml')

	print(len(sentences))
	sen = MySentence(sentences+test_sent)
	# model = gensim.models.Word2Vec(sen, size = 100, window = 5, min_count=0, workers = 4)
	# model.save("my_gensim_model")
	build_vocab(sentences+test_sent, labels+test_label, tags+test_tag, train_size = len(sentences), emb_size = 100)
	# for word in model.wv.vocab:
	# 	print(word)