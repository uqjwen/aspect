import os 
import pickle
import numpy as np 
from keras.preprocessing.sequence import pad_sequences
#import torch
from gensim.models import Word2Vec
from keras.utils.np_utils import to_categorical
from nltk.corpus import stopwords


class Data_Loader():
	def __init__(self, batch_size, data_set):
		self.batch_size = batch_size
		# self.sent_len = sent_len
		# self.maxlen = maxlen
		# fr = open('data.pkl', 'rb')
		# fr = open('data_cat_laptop.pkl', 'rb')
		if data_set == 'res':
			data_file = 'data_cat_res.pkl'
		else:
			data_file='data_cat_laptop.pkl'
		fr = open(data_file, 'rb')
		data = pickle.load(fr)

		self.word2idx = data['word2idx']
		self.idx2word = data['idx2word']

		self.clabel2idx = data['clabel2idx']
		self.idx2clabel = data['idx2clabel']


		self.vocab_size = data['vocab_size']
		# self.emb_size = data['emb_size']

		self.label_mask = data['label_mask']


		self.num_tag = len(data['tag2idx'])+1

		self.num_cat = data['num_cat']
		self.clabels = data['cat_labels']
		# self.clabels = to_categorical(self.clabels, self.num_cat)
		self.clabels = self.my_categorical(self.clabels, self.num_cat)
		self.clabel_mask = self.get_cat_mask(self.clabels, data['clabel2idx'])


		



		self.emb_size = 100
		self.gen_size = 300
		self.psent 		= data['processed_sentence']
		labels 		= data['labels']
		tags = data['tags']


		# sentences, tags, labels 	= self.filter_stopwords(self.psent, tags, labels, data['idx2word'])
		sentences = self.psent

		tfidf 		= self.my_tfidf(sentences, self.clabels)

		
		# labels = self.get_label_from_file('./data/sent_annot.txt')
		assert len(labels) == len(sentences)

		# self.maxlen = max([len(sent) for sent in sentences])
		# self.maxlen = int(np.mean([len(sent) for sent in sentences]))
		self.maxlen = 36

		
		# sentences = pad_sequences(sentences,self.maxlen, padding='post')

		# tags = data['tags']
		# tags = pad_sequences(tags, self.maxlen, padding='post')



		# print(labels)
		self.emb_mat = self.embed_mat(data_set)
		self.gen_mat = self.genel_mat(data_set)


		self.labels 	= pad_sequences(labels, self.maxlen)

		self.sent 		= pad_sequences(sentences, self.maxlen)

		self.tfidf 		= pad_sequences(tfidf, self.maxlen)

		self.sent_tag 	= to_categorical(pad_sequences(tags, self.maxlen), self.num_tag)

		self.mask 		= np.ones((len(self.sent), self.maxlen))

		self.mask[self.sent==0] = 0
		self.pointer = 0


		self.data_size = len(self.sent)

		if 'permutation' not in data:
			self.permutation = np.random.permutation(self.data_size)
			data['permutation'] = self.permutation
			pickle.dump(data,open(data_file, 'wb'))
		else:
			self.permutation = data['permutation']

		# self.train_val_test() ## it splits training testing here
		self.train_test_split(self.permutation) ## it splits training testing here

	def my_tfidf(self, sents, clabels):
		clabel2sent = {}
		word2sent = {}

		for i,sent in enumerate(sents):
			for token in sent:
				if token not in word2sent:
					word2sent[token] = [i]
				elif i not in word2sent[token]:
					word2sent[token].append(i)
			for clabel in clabels[i]:
				if clabel not in clabel2sent:
					clabel2sent[clabel] = [i]
				else:
					clabel2sent[clabel].append(i)

		tfidf = []
		for sent in sents:
			from collections import Counter
			# print(sent)
			counter = Counter(sent)
			temp_tfidf = []
			for token in sent:
				tf = counter[token]
				idf = np.log(len(sents)*1.0/len(word2sent[token])+1)
				temp_tfidf.append(tf*idf)
			tfidf.append(temp_tfidf)
		return tfidf




	def filter_stopwords(self, sentences, tags, labels, idx2word):
		import string
		ascii_ = [c for c in string.ascii_lowercase]
		stop = stopwords.words('english')
		to_filter = ascii_+stop

		res_sent = []
		res_tags = []
		res_label= []
		for sentence, tag, label in zip(sentences,tags,labels):
			temp_sent = []
			temp_tags = []
			temp_label= []
			for i,token in enumerate(sentence):
				if idx2word[token] in to_filter:
					continue
				else:
					temp_sent.append(token)
					temp_tags.append(tag[i])
					temp_label.append(label[i])
			res_sent.append(temp_sent)
			res_tags.append(temp_tags)
			res_label.append(temp_label)
			# temp = [token for token in sentence if idx2word[token] not in to_filter]
			# sent.append(temp)
		return res_sent, res_tags, res_label

		
	def get_cat_mask(self, clabels, clabel2idx):
		cat_mask = []
		index = clabel2idx['unknown']
		for clabel in clabels:
			if len(clabel)==0 and clabel[0]==index:
				cat_mask.append(0)
			else:
				cat_mask.append(1)
		return np.array(cat_mask, dtype = np.float32)

	def my_categorical(self, labels, num_class):
		res = []
		for label in labels:
			temp = np.zeros(num_class)
			temp[label] = 1
			res.append(temp)
		return np.array(res,dtype = np.float32)
		# return res.astype(np.float32)

		# print(self.train_size)
	def get_label_from_file(self,filename):
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

	def embed_mat(self, data_set):
		if data_set == 'laptop':
			model = Word2Vec.load('gensim_laptop')
		else:
			model = Word2Vec.load('gensim_res')
		mat = np.random.uniform(-1,1,(self.vocab_size, self.emb_size))
		for i in range(1,self.vocab_size):
			mat[i] =  model[self.idx2word[i]]
		return mat

	def genel_mat(self, data_set):

		gen_file = 'gen_laptop.npy' if data_set == 'laptop' else 'gen_res.npy'

		if os.path.exists(gen_file):
			return np.load(gen_file)
		else:
			mat = np.random.uniform(-1,1,(self.vocab_size, self.gen_size))
			fr = open('/media/wenjh/Ubuntu 16.0/Downloads/glove.6B/glove.6B.300d.txt')
			data = fr.readlines()
			for line in data:
				line = line.strip()
				listfromline = line.split()
				word,vec = listfromline[0], listfromline[1:]
				if word in self.word2idx:
					index = self.word2idx[word]
					mat[index] = np.array(list(map(float,vec))).astype(np.float32)
			np.save(gen_file.split('.')[0], mat)
			return mat


	def reset_pointer(self):
		self.pointer = 0

	def __next__(self):
		begin = self.pointer*self.batch_size
		# end = min(self.train_size, (self.pointer+1)*self.batch_size)

		end = (self.pointer+1)*self.batch_size
		if (self.pointer+1)*self.batch_size >= self.train_size:
			end = self.train_size
			self.pointer = 0
		else:
			self.pointer+=1
		# temp = torch.from_numpy(self.labels[begin:end])
		# print(temp.dtype)
		# print(temp)
		# return torch.tensor(self.sent[begin:end], dtype=torch.long), torch.from_numpy(self.mask[begin:end]), torch.tensor(self.labels[begin:end],dtype=torch.long)
		# print(begin, end, self.train_size)

		# return self.sent[begin:end],self.sent_tag[begin:end], self.mask[begin:end], self.labels[begin:end], self.label_mask[begin:end]
		return self.train_sent[begin:end],\
				self.train_sent_tag[begin:end],\
				self.train_mask[begin:end],\
				self.train_labels[begin:end],\
				self.train_label_mask[begin:end],\
				self.train_cat_labels[begin:end],\
				self.train_tfidf[begin:end]


	def val(self, sample_rate = 0.3):
		# test_size = self.train_size - 2000
		val_size = len(self.val_sent)
		sample_size = int(val_size*sample_rate)

		idx = np.random.choice(range(len(self.val_sent)), sample_size, replace = False)


		v_sent 		= []
		v_sent_tag 	= []
		v_mask 		= []
		v_labels 	= []
		v_c_labels 	= []
		v_c_masks 	= []
		v_tfidf 	= []

		exists_dic = {}

		if sample_rate == 1:
			# print('\n')
			index = self.permutation[self.train_size:]
			return self.val_sent,\
					self.val_sent_tag, \
					self.val_mask, \
					self.val_labels, \
					self.val_cat_labels, \
					self.val_cat_mask, \
					index,\
					self.val_tfidf

		else:
			index = []
			while len(v_sent)<sample_size:
				idx = np.random.choice(range(len(self.val_sent)))
				if idx not in exists_dic:
					exists_dic[idx] = 1
					index.append(idx)
				else:
					continue
				# if np.sum(self.val_labels[idx])==0:
				# 	continue
				v_sent.append(self.val_sent[idx])
				v_sent_tag.append(self.val_sent_tag[idx])
				v_mask.append(self.val_mask[idx])
				v_labels.append(self.val_labels[idx])
				v_c_labels.append(self.val_cat_labels[idx])
				v_c_masks.append(self.val_cat_mask[idx])
				v_tfidf.append(self.val_tfidf[idx])
			# v_c_masks.append(self.)



		# return self.sent[idx], self.sent_tag[idx], self.mask[idx], self.labels[idx]
		# return self.val_sent[idx], self.val_sent_tag[idx], self.val_mask[idx], self.val_labels[idx]
		index = self.permutation[self.train_size:][index]
		return np.array(v_sent),\
				np.array(v_sent_tag),\
				np.array(v_mask),\
				np.array(v_labels),\
				np.array(v_c_labels),\
				np.array(v_c_masks),\
				index,\
				np.array(v_tfidf)




	def train_test_split(self, permutation):
		train_size = int(self.data_size*0.8)

		# permutation = np.random.permutation(self.data_size)

		train_pmt = permutation[:train_size]
		test_pmt = permutation[train_size:]

		self.train_sent 		= self.sent[train_pmt]
		self.train_sent_tag		= self.sent_tag[train_pmt]
		self.train_mask 		= self.mask[train_pmt]
		self.train_labels 		= self.labels[train_pmt]
		self.train_label_mask 	= self.label_mask[train_pmt]
		self.train_cat_labels	= self.clabels[train_pmt]
		self.train_cat_mask 	= self.clabel_mask[train_pmt]
		self.train_tfidf		= self.tfidf[train_pmt]



		self.val_sent 			= self.sent[test_pmt]
		self.val_sent_tag		= self.sent_tag[test_pmt]
		self.val_mask 			= self.mask[test_pmt]
		self.val_labels 		= self.labels[test_pmt]
		self.val_label_mask 	= self.label_mask[test_pmt]
		self.val_cat_labels		= self.clabels[test_pmt]
		self.val_cat_mask 		= self.clabel_mask[test_pmt]
		self.val_tfidf			= self.tfidf[test_pmt]

		self.train_size 		= train_size





	def train_val_test(self):
		val_b = 2000
		val_e = 2676

		self.val_sent 		= self.sent[val_b:val_e]
		self.val_sent_tag 	= self.sent_tag[val_b:val_e]
		self.val_mask 		= self.mask[val_b:val_e]
		self.val_labels 	= self.labels[val_b:val_e]
		self.val_label_mask = self.label_mask[val_b:val_e]

		self.sent 		= np.delete(self.sent, range(val_b, val_e), axis=0)
		self.sent_tag 	= np.delete(self.sent_tag, range(val_b, val_e), axis=0)
		self.mask 		= np.delete(self.mask, range(val_b, val_e), axis=0)
		self.labels 	= np.delete(self.labels, range(val_b, val_e), axis=0)
		self.label_mask = np.delete(self.label_mask, range(val_b, val_e), axis=0)

		self.test_sent 			= self.sent[val_b:]
		self.test_sent_tag 		= self.sent_tag[val_b:]
		self.test_mask 			= self.mask[val_b:]
		self.test_labels 		= self.labels[val_b:]
		self.test_label_mask 	= self.label_mask[val_b:]



if __name__ == '__main__':
	data_loader = Data_Loader(128)
