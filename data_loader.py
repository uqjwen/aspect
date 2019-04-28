import pickle
import numpy as np 
from keras.preprocessing.sequence import pad_sequences
#import torch
from gensim.models import Word2Vec
from keras.utils.np_utils import to_categorical

class Data_Loader():
	def __init__(self, batch_size):
		self.batch_size = batch_size
		# self.sent_len = sent_len
		# self.maxlen = maxlen
		fr = open('data.pkl', 'rb')
		data = pickle.load(fr)

		self.word2idx = data['word2idx']
		self.idx2word = data['idx2word']
		self.vocab_size = data['vocab_size']
		# self.emb_size = data['emb_size']

		self.label_mask = data['label_mask']


		self.num_tag = len(data['tag2idx'])+1
		tags = data['tags']



		self.emb_size = 100
		sentences = data['processed_sentence']
		labels = data['labels']

		# self.maxlen = max([len(sent) for sent in sentences])
		# self.maxlen = int(np.mean([len(sent) for sent in sentences]))
		self.maxlen = 36

		
		# sentences = pad_sequences(sentences,self.maxlen, padding='post')

		# tags = data['tags']
		# tags = pad_sequences(tags, self.maxlen, padding='post')



		# print(labels)
		self.emb_mat = self.embed_mat()

		self.labels = pad_sequences(labels, self.maxlen, padding='post')

		self.sent = pad_sequences(sentences, self.maxlen, padding='post')

		self.sent_tag = to_categorical(pad_sequences(tags, self.maxlen, padding='post'), self.num_tag)

		self.mask = np.ones((len(self.sent), self.maxlen))

		self.mask[self.sent==0] = 0
		self.pointer = 0
		self.train_size = len(self.sent)

		# print(self.train_size)


	def embed_mat(self):
		model = Word2Vec.load('my_gensim_model')
		mat = np.random.uniform(-1,1,(self.vocab_size, self.emb_size))
		for i in range(1,self.vocab_size):
			mat[i] =  model[self.idx2word[i]]
		return mat


	def reset_pointer(self):
		self.pointer = 0

	def __next__(self):
		begin = self.pointer*self.batch_size
		# end = min(self.train_size, (self.pointer+1)*self.batch_size)
		end = (self.pointer+1)*self.batch_size
		if (self.pointer+1)*self.batch_size > self.train_size:
			end = self.train_size
			self.pointer = 0

		self.pointer+=1
		# temp = torch.from_numpy(self.labels[begin:end])
		# print(temp.dtype)
		# print(temp)
		# return torch.tensor(self.sent[begin:end], dtype=torch.long), torch.from_numpy(self.mask[begin:end]), torch.tensor(self.labels[begin:end],dtype=torch.long)

		return self.sent[begin:end],self.sent_tag[begin:end], self.mask[begin:end], self.labels[begin:end], self.label_mask[begin:end]


	def val(self, sample_rate = 0.3):
		test_size = self.train_size - 2000
		sample_size = int(test_size*sample_rate)

		idx = np.random.choice(range(2001,self.train_size), sample_size, replace = False)

		return self.sent[idx], self.sent_tag[idx], self.mask[idx], self.labels[idx]
		# lens = [len(sen) for sen in sentences]
		# print(max(lens), np.mean(lens))


if __name__ == '__main__':
	data_loader = Data_Loader(128)
