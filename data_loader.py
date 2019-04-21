import pickle
import numpy as np 
from keras.preprocessing.sequence import pad_sequences
import torch
from gensim.models import Word2Vec

class Data_Loader():
	def __init__(self, batch_size, maxlen):
		self.batch_size = batch_size
		# self.sent_len = sent_len
		self.maxlen = maxlen
		fr = open('data.pkl', 'rb')
		data = pickle.load(fr)

		self.word2idx = data['word2idx']
		self.idx2word = data['idx2word']
		self.vocab_size = data['vocab_size']
		self.emb_size = data['emb_size']
		sentences = data['processed_sentence']
		labels = data['labels']
		
		sentences = pad_sequences(sentences,maxlen=36, padding='post')


		# print(labels)
		self.emb_mat = self.embed_mat()

		self.labels = np.array(pad_sequences(labels, maxlen=36, padding='post'))

		self.sent = np.array(sentences).astype(np.int32)

		self.mask = np.ones((len(self.sent), maxlen))

		self.mask[self.sent==0] = 0
		self.pointer = 0
		self.train_size = len(self.sent)


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
		temp = torch.from_numpy(self.labels[begin:end])
		# print(temp.dtype)
		# print(temp)
		return torch.tensor(self.sent[begin:end], dtype=torch.long), torch.from_numpy(self.mask[begin:end]), torch.tensor(self.labels[begin:end],dtype=torch.long)




		# lens = [len(sen) for sen in sentences]
		# print(max(lens), np.mean(lens))


if __name__ == '__main__':
	data_loader = Data_Loader(128, 36)