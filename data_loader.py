import pickle
import numpy as np 
from keras.preprocessing.sequence import pad_sequences
import torch

class Data_Loader():
	def __init__(self, batch_size, sent_len):
		self.batch_size = batch_size
		self.sent_len = sent_len
		fr = open('data.pkl', 'rb')
		data = pickle.load(fr)

		word2idx = data['word2idx']
		sentences = data['processed_sentence']
		labels = data['labels']
		
		sentences = pad_sequences(sentences,maxlen=36, padding='post')


		print(labels)

		self.labels = np.array(pad_sequences(labels, maxlen=36, padding='post'))

		self.sent = np.array(sentences).astype(np.int32)

		self.mask = np.ones((len(self.sent), maxlen))

		self.mask[self.sent==0] = 0
		self.pointer = 0
		self.train_size = len(self.sent)

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

		return troch.tensor(self.sent[begin:end], dtype=torch.long), torch.from_numpy(self.labels[begin:end]), torch.from_numpy(self.mask[begin:end])




		# lens = [len(sen) for sen in sentences]
		# print(max(lens), np.mean(lens))


if __name__ == '__main__':
	data_loader = Data_Loader(128, 36)