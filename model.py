import sys
import torch 
import numpy as np 
import pickle 
import torch.nn.functional as F 
def get_domain_emb_weight(emb_size):
	fr = open("data.pkl",'rb')
	data = pickle.load(fr)

	word2idx = data['word2idx']

	gensim_model = gensim.models.Word2Vec.load("my_gensim_model")

	vocab_size = len(word2idx)+1

	emb_matrix = np.zeros((vocab_size, emb_size))

	for word in word2idx:
		idx = word2idx[word]
		emb_matrix[idx] = gensim_model[word]

	return emb_matrix


class Model(torch.nn.Module):
	def __init__(self,domain_emb, num_class = 3, drop_out = 0.5):
		super(Model, self).__init__()
		domain_vocab_size, domain_emb_size = domain_emb.shape
		self.domain_embedding = torch.nn.Embedding(domain_vocab_size, domain_emb_size)
		#self.domain_embedding.weight = torch.nn.Parameter(torch.tensor(domain_emb, dtype=torch.float32),requires_grad=False)
		self.domain_embedding.weight = torch.nn.Parameter(torch.tensor(domain_emb, dtype=torch.float32))

		self.conv1 = torch.nn.Conv1d(domain_emb.shape[1], 128, 5, padding = 2)
		self.conv2 = torch.nn.Conv1d(domain_emb.shape[1], 128, 3, padding = 1)

		self.dropout = torch.nn.Dropout(drop_out)

		self.conv3 = torch.nn.Conv1d(256, 256, 5, padding=2)

		self.conv4 = torch.nn.Conv1d(256, 256, 5, padding=2)

		self.conv5 = torch.nn.Conv1d(256, 256, 5, padding=2)
		self.linear_ae1 = torch.nn.Linear(256, 50)
		self.linear_ae2 = torch.nn.Linear(50, num_class)

	def forward(self, x, x_len, x_maxk, y=None, testing = False):
		x_emb = self.domain_embedding(x)    #[batch_size, x_len, emb_size]
		x_emb = self.dropout(x_emb).transpose(1,2) # [batch_size, emb_size, x_len]
		x_conv = torch.nn.functional.relu(torch.cat((self.conv1(x_emb), self.conv2(x_emb)),dim=1)) # [batch_size, 128+128, x_len]
		x_conv = self.dropout(x_conv)
		x_conv = torch.nn.functional.relu(self.conv3(x_conv)) #[batch_size, 256, x_len]
		x_conv = self.dropout(x_conv)
		x_conv = torch.nn.functional.relu(self.conv4(x_conv))
		x_conv = self.dropout(x_conv)
		x_conv = torch.nn.functional.relu(self.conv5(x_conv))
		x_conv = self.dropout(x_conv)  ##############drop out is very important, with it difficult to grad, but can change learning rate to mitigate this problem 

		x_conv = x_conv.transpose(1,2) #[batch_size, x_len, 256]

		x_logit = torch.nn.functional.relu(self.linear_ae1(x_conv)) #[batch_size, x_len, 50]

		self.x_logit = self.linear_ae2(x_logit) #[batch_size, x_len, 3]
		if testing:
			x_logit = self.x_logit.transpose(2,0) #[3, x_len, batch_size]
			score = torch.nn.functional.log_softmax(self.x_logit).transpose(2,0)
		else:
			# x_logit = torch.nn.utils.rnn.pack_padded_sequence(x_logit, x_len, batch_first=True)
			# score = torch.nn.functional.nnl_loss(torch.nn.functional.log_softmax(x_logit.data), y.data)
			# score = F.nll_loss(F.log_softmax(x_logit.transpose(1,2)), y)
			# score = F.nll_loss(F.log_softmax(self.x_logit.reshape(batch_size*x_len, 3),dim=1), y.reshape(batch_size*x_len))
			score = F.cross_entropy(self.x_logit.reshape(batch_size*x_len, 3), y.reshape(batch_size*x_len))

		return score 

if __name__ == '__main__':
	batch_size = 64
	vocab_size = 3000
	emb_size = 100
	max_len = 36

	domain_emb = np.random.uniform(0,1,(vocab_size, emb_size))
	model = Model(domain_emb, 3, 0.5)
	# optimizer = torch.nn.Adam(model.parameters)
	parameters = [p for p in model.parameters() if p.requires_grad is True]
	# optimizer = torch.optim.Adam(parameters, lr = 0.001)
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)###############learning rate is important 

	# input_data = torch.tensor(np.random.randint(0,vocab_size,(batch_size,max_len)), dtype=torch.int64)
	# mask_data = torch.tensor(np.random.randint(0,2,(batch_size, max_len)), dtype=torch.int64)
	# y_data = torch.tensor(np.random.randint(0,3,(batch_size, max_len)))


	input_data = torch.from_numpy(np.random.randint(0,vocab_size,(batch_size,max_len))).long()
	mask_data = torch.from_numpy(np.random.randint(0,2,(batch_size, max_len))).long()
	y_data = torch.from_numpy(np.random.randint(0,3,(batch_size, max_len)))


	for i in range(10000):
		loss = model(input_data, max_len, mask_data, y_data)

		# loss = model(torch.tensor(np.random.randint(0,2970, ())))
		# loss = model(input_data, )
		optimizer.zero_grad()

		loss.backward()
		torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
		optimizer.step()


		sys.stdout.write("\rloss:{},iteration:{}".format(loss, i))
		sys.stdout.flush()
		if (i+1)%100 == 0:

			# print(np.argmax(model.x_logit.detach().numpy(), axis=-1))
			# print(y_data)
			np.save('logit', model.x_logit.detach().numpy())
			np.save('y_data', y_data)
	print("\n");

