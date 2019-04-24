import sys
# import torch 
import numpy as np 
import pickle 
from keras.layers import Conv1D, Dropout

# import torch.nn.functional as F 

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model():
	def __init__(self, domain_emb, num_class, maxlen, drop_out = 0.5):
		self.vocab_size, self.emb_size = domain_emb.shape
		self.maxlen = maxlen

		self.word_embedding = tf.Variable(domain_emb.astype(np.float32))

		self.conv1 = Conv1D(128, kernel_size = 3, padding = 'same')
		self.conv2 = Conv1D(128, kernel_size = 5, padding = 'same')

		self.dropout = Dropout(drop_out)

		self.conv3 = Conv1D(256, kernel_size = 5, padding='same')
		self.conv4 = Conv1D(256, kernel_size = 5, padding = 'same')
		self.conv5 = Conv1D(256, kernel_size = 5, padding='same')

		self.linear_ae1 = Dense(50)

		self.linear_ae2 = Dense(3)


	def forward(self, num_class):
		self.x = tf.placeholder(tf.int32, shape=[None, self.maxlen])

		self.labels = tf.placeholder(tf.int32, shape = [None, self.maxlen, num_class])

		x_emb = tf.nn.embedding_lookup(self.word_embedding, self.x)

		x_emb = self.dropout(x_emb)

		x_conv = tf.nn.relu(tf.concat([self.conv1(x_emb), self.conv2(x_emb)], axis=-1))

		x_conv = self.dropout(x_conv)

		x_conv = tf.nn.relu(self.conv3(x_conv))
		x_conv = self.dropout(x_conv)
		x_conv = tf.nn.relu(self.conv4(x_conv))
		x_conv = self.dropout(x_conv)
		x_conv = tf.nn.relu(self.conv5(x_conv))
		x_conv = self.dropout(x_conv)

		x_logit = tf.nn.relu(self.linear_ae1(x_conv))

		x_logit = self.linear_ae2(x_logit)


		score = tf.nn.softmax_cross_entropy_with_logits(labels = self.labels, logits = self.x_logit)


# # if __name__ == '__main__':
# 	batch_size = 64
# 	vocab_size = 3000
# 	emb_size = 100
# 	max_len = 36

# 	domain_emb = np.random.uniform(0,1,(vocab_size, emb_size))
# 	model = Model(domain_emb, 3, 0.5)
# 	# optimizer = torch.nn.Adam(model.parameters)
# 	parameters = [p for p in model.parameters() if p.requires_grad is True]
# 	# optimizer = torch.optim.Adam(parameters, lr = 0.001)
# 	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)###############learning rate is important 



# 	input_data = torch.from_numpy(np.random.randint(0,vocab_size,(batch_size,max_len))).long()
# 	mask_data = torch.from_numpy(np.random.randint(0,2,(batch_size, max_len))).long()
# 	y_data = torch.from_numpy(np.random.randint(0,3,(batch_size, max_len)))


# 	for i in range(10000):
# 		loss = model(input_data, max_len, mask_data, y_data)

# 		# loss = model(torch.tensor(np.random.randint(0,2970, ())))
# 		# loss = model(input_data, )
# 		optimizer.zero_grad()

# 		loss.backward()
# 		torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
# 		optimizer.step()


# 		sys.stdout.write("\rloss:{},iteration:{}".format(loss, i))
# 		sys.stdout.flush()
# 		if (i+1)%100 == 0:

# 			# print(np.argmax(model.x_logit.detach().numpy(), axis=-1))
# 			# print(y_data)
# 			np.save('logit', model.x_logit.detach().numpy())
# 			np.save('y_data', y_data)
# 	print("\n");

