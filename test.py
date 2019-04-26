import sys
import numpy as np 
#import torch
#import torch.nn.functional as F
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Data_Loader():
# 	def __init__(self, batch_size = 64):
# 		self.num_user = 2000
# 		self.num_item = 4000
# 		self.batch_size = batch_size
# 		# rating_mat = np.zeros((self.num_user, self.num_item))
# 		rating_mat = [[1 if np.random.uniform(0,1)<0.05 else 0 for i in range(self.num_item) ] for j in range(self.num_user)]
# 		user = []
# 		item = []
# 		for i in range(self.num_user):
# 			for j in range(self.num_item):
# 				if rating_mat[i][j] == 1:
# 					user.append(i)
# 					item.append(j)
# 		self.user = np.array(user)
# 		self.item = np.array(item)

# 	def reset_pointer(self):
# 		self.pointer = 0

# 	def next(self):
# 		if self.pointer*self.batch_size>len(self.user):
# 			self.reset_pointer()

# 		begin = self.pointer*self.batch_size
# 		end = (self.pointer+1)*self.batch_size

# 		end = min(end, len(self.user))

# 		self.pointer+=1

# 		return self.user[begin:end], self.item[begin:end]

# class Model(torch.nn.Module):
# 	def __init__(self, num_user, num_item, emb_size, neg_size):
# 		super(Model, self).__init__()
# 		self.num_user = num_user
# 		self.num_item = num_item
# 		self.emb_size = emb_size
# 		self.neg_size = neg_size

# 		self.user_embedding = torch.nn.Embedding(num_user, emb_size)
# 		self.item_embedding = torch.nn.Embedding(num_item, emb_size)


# 		self.hidden_1 = torch.nn.Linear(2*emb_size+1, emb_size)

# 		self.hidden_2 = torch.nn.Linear(emb_size, int(emb_size/2))

# 		self.hidden_3 = torch.nn.Linear(int(emb_size/2), 1)

# 		self.neg_size = neg_size
# 	def neg_sample(self,user, item,num_item, neg_size):
# 		labels = [1]*len(user)+[0]*(len(user)*neg_size)
# 		new_user = list(user)
# 		new_item = list(item)

# 		for usr in user:
# 			neg_item = list(np.random.choice(num_item, neg_size))
# 			new_user.extend([usr]*neg_size)
# 			new_item.extend(neg_item)

# 		new_user = torch.tensor(new_user, dtype=torch.long)
# 		new_item = torch.tensor(new_item, dtype=torch.long)
# 		labels = torch.tensor(labels, dtype=torch.float)
# 		return new_user.to(device), new_item.to(device), labels.to(device)


# 	def forward(self,user,item):
# 		user,item,labels = self.neg_sample(user, item, self.num_item, self.neg_size)


# 		user_latent = self.user_embedding(user)
# 		item_latent = self.item_embedding(item)

# 		user_item_dot = torch.sum(user_latent*item_latent, dim=-1, keepdim=True)

# 		vec = torch.cat((user_latent, item_latent, user_item_dot), dim=-1)

# 		vec = self.hidden_1(vec)
# 		vec = F.relu(vec)
# 		vec = self.hidden_2(vec)
# 		vec = F.relu(vec)
# 		vec = self.hidden_3(vec)
# 		vec = F.sigmoid(vec)

# 		loss = F.binary_cross_entropy(vec,labels)
# 		return loss
		

# def main():
# 	data_loader = Data_Loader()

# 	model = Model(data_loader.num_user, data_loader.num_item, emb_size=64, neg_size = 4).to(device)

# 	epoches = 100
# 	iterations_per_epoch = 100

# 	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)###############learning rate is important 

# 	for e in range(epoches):
# 		data_loader.reset_pointer()
# 		for i in range(iterations_per_epoch):
# 			user,item = data_loader.next()
# 			loss = model(user,item)
# 			optimizer.zero_grad()

# 			loss.backward()
# 			torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
# 			optimizer.step()


# 			sys.stdout.write("\rloss:{},epoch:{}, iter:{}".format(loss, e, i))
# 			sys.stdout.flush()

##++==---------------------------------------------------------------------------------
# from keras.layers import Dense
# import tensorflow as tf 
# class Model():
# 	def __init__(self):
# 		emb_size = 100
# 		vocab_size = 3981
# 		batch_size = 32
# 		maxlen = 12	
# 		num_class = 3

# 		word_embedding = tf.Variable(tf.random_uniform([vocab_size, emb_size],-1.0,1.0))
# 		# linear_ae1 = Dense()

# 		self.x = tf.placeholder(tf.int32, shape=[batch_size, maxlen])
# 		self.label = tf.placeholder(tf.int32, shape = [batch_size, maxlen, num_class])

# 		emb = tf.nn.embedding_lookup(word_embedding, self.x)

# 		x_emb = tf.nn.dropout(emb, 0.5)

# 		x_logit = Dense(50, activation='relu', kernel_initializer = 'lecun_uniform')(x_emb)

# 		x_logit = Dense(3, kernel_initializer = 'lecun_uniform')(x_logit)

# 		loss = tf.nn.softmax_cross_entropy_with_logits(logits = x_logit, labels = self.label)

# 		self.cost = tf.reduce_mean(loss)

# 		self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)


# def main():
# 	model = Model()
# 	with tf.Session() as sess:
# 		sess.run(tf.global_variables_initializer())
# 		for i in range(100):
# 			for j in range(100):

# 				x = np.random.randint(0,3981,(32,12))
# 				y = np.random.randint(0,3,(32,12,3))
# 				loss,_ = sess.run([model.cost, model.train_op], feed_dict = {model.x:x,model.label:y})

# 				sys.stdout.write('\repoch:{}, batch:{}, loss:{}'.format(i,j,loss))
# 				sys.stdout.flush()


import pickle
def main():
	fr = open('data.pkl', 'rb')
	data = pickle.load(fr)

	# tags = data['tags']
	# print(len(tags))
	# print(tags)
	tag2idx = data['tag2idx']
	print(len(tag2idx))
	for key in tag2idx:
		print(key)


if __name__ == '__main__':
	main()
