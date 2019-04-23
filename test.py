import numpy as np 
import torch
import torch.nn.functional as F

class Data_Loader():
	def __init__(self, batch_size = 64):
		self.num_user = 2000
		self.num_item = 4000
		self.betch_size = batch_size
		# rating_mat = np.zeros((self.num_user, self.num_item))
		rating_mat = [[1 if np.random.uniform(0,1)<0.05 else 0 for i in range(self.num_item) ] for j in range(self.num_user)]
		user = []
		item = []
		for i in range(self.num_user):
			for j in range(self.num_item):
				if rating_mat[i][j] == 1:
					user.append(i)
					item.append(j)
		self.user = np.array(user)
		self.item = np.array(item)

	def reset_pointer(self):
		self.pointer = 0

	def next(self):
		if self.pointer*self.batch_size>len(self.user):
			self.reset_pointer()

		begin = self.pointer*self.batch_size
		end = (self.pointer+1)*self.batch_size

		end = min(end, len(self.user))

		self.pointer+=1

		return self.user[begin:end], self.item[begin:end]

class Model(torch.nn.Module):
	def __init__(self, num_user, num_item, emb_size, neg_size):
		super(Model, self).__init__()

		self.user_embedding = torch.nn.Embedding(num_user, emb_size)
		self.item_embedding = torch.nn.Embedding(num_item, emb_size)


		self.hidden_1 = torch.nn.Linear(2*emb_size+1, emb_size)

		self.hidden_2 = torch.nn.Linear(emb_size, int(emb_size/2))

		self.hidden_3 = torch.nn.Linear(int(emb_size/2), 1)

		self.neg_size = neg_size
	def neg_sample(self,user, item,num_item, neg_size):
		labels = [1]*len(user)+[0]*(len(user)*neg_size)
		new_user = list(user)
		new_item = list(item)

		for usr in user:
			neg_item = list(np.random.choice(num_item, neg_size))
			new_user.extend([usr]*neg_size)
			new_item.extend(neg_item)

		new_user = torch.tensor(new_user, dtype=torch.long)
		new_item = torch.tensor(new_item, dtype=torch.long)
		labels = torch.tensor(labels, dtype=torch.long)
		return new_user, new_item, labels


	def forward(self,user,item):
		user,item,labels = self.neg_sample(user, item, self.num_item, self.neg_size)


		user_latent = self.user_embedding(user)
		item_latent = self.item_embedding(item)

		user_item_dot = torch.sum(user_latent*item_latent, dim=-1, keepdim=True)

		vec = torch.cat((user_latent, item_latent, user_item_dot), dim=-1)

		




