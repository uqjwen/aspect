from data_loader import Data_Loader
from model import Model 
import torch
import sys


def train():
	batch_size = 128
	maxlen = 36
	data_loader = Data_Loader(batch_size, maxlen)
	model = Model(data_loader.emb_mat, num_class = 3, drop_out = 0.5)
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)###############learning rate is important 
	epochs = 100
	for i in range(epochs):
		data_loader.reset_pointer()
		num_batch = int(data_loader.train_size/batch_size)
		for b in range(num_batch+1):
			input_data, mask_data, y_data = data_loader.__next__()

			loss = model(input_data, maxlen, mask_data, y_data)

			optimizer.zero_grad()

			loss.backward()

			torch.nn.utils.clip_grad_norm(model.parameters(),1.)

			optimizer.step()

			sys.stdout.write("\repoch:{}, batch:{}, loss:{}".format(i,b,loss))



if __name__ == '__main__':
	train()