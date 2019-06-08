import os 
import sys
# import torch 
import numpy as np 
import pickle 
from keras.layers import Conv1D, Dropout, Dense
from keras.utils.np_utils import to_categorical
from data_loader import Data_Loader
import tensorflow as tf
from sklearn.metrics import f1_score
#b = to_categorical(a,9)





class Model():
	def __init__(self,gen_emb, domain_emb, num_class, num_tag, num_cat, maxlen,batch_size = 64, drop_out = 0.5, neg_size = 4):
		self.vocab_size, self.emb_size = domain_emb.shape
		self.maxlen 			= maxlen
		self.dropout 			= drop_out
		self.batch_size 		= batch_size
		self.neg_size 			= neg_size
		self.aspect_size 		= 12
		self.aspect_emb_size 	= self.emb_size
		self.num_cat 			= num_cat
		self.x 			= tf.placeholder(tf.int32, shape=[None, maxlen])
		self.labels 	= tf.placeholder(tf.int32, shape=[None, maxlen, num_class])
		self.clabels 	= tf.placeholder(tf.float32, shape=[None, num_cat])
		self.t 			= tf.placeholder(tf.float32, shape=[None, maxlen, num_tag])
		self.is_training= tf.placeholder(tf.bool)
		self.tfidf		= tf.placeholder(tf.float32, shape=[None, maxlen])

		self.mask 		= tf.placeholder(tf.float32, shape=[None, maxlen])

		self.label_mask = tf.placeholder(tf.float32, shape = [None])

		self.neg 		= tf.placeholder(tf.int32, shape=[None, maxlen, neg_size])

		self.word_embedding 	= tf.Variable(domain_emb.astype(np.float32))
		self.gen_embedding 		= tf.Variable(gen_emb.astype(np.float32))

		# self.aspect_embedding = tf.Variable(tf.random_uniform([self.aspect_size, self.emb_size],-1.0,1.0))
		self.aspect_embedding 	= tf.Variable(tf.random_uniform([self.aspect_size,self.aspect_emb_size],-1.0,1.0))
		# self.word_embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.emb_size], -1.0,1.0))

		# x_latent = self.get_x_latent(self.x)

		##---------------------------------------------------------
		latent,c_latent = self.get_latent(self.x, self.t)

		print('latent: ', latent)
		# latent = tf.concat([latent, tf.nn.embedding_lookup(self.word_embedding,self.x), self.t],axis=-1)
		# print(latent)

		#-----------------------------------------------------------------------------

		# att_score 	= Dense(self.aspect_size, activation = 'softmax') (latent)

		# att_aspect 	= tf.matmul(tf.reshape(att_score,[-1,self.aspect_size]), self.aspect_embedding)

		# att_aspect 	= tf.reshape(att_aspect, [-1, maxlen, self.aspect_emb_size])

		#-----------------------------------------------------------------------------

		# cat_latent 		= self.get_cat_attention(c_latent)
		# cat_latent 		= self.get_cat_maxpooling(c_latent)
		
		cat_latent 		= self.get_cnn_maxpool(self.x)
		self.cat_logits = Dense(self.num_cat, kernel_initializer='lecun_uniform')(cat_latent)

		cat_loss 		= tf.nn.sigmoid_cross_entropy_with_logits(logits = self.cat_logits, labels = self.clabels)
		cat_loss 		= tf.reduce_mean(cat_loss)
		# self.cat_pred = tf.argmax(cat_logits, axis=-1)

		#------------------------------------------------------------------------------



		x_logit 		= Dense(50, activation='relu', kernel_initializer = 'lecun_uniform')(latent)
		self.x_logit 	= Dense(3, kernel_initializer='lecun_uniform')(x_logit) #[batch_size, maxlen, 3]

		self.prediction = tf.argmax(self.x_logit, axis=-1)



		loss 		= tf.nn.softmax_cross_entropy_with_logits(logits = self.x_logit, labels = self.labels)

		loss 		= tf.reduce_mean(loss*self.mask) #[batch_size, maxlen]

		# label_mask 	= tf.reshape(self.label_mask, [-1,1]) #[batch_size,1]

		# self.loss 	= tf.reduce_sum(loss*label_mask)/tf.maximum(tf.reduce_sum(self.mask*label_mask), 1)






		# self.un_loss 		= self.get_un_loss(att_aspect, self.x, self.neg)

		# self.cost = self.loss# + self.un_loss

		# self.cost 			= cat_loss
		self.cost 			= loss + cat_loss


		self.global_step 	= tf.Variable(0, trainable = False)

		self.lr 			= tf.train.exponential_decay(0.0001, self.global_step, decay_steps=200, decay_rate=0.1)

		# self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)
		self.train_op 		= tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

		# optimizer 	= tf.train.AdamOptimizer(self.lr)
		# grads, vars = zip(*optimizer.compute_gradients(self.cost))
		# grads,_ 	= tf.clip_by_global_norm(grads, clip_norm = 2)
		# self.train_op = optimizer.apply_gradients(zip(grads,vars))


	def get_cat_maxpooling(self, latent):
		latent_pool = tf.layers.max_pooling1d(latent, pool_size = [self.maxlen], strides = 1)
		latent_pool = tf.squeeze(latent_pool, axis=1)
		return latent_pool

	def get_cat_attention(self, latent):
		scores = Dense(1, kernel_initializer = 'lecun_uniform')(latent) #batch_size, maxlen, 1
		scores = tf.squeeze(scores,-1)

		thres = tf.argmax(self.labels, axis=-1)
		thres = tf.cast(tf.greater(thres,0),tf.float32)

		exp_scores = self.mask*tf.exp(5*(scores+0.1*thres))
		self.sum_score = tf.reduce_sum(exp_scores, axis=-1, keepdims=True)

		softmax_scores = exp_scores/self.sum_score

		self.atts = softmax_scores
		softmax_scores = tf.expand_dims(scores,-1)
		cat_latent = tf.reduce_sum(softmax_scores*latent, axis=1) #batch_size, embed_size
		# return tf.reduce_mean(latent, axis=1)
		return cat_latent



	def get_cnn_maxpool(self,x):

		kernel_size = [3,4,5]
		domain_latent 	= tf.nn.embedding_lookup(self.word_embedding,x)
		gen_latent 		= tf.nn.embedding_lookup(self.gen_embedding, x)
		x_latent = tf.concat([domain_latent, gen_latent], axis=-1)
		res = []
		for size in kernel_size:
			conv = tf.layers.conv1d(x_latent, filters = 64, kernel_size=size, strides = 1)
			h = tf.nn.relu(conv)
			drop = tf.layers.dropout(h, rate = self.dropout, training = self.is_training)
			maxp = tf.layers.max_pooling1d(drop, pool_size=[self.maxlen-size+1], strides=1)

			res.append(tf.squeeze(maxp,axis=1))
		return tf.concat(res,axis=-1)



	def get_cnn(self,x):

		conv1 = Conv1D(128, kernel_size = 3, padding = 'same')

		conv2 = Conv1D(128, kernel_size = 5, padding = 'same')

		conv3 = Conv1D(128, kernel_size = 5, padding = 'same')

		x = tf.nn.relu(tf.concat([conv1(x), conv2(x)], axis=-1))
		x = tf.layers.dropout(x, rate = self.dropout, training = self.is_training)

		x = tf.nn.relu(conv3(x))
		x = tf.layers.dropout(x, rate = self.dropout, training = self.is_training)

		return x
		# x = tf.cond(self.is_training)

		# conv2 = conv

	def get_lstm(self,x):
		lstm1 = tf.keras.layers.LSTM(128, return_sequences = True)

		lstm2 = tf.keras.layers.LSTM(128, return_sequences = True)

		x = lstm1(x)
		x = tf.layers.dropout(x, rate = self.dropout, training = self.is_training)
		x = lstm2(x)
		x = tf.layers.dropout(x, rate = self.dropout, training = self.is_training)

		return x


	def get_latent(self, x, t):
		domain_latent 	= tf.nn.embedding_lookup(self.word_embedding, x)
		gen_latent 		= tf.nn.embedding_lookup(self.gen_embedding, x)

		x_latent = tf.concat([domain_latent, gen_latent], axis=-1)
		# x_latent = gen_latent

		# return x_latent
		# x_latent = tf.nn.relu(tf.concat([self.x_conv_1(x_latent), self.x_conv_2(x_latent)],axis=-1))
		# x_latent = tf.layers.dropout(x_latent, rate=self.dropout, training = self.is_training)


		# x_latent = tf.nn.relu(self.x_conv_3(x_latent))
		# x_latent = tf.layers.dropout(x_latent, rate=self.dropout, training = self.is_training)


		x_latent = self.get_cnn(x_latent)

		t_latent = self.get_cnn(t)
		# x_latent = self.get_lstm(x_latent)


		# t_latent = tf.nn.relu(self.t_conv_1(t))
		# t_latent = tf.cond(self.is_training, lambda:tf.nn.dropout(t_latent, self.dropout), lambda:t_latent)

		

		# t_latent = tf.nn.relu(self.t_conv_2(t_latent))
		# t_latent = tf.cond(self.is_training, lambda:tf.nn.dropout(t_latent, self.dropout), lambda:t_latent)

		# t_latent = tf.nn.relu(self.t_conv_3(t_latent))
		# t_latent = tf.cond(self.is_training, lambda:tf.nn.dropout(t_latent, self.dropout), lambda:t_latent)


		gate = tf.nn.sigmoid(Dense(128, use_bias = True)(x_latent)+Dense(128)(t_latent))

		latent = gate*t_latent+(1-gate)*x_latent



		# t_latent = tf.nn.dropout(t_latent, self.dropout)

		# latent = tf.concat([x_latent, t_latent], axis=-1)


		return latent,x_latent
		return x_latent
	def get_un_loss(self,att_aspect, x, neg_x):
		batch_size, maxlen, emb_size = att_aspect.shape.as_list()
		x_latent = tf.nn.embedding_lookup(self.word_embedding, x)

		pos = tf.reduce_sum(att_aspect*x_latent, axis=-1, keepdims=True) #[batch_size, maxlen, 1]

		neg_x_latent = tf.nn.embedding_lookup(self.word_embedding, neg_x)#[batch_size, maxlen, neg_size, emb_size]

		neg = tf.reduce_sum(tf.expand_dims(att_aspect, 2)*neg_x_latent,axis=-1) # [batch_size, maxlen, neg_size]


		un_loss = tf.maximum(0., 1.-pos+neg) #[batch_size, maxlen, neg_size]

		# un_loss = tf.expand_dims(self.mask, -1)
		new_mask = tf.expand_dims(self.mask, -1)

		un_loss = un_loss*new_mask

		un_loss = tf.reduce_sum(un_loss) / tf.reduce_sum(new_mask) / self.neg_size

		return un_loss





def f_score(y_pred, y_true, y_mask):
	y_pred = y_pred.reshape(-1)
	y_true = y_true.reshape(-1)
	y_mask = y_mask.reshape(-1)

	index = np.where(y_mask==1)[0]

	return f1_score(y_pred[index], y_true[index], average='macro')


def res(idx2word,input_data, y_pred, y_true, mask_data, x_logit):
	for i,line in enumerate(input_data):
		mask_index = np.where(mask_data[i]==1)[0]
		index = line[mask_index]
		# print(line, mask_index)
		tokens = [idx2word[idx] for idx in index]


		# for t,yt,yp,xl in zip(tokens, y_true[i][mask_index], y_pred[i][mask_index], x_logit[i][mask_index]):
		# 	print(t,'-----',yt,'-----',yp,'-----',xl)
		
		# sent = '\t'.join(tokens)
		# labels = '\t'.join(map(str,y_true[i][mask_index]))
		# predict = '\t'.join(map(str,y_pred[i][mask_index]))
		# print(sent)
		# print(labels)
		# print(predict)
		# print('-------------------------------------')
def cat_metrics(clabels, clogits):
	y_true = []
	y_pred = []


	res_pred = []
	for clabel, clogit in zip(clabels, clogits):
		# if cmask == 0:
			# continue
		labels = np.where(clabel!=0)[0]
		# num = min(5,len(labels))
		num = len(labels)
		logits = list(np.argsort(clogit)[::-1][:num])
		res_pred.append(logits)
		for label in labels:
			y_true.append(label)
			if label in logits:
				y_pred.append(label)
				# logits.remove(label)
			else:
				# y_pred.append(logits.pop(-1))
				y_pred.append(logits[0])
				logits.append(logits.pop(0))
		#-------------------------------------------------------
		# labels = np.where(clabel !=0)[0]
		# num = len(labels)
		# logits = np.argsort(clogit)[::-1][:num]

		# flag = 0
		# for logit in logits:
		# 	if logit in labels:
		# 		y_true.append(logit)
		# 		y_pred.append(logit)
		# 		flag = 1
		# if flag == 0:
		# 	y_true.append(np.random.choice(labels))
		# 	y_pred.append(logits[0])
		#------------------------------------------------------


	return f1_score(y_true, y_pred, average = 'micro'), res_pred

def val(sess, model, data_loader):
	input_data, input_tag, mask_data, y_data, clabels, index, tfidf= data_loader.val(1)

	y_data = to_categorical(y_data, 3)
	x_logit, y_pred, cat_logits = sess.run([model.x_logit, model.prediction,model.cat_logits],
								feed_dict = {model.x:input_data,
											model.t:input_tag,
											model.mask:mask_data,
											model.labels:y_data,
											model.clabels:clabels,
											model.is_training:False,
											model.tfidf:tfidf})
	# f_score = f1_score()




	# clabels = np.argmax(clabels, axis=-1)
	y_true = np.argmax(y_data,axis=-1)
	fscore_1 = f_score(y_pred, y_true, mask_data)
	# fscore = f1_score(cat_pred, clabels, average = 'micro')
	fscore_2,_ = cat_metrics(clabels, cat_logits)
	# if fscore>0.5:
	# 	res(data_loader.idx2word, input_data, y_pred, y_true, mask_data, x_logit)
	return fscore_1,fscore_2





def train():
	batch_size = 32
	neg_size = 4
	domain = sys.argv[1]
	data_loader = Data_Loader(batch_size, domain)
	maxlen = data_loader.maxlen
	model = Model(data_loader.gen_mat,
				data_loader.emb_mat,
				num_tag = data_loader.num_tag,
				num_cat = data_loader.num_cat,
				num_class = 3,
				maxlen = maxlen,
				batch_size = batch_size,
				drop_out = 0.5,
				neg_size = neg_size)
	epochs 		= 200
	best_1 = 0; best_2 = 0
	train_loss 	= []
	val_score 	= []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# saver = tf.train.Saver(tf.global_variables())
		saver = tf.train.Saver(max_to_keep=10)


		ckpt = tf.train.get_checkpoint_state(checkpointer_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print(" [*] loading parameters success!!!")
		else:
			print(" [!] loading parameters failed...")

		for i in range(epochs):
			data_loader.reset_pointer()
			num_batch = int(data_loader.train_size/batch_size)
			# print("total batch: ", num_batch)
			for b in range(num_batch+1):
				input_data, input_tag, mask_data, y_data, clabels, tfidf = data_loader.__next__()
				# print(input_data.shape, input_tag.shape, mask_data.shape, y_data.shape, label_mask.shape)
				# print(input_data.shape, mask_data.shape, y_data.shape)
				input_neg = np.random.randint(1,data_loader.vocab_size, (input_data.shape[0], maxlen, neg_size))
				# print(input_neg)
				y_data = to_categorical(y_data, 3)
				# print(y_data.shape,'uqjwen')
				_,loss = sess.run([model.train_op, model.cost], feed_dict = {model.x:input_data,
																			model.t:input_tag,
																			model.mask:mask_data,
																			model.neg:input_neg,
																			model.labels:y_data,
																			model.clabels:clabels,
																			model.is_training:True,
																			model.tfidf:tfidf})

				sys.stdout.write('\repoch:{}, batch:{}, loss:{}'.format(i,b,loss))
				sys.stdout.flush()
				# if loss is np.nan:
			# print('sum_score',sum_score,'\nd1',d1,'\nd2',d2)

				# break
			# print("validation....")
			lr = sess.run(model.lr, feed_dict = {model.global_step:i})
			print('\t learning_rate: ',lr)
			fscore_1, fscore_2 = val(sess, model, data_loader)
			if fscore_1 > best_1:
				best_1 = fscore_1
				saver.save(sess, checkpointer_dir+'model.ckpt', global_step=i)
			if fscore_2 > best_2:
				best_2 = fscore_2
				saver.save(sess, checkpointer_dir+'model.ckpt', global_step=i)


			print("\nfscore_1: ", fscore_1, "fscore_2: ", fscore_2)
			# break
			train_loss.append(loss)
			val_score.append([fscore_1, fscore_2])
		np.save(checkpointer_dir+'train_loss', train_loss)
		np.save(checkpointer_dir+'val_score', val_score)



def save_for_visual(sents, masks, y_pred, atts, clogits, clabels, data_loader, index, cat_pred):
	fr = open(checkpointer_dir+'visual.txt', 'w')
	idx2word = data_loader.idx2word
	idx2clabel = data_loader.idx2clabel
	for sent, mask, att, clabel, idx, cpred in zip(sents, masks, atts, clabels, index, cat_pred):
		sen = [idx2word[item] for i,item in enumerate(sent) if mask[i]==1]
		psent = data_loader.psent[idx]
		psen = [idx2word[item] for item in psent]
		att = [item for i,item in enumerate(att) if mask[i] == 1]
		att = np.round(np.array(att),3)
		labels = [idx2clabel[i] for i,label in enumerate(clabel) if label!=0]
		pre_labels = [idx2clabel[i] for i in cpred]
		# print(clabel)
		fr.write('\t'.join(psen)+'\n')
		fr.write('\t'.join(sen)+'\n')
		fr.write('\t'.join(map(str,att))+'\n')
		fr.write('\t'.join(labels)+'\n')
		fr.write('\t'.join(pre_labels)+'\n')
		fr.write('\t'.join(map(str,clabel))+'\n')

		fr.write('------------------------------------------\n')
	fr.close()

def test_debug(data_loader, input_data, clabels, cat_logits):
	idx2word = data_loader.idx2word
	idx2clabel = data_loader.idx2clabel

	for sent, clabel, cat_logit in zip(input_data, clabels, cat_logits):
		sent = [idx2word[idx] for idx in sent if idx!=0]
		print(' '.join(sent))
		clabel = np.where(clabel!=0)[0]
		clabel = [idx2clabel[c] for c in clabel]
		print(' '.join(clabel))

		num = len(clabel)
		cat_logit = np.argsort(cat_logit)[::-1][:num]
		cat_logit = [idx2clabel[c] for c in cat_logit]
		print(' '.join(cat_logit))
		print('--------------------------------------------------')
	




def test():

	batch_size = 32
	neg_size = 4
	domain = sys.argv[1]
	data_loader = Data_Loader(batch_size, domain)
	maxlen = data_loader.maxlen
	model = Model(data_loader.gen_mat,
				data_loader.emb_mat,
				num_tag = data_loader.num_tag,
				num_cat = data_loader.num_cat,
				num_class = 3,
				maxlen = maxlen,
				batch_size = batch_size,
				drop_out = 0.5,
				neg_size = neg_size)
	iterations = 10
	res = []
	with tf.Session() as sess:
		# sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())


		ckpt = tf.train.get_checkpoint_state(checkpointer_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print(" [*] loading parameters success!!!")
		else:
			print(" [!] loading parameters failed...")
			return




		for i in range(iterations):

			input_data, input_tag, mask_data, y_data, clabels, index, tfidf = data_loader.val(0.9)
			y_data = to_categorical(y_data, 3)
			

			x_logit, y_pred, cat_logits = sess.run([model.x_logit, model.prediction,model.cat_logits],
										feed_dict = {model.x:input_data,
													model.t:input_tag,
													model.mask:mask_data,
													model.labels:y_data,
													model.clabels:clabels,
													model.is_training:False,
													model.tfidf:tfidf})
			# f_score = f1_score()




			# clabels = np.argmax(clabels, axis=-1)
			y_true = np.argmax(y_data,axis=-1)
			fscore_1 = f_score(y_pred, y_true, mask_data)
			# fscore = f1_score(cat_pred, clabels, average = 'micro')
			# fscore = cat_metrics(input_data, mask_data, clabels, cat_logits, clabel_mask)
			fscore_2, cat_pred = cat_metrics(clabels, cat_logits)
			print(fscore_1, fscore_2)
			res.append([fscore_1, fscore_2])
			# print(fscore)
		res = np.array(res)
		# print(np.mean(res), np.var(res))
		print(np.mean(res,axis=0))
		print(np.var(res, axis=0))
		# save_for_visual(input_data, mask_data, y_pred, atts, cat_logits, clabels, data_loader, index, cat_pred)
		np.save(checkpointer_dir+'res', res)
		# test_debug(data_loader, input_data, clabels, cat_logits)




checkpointer_dir = './ckpt/'
checkpointer_dir = './ckpt_'+sys.argv[1]+'/'
if not os.path.exists(checkpointer_dir):
	os.makedirs(checkpointer_dir)





if __name__ == '__main__':
	if sys.argv[2] == 'train':
		train()
	else:
		test()
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

