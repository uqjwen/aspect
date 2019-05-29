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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
		print("embedding size", domain_emb.shape)
		self.x 			= tf.placeholder(tf.int32, shape=[None, maxlen])
		self.labels 	= tf.placeholder(tf.int32, shape=[None, maxlen, num_class])
		self.clabels 	= tf.placeholder(tf.float32, shape=[None, num_cat])
		self.t 			= tf.placeholder(tf.float32, shape=[None, maxlen, num_tag])
		self.is_training= tf.placeholder(tf.bool)


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
		self.x_conv_1 = Conv1D(128, kernel_size = 3, padding = 'same')

		self.x_conv_2 = Conv1D(128, kernel_size = 5, padding = 'same')

		self.x_conv_3 = Conv1D(128, kernel_size = 5, padding = 'same')

		self.x_conv_4 = Conv1D(128, kernel_size = 5, padding = 'same')

		self.x_conv_5 = Conv1D(128, kernel_size = 5, padding = 'same')
		#conv2 = Conv1D(128, kernel_size = 5, padding = 'same')

		#conv = tf.nn.relu(tf.concat([conv1(x_emb), conv2(x_emb)], axis=-1))
		#conv = tf.nn.dropout(conv, self.dropout)

		#conv3 = Conv1D(256, kernel_size = 5, padding = 'same')
		#conv = tf.nn.relu(conv3(conv))
		#conv = tf.nn.dropout(conv, self.dropout)

		#conv4 = Conv1D(256, kernel_size = 5, padding = 'same')
		#conv = tf.nn.relu(conv4(conv))
		#conv = tf.nn.dropout(conv, self.dropout)

		#conv5 = Conv1D(256, kernel_size = 5, padding = 'same')
		#conv = tf.nn.relu(conv5(conv))
		#x_emb = tf.nn.dropout(conv, self.dropout)
		##-------------------------------------------------------

		self.t_conv_1 = Conv1D(128, kernel_size = 5, padding = 'same')

		self.t_conv_2 = Conv1D(128, kernel_size = 5, padding = 'same')

		self.t_conv_3 = Conv1D(128, kernel_size = 5, padding = 'same')

		# t_latent = tf.nn.relu(t_conv_1(self.t))

		# t_latent = tf.nn.dropout(t_latent, self.dropout)

		# latent = tf.concat([x_latent, t_latent], axis=-1)

		latent = self.get_latent(self.x, self.t)

		# latent = tf.concat([latent, tf.nn.embedding_lookup(self.word_embedding,self.x), self.t],axis=-1)
		# print(latent)

		#-----------------------------------------------------------------------------

		att_score = Dense(self.aspect_size, activation = 'softmax') (latent)

		att_aspect = tf.matmul(tf.reshape(att_score,[-1,self.aspect_size]), self.aspect_embedding)

		att_aspect = tf.reshape(att_aspect, [-1, maxlen, self.aspect_emb_size])

		#-----------------------------------------------------------------------------

		cat_latent = self.get_cat_latent(latent)
		self.cat_logits = Dense(self.num_cat, kernel_initializer='lecun_uniform')(cat_latent)

		cat_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.cat_logits, labels = self.clabels)
		cat_loss = tf.reduce_mean(cat_loss)
		# self.cat_pred = tf.argmax(cat_logits, axis=-1)

		#------------------------------------------------------------------------------



		x_logit = Dense(50, activation='relu', kernel_initializer = 'lecun_uniform')(latent)
		self.x_logit = Dense(3, kernel_initializer='lecun_uniform')(x_logit) #[batch_size, maxlen, 3]

		self.prediction = tf.argmax(self.x_logit, axis=-1)
		groundtruth = tf.argmax(self.labels, axis=-1)
		truepositive = tf.cast(tf.equal(self.prediction, groundtruth),tf.float32)

		self.accuracy_1 = tf.reduce_sum(truepositive*self.mask)/tf.reduce_sum(self.mask)

		self.accuracy_2 = tf.reduce_mean(truepositive)


		# self.linear_ae1 = Dense(50, activation='relu', kernel_initializer = 'lecun_uniform')

		# self.linear_ae2 = Dense(3, kernel_initializer='lecun_uniform')

		# self.logits, self.loss = self.forward(num_class)

		loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.x_logit, labels = self.labels)

		loss = loss*self.mask #[batch_size, maxlen]

		label_mask = tf.reshape(self.label_mask, [-1,1]) #[batch_size,1]

		self.loss = tf.reduce_sum(loss*label_mask)/tf.maximum(tf.reduce_sum(self.mask*label_mask), 1)
		# self.loss = tf.reduce_sum(loss)/tf.reduce_sum(self.mask)


		# self.loss = tf.reduce_sum(loss)/tf.reduce_sum(self.mask)

		# self.cost = tf.reduce_mean(loss)
		# self.cost = loss 

		# self.cost += un_loss
		self.un_loss = self.get_un_loss(att_aspect, self.x, self.neg)

		# self.cost = self.loss# + self.un_loss

		self.cost = cat_loss


		self.global_step = tf.Variable(0, trainable = False)

		self.lr = tf.train.exponential_decay(0.0001, self.global_step, decay_steps=200, decay_rate=0.1)

		# self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

		# optimizer 	= tf.train.AdamOptimizer(self.lr)
		# grads, vars = zip(*optimizer.compute_gradients(self.cost))
		# grads,_ 	= tf.clip_by_global_norm(grads, clip_norm = 2)
		# self.train_op = optimizer.apply_gradients(zip(grads,vars))

	def get_cat_latent(self, latent):
		scores = Dense(1, kernel_initializer = 'lecun_uniform')(latent) #batch_size, maxlen, 1
		scores = tf.squeeze(scores,-1)

		# self.debug = tf.reduce_sum(self.mask*tf.exp(scores), axis=-1, keepdims = True)


		#latent: batch_size, maxlen, embed_size
		#score: batch_size, maxlen
		#mask: batch_size, maxlen
		self.d1 = scores 
		exp_scores = self.mask*tf.exp(10*scores)
		self.d2 = exp_scores
		self.sum_score = tf.reduce_sum(exp_scores, axis=-1, keepdims=True)
		# sum_score = tf.maximum(sum_score,1)
		self.sum_score += 1e-10
		softmax_scores = exp_scores/self.sum_score
		self.atts = softmax_scores
		softmax_scores = tf.expand_dims(scores,-1)



		# scores = tf.nn.softmax(scores)
		# scores = self.mask*scores
		# self.atts = scores
		# scores = tf.expand_dims(scores, -1)

		cat_latent = tf.reduce_sum(softmax_scores*latent, axis=1) #batch_size, embed_size
		print(cat_latent.shape,'cat_latent')
		return cat_latent






	def get_latent(self, x, t):
		domain_latent = tf.nn.embedding_lookup(self.word_embedding, x)
		gen_latent = tf.nn.embedding_lookup(self.gen_embedding, x)

		x_latent = tf.concat([domain_latent, gen_latent], axis=-1)


		# x_latent = tf.nn.dropout(tf.nn.relu(self.x_conv_1(x_latent)), self.dropout)
		x_latent = tf.nn.relu(tf.concat([self.x_conv_1(x_latent), self.x_conv_2(x_latent)],axis=-1))

		x_latent = tf.cond(self.is_training, lambda:tf.nn.dropout(x_latent, self.dropout), lambda: x_latent)

		x_latent = tf.nn.relu(self.x_conv_3(x_latent))

		x_latent = tf.cond(self.is_training, lambda:tf.nn.dropout(x_latent, self.dropout), lambda: x_latent)

		# x_latent = tf.nn.dropout(tf.nn.relu(self.x_conv_4(x_latent)), self.dropout)

		# x_latent = tf.nn.dropout(tf.nn.relu(self.x_conv_5(x_latent)), self.dropout)



		t_latent = tf.nn.relu(self.t_conv_1(t))
		t_latent = tf.cond(self.is_training, lambda:tf.nn.dropout(t_latent, self.dropout), lambda:t_latent)

		

		t_latent = tf.nn.relu(self.t_conv_2(t_latent))
		t_latent = tf.cond(self.is_training, lambda:tf.nn.dropout(t_latent, self.dropout), lambda:t_latent)

		t_latent = tf.nn.relu(self.t_conv_3(t_latent))
		t_latent = tf.cond(self.is_training, lambda:tf.nn.dropout(t_latent, self.dropout), lambda:t_latent)


		gate = tf.nn.sigmoid(Dense(128, use_bias = True)(x_latent)+Dense(128)(t_latent))

		latent = gate*t_latent+(1-gate)*x_latent



		# t_latent = tf.nn.dropout(t_latent, self.dropout)

		# latent = tf.concat([x_latent, t_latent], axis=-1)


		# return latent
		return x_latent
	def get_un_loss(self,att_aspect, x, neg_x):
		batch_size, maxlen, emb_size = att_aspect.shape.as_list()
		x_latent = tf.nn.embedding_lookup(self.word_embedding, x)

		pos = tf.reduce_sum(att_aspect*x_latent, axis=-1, keep_dims=True) #[batch_size, maxlen, 1]

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
def cat_metrics(clabels, clogits, clabel_mask):
	y_true = []
	y_pred = []

	for clabel, clogit, cmask in zip(clabels, clogits, clabel_mask):
		if cmask == 0:
			continue
		labels = np.where(clabel!=0)[0]
		# num = min(5,len(labels))
		num = len(labels)
		logits = list(np.argsort(clogit)[::-1][:num])
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


	return f1_score(y_true, y_pred, average = 'micro')

def val(sess, model, data_loader):
	input_data, input_tag, mask_data, y_data, clabels, clabel_mask = data_loader.val(1)

	y_data = to_categorical(y_data, 3)
	x_logit, y_pred, cat_logits = sess.run([model.x_logit, model.prediction,model.cat_logits],
								feed_dict = {model.x:input_data,
											model.t:input_tag,
											model.mask:mask_data,
											model.labels:y_data,
											model.clabels:clabels,
											model.is_training:False})
	# f_score = f1_score()




	# clabels = np.argmax(clabels, axis=-1)
	y_true = np.argmax(y_data,axis=-1)
	# fscore = f_score(y_pred, y_true, mask_data)
	# fscore = f1_score(cat_pred, clabels, average = 'micro')
	fscore = cat_metrics(clabels, cat_logits, clabel_mask)
	# if fscore>0.5:
	# 	res(data_loader.idx2word, input_data, y_pred, y_true, mask_data, x_logit)
	return fscore





def train():
	batch_size = 32
	neg_size = 4
	data_set = sys.argv[1]
	data_loader = Data_Loader(batch_size, data_set)
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
	best_metric = 0
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
				input_data, input_tag, mask_data, y_data, label_mask, clabels = data_loader.__next__()
				# print(input_data.shape, input_tag.shape, mask_data.shape, y_data.shape, label_mask.shape)
				# print(input_data.shape, mask_data.shape, y_data.shape)
				input_neg = np.random.randint(1,data_loader.vocab_size, (input_data.shape[0], maxlen, neg_size))
				# print(input_neg)
				y_data = to_categorical(y_data, 3)
				# print(y_data.shape,'uqjwen')
				_,loss,sum_score,d1,d2 = sess.run([model.train_op, model.cost, model.sum_score, model.d1, model.d2], feed_dict = {model.x:input_data,
																			model.t:input_tag,
																			model.mask:mask_data,
																			model.label_mask:label_mask,
																			model.neg:input_neg,
																			model.labels:y_data,
																			model.clabels:clabels,
																			model.is_training:True})

				sys.stdout.write('\repoch:{}, batch:{}, loss:{}'.format(i,b,loss))
				sys.stdout.flush()
				# if loss is np.nan:
			# print('sum_score',sum_score,'\nd1',d1,'\nd2',d2)

				# break
			# print("validation....")
			lr = sess.run(model.lr, feed_dict = {model.global_step:i})
			print('\t learning_rate: ',lr)
			fscore = val(sess, model, data_loader)
			if fscore > best_metric:
				best_metric = fscore
				saver.save(sess, checkpointer_dir+'model.ckpt', global_step=i)
			print("\nf1_score: ", fscore)
			# break
			train_loss.append(loss)
			val_score.append(fscore)
		np.save(checkpointer_dir+'train_loss', train_loss)
		np.save(checkpointer_dir+'val_score', val_score)



def save_for_visual(sents, masks, y_pred, atts, clogits, clabels, data_loader):
	fr = open(checkpointer_dir+'visual.txt', 'w')
	idx2word = data_loader.idx2word
	idx2clabel = data_loader.idx2clabel
	for sent, mask, att, clabel in zip(sents, masks, atts, clabels):
		sen = [idx2word[item] for i,item in enumerate(sent) if mask[i]==1]
		att = [item for i,item in enumerate(att) if mask[i] == 1]
		att = np.round(np.array(att),3)
		labels = [idx2clabel[i] for i,label in enumerate(clabel) if label!=0]
		# print(clabel)
		fr.write('\t'.join(sen)+'\n')
		fr.write('\t'.join(map(str,att))+'\n')
		fr.write('\t'.join(labels)+'\n')
	fr.close()

def test():

	batch_size = 32
	neg_size = 4
	data_set = sys.argv[1]
	data_loader = Data_Loader(batch_size, data_set)
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

			input_data, input_tag, mask_data, y_data, clabels, clabel_mask = data_loader.val(0.9)
			y_data = to_categorical(y_data, 3)
			

			x_logit, y_pred, cat_logits, atts = sess.run([model.x_logit, model.prediction,model.cat_logits, model.atts],
										feed_dict = {model.x:input_data,
													model.t:input_tag,
													model.mask:mask_data,
													model.labels:y_data,
													model.clabels:clabels,
													model.is_training:False})
			# f_score = f1_score()




			# clabels = np.argmax(clabels, axis=-1)
			y_true = np.argmax(y_data,axis=-1)
			# fscore = f_score(y_pred, y_true, mask_data)
			# fscore = f1_score(cat_pred, clabels, average = 'micro')
			# fscore = cat_metrics(input_data, mask_data, clabels, cat_logits, clabel_mask)
			fscore = cat_metrics(clabels, cat_logits, clabel_mask)
			print(fscore)
			res.append(fscore)
			# print(fscore)
		print(np.mean(res), np.var(res))
		save_for_visual(input_data, mask_data, y_pred, atts, cat_logits, clabels, data_loader)
		np.save(checkpointer_dir+'res', res)




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

