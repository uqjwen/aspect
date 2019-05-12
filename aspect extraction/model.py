import tensorflow as tf 
import logging
import argparse
import numpy as np 
from keras.layers import Dense

epsilon = 1e-7

class Model():
	def __init__(self,args,maxlen,vocab):
		#pass

		vocab_size = len(vocab)

		###########inputs#################
		self.sen_input = tf.placeholder(tf.int32, shape = [None, maxlen])
		self.neg_input = tf.placeholder(tf.int32, shape = [None, args.neg_size, maxlen])

		#########embedding###############
		self.word_emb = tf.Variable(tf.random_uniform([vocab_size, args.emb_dim], -1.0,1.0))
		self.aspect_emb = tf.Variable(tf.random_uniform([args.aspect_size, args.emb_dim], -1.0,1.0))

		e_w = tf.nn.embedding_lookup(self.word_emb, self.sen_input) #[batch_size, maxlen, emb_dim]
		y_s = tf.reduce_mean(e_w,1) #[batch_size, emb_dim]
		z_s = self.weightedSum(e_w,y_s, args.emb_dim) #[batch_size, emb_dim]

		##########negative instances ###################
		e_n = tf.nn.embedding_lookup(self.word_emb, self.neg_input) #[batch_size, neg_size, maxlen, emb_dim]
		z_n = tf.reduce_mean(e_n, axis=2) #[batch_size, neg_size, emb_dim]

		#########reconstruction#############
		p_t = Dense(args.aspect_size, activation='softmax', kernel_initializer='lecun_uniform')(z_s)#[batch_size, aspect_size]
		r_s = tf.matmul(p_t, self.aspect_emb) ##[batch_size, emb_dim]


		loss = self.maxMarginLoss(z_s, z_n, r_s)

		self.cost = tf.reduce_mean(loss) + self.ortho_reg(args.ortho_reg, self.aspect_emb) #+orthogonal regulariation
		self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)





	def maxMarginLoss(self,z_s, z_n, r_s):
		z_s = z_s/tf.sqrt(tf.reduce_sum(tf.square(z_s), axis=-1, keep_dims = True))
		z_n = z_n/tf.sqrt(tf.reduce_sum(tf.square(z_n), axis=-1, keep_dims = True))
		r_s = r_s/tf.sqrt(tf.reduce_sum(tf.square(r_s), axis=-1, keep_dims = True))

		pos = tf.reduce_sum(z_s*r_s, axis=-1, keep_dims = True)# [batch_size, 1]
		e_rs = tf.expand_dims(r_s, axis=1)# [batch_size, 1, emb_dim]
		neg = tf.reduce_sum(z_n*e_rs, axis=-1)#[batch_size, neg_size]

		loss = tf.reduce_sum(tf.maximum(0.,1.-pos+neg),axis=-1, keep_dims = True) ##[batch_size,1]
		return loss


	def weightedSum(self,e_w,y_s, emb_dim):

		att_layer = Dense(emb_dim, kernel_initializer='lecun_uniform', name = 'att_layer')

		e_m = att_layer(e_w)# [batch_size, maxlen, emb_dim]
		expand_ys = tf.expand_dims(y_s,1)#[batch_size,1,emb_dim]

		atts = tf.reduce_sum(e_m*expand_ys, axis=-1) # [batch_size, maxlen]
		atts = tf.expand_dims(tf.nn.softmax(atts),-1)#[batch_size, maxlen,1]
		w_s = tf.reduce_sum(e_w*atts,axis=1) ## [batch_size, emb_dim]

		return w_s


	def ortho_reg(self, ortho_reg, weight_matrix):
		w_n = weight_matrix/(epsilon+ tf.reduce_sum(tf.square(weight_matrix), axis=-1, keep_dims = True))
		reg = tf.reduce_sum(tf.square(tf.matmul(w_n, tf.transpose(w_n))- tf.eye(w_n.shape.as_list()[0])))
		return ortho_reg*reg









if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>',default = './output_dir/', help="The path to the output directory")
	parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=200, help="Embeddings dimension (default=200)")
	parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50, help="Batch size (default=50)")
	parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000, help="Vocab size. '0' means no limit (default=9000)")
	parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=14, help="The number of aspects specified by users (default=14)")
	parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>',default = './preprocessed_data/restaurant/w2v_embedding/', help="The path to the word embeddings file")
	parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15, help="Number of epochs (default=15)")
	parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20, help="Number of negative instances (default=20)")
	parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
	parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
	parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
	parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='restaurant', help="domain of the corpus {restaurant, beer}")
	parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1, help="The weight of orthogonol regularizaiton (default=0.1)")

	args = parser.parse_args()
	out_dir = args.out_dir_path + '/' + args.domain	

	model = Model(args, 100, {})