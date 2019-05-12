from model import Model
import numpy as np 
import argparse
import sys
import tensorflow as tf 
from keras.preprocessing import sequence
import reader as dataset
import codecs


vocab_size = 1000
vocab = {str(i):i for i in range(vocab_size)}

maxlen = 20
batch_size = 50
trainx = np.random.randint(0,vocab_size,(10000,maxlen))
testx = np.random.randint(0,vocab_size,(2000,maxlen))





def sentence_batch_generator(data, batch_size):
    n_batch = len(data) / batch_size
    batch_count = 0
    np.random.shuffle(data)

    while True:
        if batch_count == n_batch:
            np.random.shuffle(data)
            batch_count = 0

        batch = data[batch_count*batch_size: (batch_count+1)*batch_size]
        batch_count += 1
        yield batch

def negative_batch_generator(data, batch_size, neg_size):
    data_len = data.shape[0]
    dim = data.shape[1]

    while True:
        indices = np.random.choice(data_len, batch_size * neg_size)
        samples = data[indices].reshape(batch_size, neg_size, dim)
        yield samples

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>',default = './output_dir/', help="The path to the output directory")
	parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=200, help="Embeddings dimension (default=200)")
	parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50, help="Batch size (default=50)")
	parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000, help="Vocab size. '0' means no limit (default=9000)")
	parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=14, help="The number of aspects specified by users (default=14)")
	parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>',default = './preprocessed_data/restaurant/w2v_embedding', help="The path to the word embeddings file")
	parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15, help="Number of epochs (default=15)")
	parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20, help="Number of negative instances (default=20)")
	parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
	parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
	parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
	parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='restaurant', help="domain of the corpus {restaurant, beer}")
	parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1, help="The weight of orthogonol regularizaiton (default=0.1)")

	args = parser.parse_args()
	out_dir = args.out_dir_path + '/' + args.domain	


	vocab, train_x, test_x, overall_maxlen = dataset.get_data(args.domain, vocab_size=args.vocab_size, maxlen=args.maxlen)
	vocab_inv = {ind:w for w,ind in vocab.items()}

	model = Model(args, maxlen, vocab)
	sen_gen = sentence_batch_generator(trainx, batch_size)
	neg_gen = negative_batch_generator(trainx, batch_size, args.neg_size)

	batches_per_epoch = 1000
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		if args.emb_path:
			from w2vEmbReader import W2VEmbReader as EmbReader 
			emb_reader = EmbReader(args.emb_path, emb_dim = args.emb_dim)
			word_emb = sess.run(model.word_emb)
			sess.run(tf.assign(model.word_emb, emb_reader.get_emb_matrix_given_vocab(vocab, word_emb)))
			sess.run(tf.assign(model.aspect_emb, emb_reader.get_aspect_matrix(args.aspect_size)))
		checkpoint_dir = './ckpt/'
		min_loss = float('inf')
		
		for ii in range(args.epochs):
			loss,max_margin_loss = 0.,0.






			for b in range(batches_per_epoch):
				sen_input = sen_gen.__next__()
				neg_input = neg_gen.__next__()
				_,cost = sess.run([model.train_op, model.cost], feed_dict={model.sen_input:sen_input,
																			model.neg_input:neg_input})
				sys.stdout.write('\r {}/{} epoch, {}/{} batch, train loss:{}'.format(ii,args.epochs,b,batches_per_epoch,cost))

				loss += cost/batches_per_epoch



			if loss<min_loss :
				min_loss = loss
				saver.save(sess, checkpoint_dir+'model.ckpt', global_step = (ii+1)*batches_per_epoch)
				word_emb,aspect_emb = sess.run([model.word_emb, model.aspect_emb])
				word_emb = word_emb/np.linalg.norm(word_emb, axis=-1, keepdims=True)
				aspect_emb = aspect_emb/np.linalg.norm(aspect_emb, axis=-1, keepdims=True)

				aspect_file = codecs.open(out_dir+'/aspect.log','w','utf-8')

				for ind in range(len(aspect_emb)):
					desc = aspect_emb[ind]
					sims = word_emb.dot(desc.T)
					ordered_words = np.argsort(sims)[::-1]
					desc_list = [vocab_inv[w] for w in ordered_words[:100]]
					aspect_file.write('Aspect %d:\n'%ind)
					aspect_file.write(' '.join(desc_list)+'\n\n')

