import numpy as np 
import gensim 
import pickle 
from gensim.models import Word2Vec
import sys
class MySentence():
	def __init__(self, sentence):
		self.sentence = sentence
	def __iter__(self):
		for sentence in self.sentence:
			yield sentence
def main(domain ,emb_size):
	if domain == 'rest':
		data_file = './pkl/data_rest_2016_2.pkl'
		save_file = './pkl/gensim_rest_2016_2_'+str(emb_size)
		# data_file = 'data_rest.pkl'
	else:
		data_file = './pkl/data_laptop_2014.pkl'
		save_file = './pkl/gensim_laptop_2014_'+str(emb_size)



	data = pickle.load(open(data_file, 'rb'))



	sentences = data['raw_sentence']
	# sen = MySentence(sentences)
	model = gensim.models.Word2Vec(sentences, size=emb_size, window=10,min_count=1, workers=4)
	model.save(save_file)
	# model = Word2Vec(min_count=1)
	# model.build_vocab(sentences)
	# model.train(sentences*100, total_examples=model.corpus_count, epochs = model.iter)
	# model.save('my_gensim_model')



if __name__ == '__main__':
	domain = sys.argv[1]
	emb_size = int(sys.argv[2])
	print(domain,emb_size)
	main(domain, emb_size)