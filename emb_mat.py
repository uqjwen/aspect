import numpy as np 
import gensim 
import pickle 
from gensim.models import Word2Vec
class MySentence():
	def __init__(self, sentence):
		self.sentence = sentence
	def __iter__(self):
		for sentence in self.sentence:
			yield sentence
def main():
	data = pickle.load(open('data.pkl', 'rb'))
	sentences = data['raw_sentence']
	# sen = MySentence(sentences)
	model = gensim.models.Word2Vec(sentences*100, size=100, window=10,min_count=1, workers=4)
	model.save("my_gensim_model")
	# model = Word2Vec(min_count=1)
	# model.build_vocab(sentences)
	# model.train(sentences*100, total_examples=model.corpus_count, epochs = model.iter)
	# model.save('my_gensim_model')
if __name__ == '__main__':
	main()