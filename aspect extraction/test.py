import gensim 


class MySentence():
	def __init__(self,filename):
		self.filename = filename
	def __iter__(self):
		fr = open(self.filename)
		data = fr.readlines()
		for line in data:
			yield line.lower().split()

sentence = MySentence('./preprocessed_data/restaurant/train.txt')

model = gensim.models.Word2Vec(sentence, size=200, window=5, min_count=10, workers=4)

model.save('./my_gensim_model')
