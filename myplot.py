import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
import sys
from keras.preprocessing.sequence import pad_sequences

my_color = ['#107c10','#DC3C00','#7719AA','#0078D7','#DC6141','#4269A5','#39825A','#DC6141']
ckpt_file = 'ckpt_'+sys.argv[1]+'/'


def main():
	train_loss = np.load(ckpt_file+'train_loss.npy')
	val_score = np.load(ckpt_file+'val_score.npy')
	val_1,val_2 = val_score[:,0], val_score[:,1]

	fig,ax1 = plt.subplots()
	ax2 = ax1.twinx()
	index = list(range(len(train_loss)))

	l1, = ax1.plot(index, train_loss, my_color[0], label='train loss')
	l2, = ax2.plot(index, val_1, my_color[1], label = 'test f1score_1')
	l3, = ax2.plot(index, val_2, my_color[2], label = 'test f1score_2')

	ax1.set_xlabel('epoch')
	ax1.set_ylabel('training loss')
	ax2.set_ylabel('testing score')


	# plt.legend([l1,l2],['train loss','test score'], loc='center right')
	ax1.legend(loc='center left')
	ax2.legend(loc='center right')
	plt.show()

def get_mat(sent1, sent2, atts):
	i = 0
	j = 0
	sent = []
	sent_att = []
	while i<len(sent1) and j<len(sent2):
		while i<len(sent1) and sent1[i]!=sent2[j]:
			sent.append(sent1[i])
			sent_att.append(0)
			i+=1
		if i<len(sent1):
			assert sent1[i] == sent2[j]
			sent.append(sent1[i])
			sent_att.append(atts[j])
			i+=1
			j+=1
	return sent, sent_att


def att():
	# filename = './ckpt_'+sys.argv[1]+'/visual.txt'
	filename = ckpt_file+'visual.txt'
	fr = open(filename)
	clabels = {}
	data = fr.readlines()
	fr.close()

	sents = []
	sent_atts = []
	labels = []
	counter = 0
	for i in range(0,len(data),7):
		sent1 = data[i].strip().split('\t')
		sent2 = data[i+1].strip().split('\t')
		atts = data[i+2].strip().split('\t')
		atts = list(map(float, atts))
		# print(atts)
		label = data[i+3].strip().split('\t')

		sent,sent_att = get_mat(sent1, sent2, atts)
		# print(sent_att)
		sents.append(sent)
		sent_atts.append(sent_att)
		labels.append(label)
		for la in label:
			if la not in clabels:
				clabels[la] = [counter]
			else:
				clabels[la].append(counter)



		counter+=1
		# print(sent, len(sent))
		# print(sent_att, len(sent_att))
		# print('-------------------')
		# break
	# print(sent_atts)
	return sents, sent_atts, labels, clabels




def pad_my_sequence(to_pad, maxlen, pad_with):
	res = []
	for pad in to_pad:
		if len(pad)>maxlen:
			res.append(pad[:maxlen])
		else:
			res.append([pad_with]*(maxlen-len(pad))+pad)
	return np.array(res)


# def pad_str_sequence(sent, maxlen):
# 	sents = []
# 	for s in sent:
# 		if len(s)>maxlen:
# 			sents.append(s[:maxlen])
# 		else:
# 			sents.append(['']*(maxlen-len(s))+s)
# 	return np.array(sents)

# def pad_float_sequence(att, maxlen):
# 	atts = []
# 	for a in att:
# 		if len(a)>maxlen:
# 			atts.append(a[:maxlen])
# 		else:
# 			atts.append([0]*(maxlen-len(a))+a)
# 	return np.array(atts)


def sampling(sents, sent_atts, labels, clabels, max = 10):
	clabel = np.random.choice(list(clabels.keys()))
	index = np.random.choice(clabels[clabel], max, replace = False)

	return sents[index], sent_atts[index], np.array(labels)[index]



def visual_atts():
	sents, sent_atts, labels, clabels = att()

	maxlen = 20
	# print(sent_atts)
	sents = pad_my_sequence(sents, maxlen, '')
	# sent_atts = pad_sequences(sent_atts, maxlen)
	sent_atts = pad_my_sequence(sent_atts, maxlen, 0)
	print(sent_atts)
	# print(sent_atts)
	
	sub_sents, sub_atts, sub_labels = sampling(sents, sent_atts, labels, clabels)


	# print(sub_atts)

	plt.figure()
	sns.heatmap(sub_atts, annot = sub_sents, cmap = 'autumn_r', fmt='s', linewidths=0.5)
	plt.show()

# def attention()

if __name__ == '__main__':
	main()
	# att('laptop')
	# visual_atts()
