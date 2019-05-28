import matplotlib.pyplot as plt 
import numpy as np 
import sys

my_color = ['#107c10','#DC3C00','#7719AA','#0078D7','#DC6141','#4269A5','#39825A','#DC6141']
ckpt_file = 'ckpt_'+sys.argv[1]


def main():
	train_loss = np.load(ckpt_file+'/train_loss.npy')
	val_score = np.load(ckpt_file+'/val_score.npy')

	fig,ax1 = plt.subplots()
	ax2 = ax1.twinx()
	index = list(range(len(train_loss)))

	l1, = ax1.plot(index, train_loss, my_color[0])
	l2, = ax2.plot(index, val_score, my_color[1])

	ax1.set_xlabel('epoch')
	ax1.set_ylabel('training loss')
	ax2.set_ylabel('testing score')


	plt.legend([l1,l2],['train loss','test score'], loc='center right')
	plt.show()


if __name__ == '__main__':
	main()
