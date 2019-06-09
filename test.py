import sys
import numpy as np 
import tensorflow as tf 
def main():
	tf.flags.DEFINE_string('ckpt_dir', './ckpt/','checkpoint dir')
	FLAGS = tf.flags.FLAGS
	FLAGS(sys.argv)
	print(FLAGS.ckpt_dir)	


if __name__ == '__main__':
	main()