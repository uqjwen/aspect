import numpy as np 

import nltk

import os
import xml.etree.ElementTree as ET

def my_split(line):
	tokens = []
	token = ''
	idx = 0
	line += "#"
	while idx < len(line):
		if line[idx].isalpha():
			begin = idx
			while idx < len(line) and line[idx].isalpha():
				idx +=1
			token = line[begin:idx]
			tokens.append(token.lower())
		elif line[idx].isdigit():
			begin = idx
			while idx<len(line) and line[idx].isdigit():
				idx+=1
			token = line[begin:idx]
			tokens.append('#num')
		else:
			idx+=1
	return tokens


def main(filename):
	xmlFilePath = os.path.abspath(filename)
	tree = ET.parse(xmlFilePath)
	root = tree.getroot()
	# print(dir(root))
	sentences = []
	labels = []
	pos = []



	for sen in root.iter('sentence'):
		sentext = sen.find("text").text

		tokens = my_split(sentext)

		print(' '.join(tokens))

def leagal_sub_labels(labels, length):
	label = labels.split(',')
	try:
		label = list(map(int,label))
	except:
		return False

	for i,l in enumerate(label):
		if i==0:
			begin = l 
		else:
			if l-begin !=i or l >= length:
				# print('error->sub_',)
				return False
	return True



def leagal_input(labels, length):
	if len(labels)==0:
		return True

	sub_labels = labels.split(';')
	# length = len(labels)

	for i,sub_label in enumerate(sub_labels):
		if not leagal_sub_labels(sub_label, length):
			print("error->sub:",i,', input again')
			return False
	return True

def get_labels(input_labels, length):
	labels = ['0']*length


	for sub_labels in input_labels.split(';'):
		for i,label in enumerate(sub_labels.split(',')):
			if label != '':
				if i==0:
					labels[int(label)] = '1'
				else:
					labels[int(label)] = '2'
	return ' '.join(labels)


def annot(filename):
	fr = open("sent.txt")
	data = fr.readlines()
	fr.close()

	print("begin line num:")
	begin = int(input())

	fr = open('sent_annot.txt','a')

	for line in data[begin:]:
		line = line.strip()
		listfromline = line.split()

		num_token = ''
		for i,token in enumerate(listfromline):
			num_token += str(i)+':'+token+' '
		print('-------------------------------------------')
		print(line)
		print('====================')
		print(num_token)
		length = len(listfromline)

		labels = input()
		while leagal_input(labels, length)==False:
			labels = input()
		labels = get_labels(labels,length)
		fr.write(line+'\n')
		fr.write(labels+'\n')







if __name__ == '__main__':
	# main('ABSA16_Laptops_Train_SB1_v2.xml')
	# main('EN_LAPT_SB1_TEST_.xml.gold')
	annot('sent.txt')