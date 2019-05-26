import numpy as np 

import nltk

import os
import xml.etree.ElementTree as ET
from wordcloud import WordCloud 

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
		cat = []
		for op in sen.iter("Opinion"):
			# cat += op.attrib['category']+';'
			op_cat = op.attrib['category'].split('#')
			if op_cat[0]=='RESTAURANT':
				key = op_cat[1]
			else:
				key = op_cat[0]
			if key not in cat:
				cat.append(key)

		# if len(cat)==0:
		# 	cat = 'unknown'
		# else:
		# 	cat = cat[:-1]
		if len(cat) == 0:
			cat = 'unknown'
		else:
			cat = ';'.join(cat)

		print(' '.join(tokens)+' '+cat)

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



def analysis(filename):
	fr = open(filename)
	data = fr.readlines()
	fr.close()

	cat = []
	m_cat = []

	cat_s = {}
	for line in data:
		line = line.strip()
		listfromline = line.split()
		cat.append(listfromline[-1])

		c = listfromline[-1]
		c = c.split(';')
		# m_cat.append('#'.join(c))


		# if len(c) == 2:
			# # print(c)
			# if c[0]!='LAPTOP':
			# 	key = c[0]
			# else:
			# 	key = c[1]
		for key in c:
			import nltk.stem as ns 
			lemmatizer = ns.WordNetLemmatizer()

			sent = [lemmatizer.lemmatize(lemmatizer.lemmatize(word,'n'),'v') for word in listfromline[:-1] if word not in ['#num','was']]
			if 'wa' in sent:
				print('wa in sent')
			if key not in cat_s:
				cat_s[key] = ' '.join(sent)
			else:
				cat_s[key] += ' '+' '.join(sent)

	for key,strings in cat_s.items():
		# print(key)
		import collections
		obj = collections.Counter(strings.split())
		tuples = obj.most_common(10)
		for word in tuples:
			print(word)
		print('---------------------------------')
		wordcloud = WordCloud(background_color = 'white', width=800, height=600, margin=2).generate(strings)
		import matplotlib.pyplot as plt 
		plt.imshow(wordcloud)
		plt.axis('off')
		plt.title(key)
		# print(strings)
		# plt.show()
		plt.savefig('lap_'+key+'.png')
		# wordcloud.to_file('wc'+key+'.png')









	# import collections
	# obj = collections.Counter(cat)
	# for key in sorted(obj.keys()):
	# 	print(key, obj[key])

	# print('---------------------------------')
	# obj = collections.Counter(m_cat)
	# for key in obj:
	# 	print(key, obj[key])



if __name__ == '__main__':
	# main('ABSA16_Laptops_Train_SB1_v2.xml')
	# main('EN_LAPT_SB1_TEST_.xml.gold')
	# annot('sent.txt')

	# analysis('sent_cat.txt')
	# main('ABSA16_Restaurants_Train_SB1_v2.xml')
	# main('EN_REST_SB1_TEST_gold.xml')
	# analysis('sent_res_cat.txt')
