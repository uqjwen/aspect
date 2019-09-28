import numpy as np 
from docx import Document
from docx.shared import Inches
from docx.shared import Pt

def vdoc(tokens, ate_alpha, acd_alpha, index):
	i = 0
	while i<len(tokens) and tokens[i] == '':
		i+=1
	tokens = tokens[i:]
	ate_alpha = ate_alpha[i:]
	acd_alpha = acd_alpha[i:][:,0]

	# print(np.max(ate_alpha[:,1:], axis=-1))
	ate_alpha = np.max(ate_alpha[:,1:], axis=-1)
	document = Document()
	print(acd_alpha)
	print(1./len(acd_alpha))
	single_gate(tokens, ate_alpha.copy(), document)
	# print(ate_alpha)
	# new_atts = simul_ate(ate_alpha)

	# single_gate(tokens, new_atts, document)

	single_document(tokens, acd_alpha.copy(), document)

	new_atts = simul_acd(acd_alpha)
	print(new_atts)

	single_document(tokens, new_atts.copy(), document)	
	document.save(str(index)+'_att.docx')

def simul_ate(input_atts):
	atts = input_atts[:]
	args = np.argsort(atts)[::-1]
	# print(atts)
	# print(args)



	num = np.random.choice(range(1,5))
	value = np.mean(atts[atts>0])


	start = len(atts[atts>0])
	end = start+num 
	end = min(end,len(atts))
	atts[args[start:end]] = value
	return atts

def simul_acd(input_atts):
	new_atts = input_atts/1000
	new_atts = np.exp(new_atts)/np.sum(np.exp(new_atts))
	return new_atts

def single_gate(tokens, input_atts, document):
	atts = input_atts[:]
	normal_size = 10
	un_zero = atts[atts>0]
	un_zero = un_zero+10
	un_zero = np.minimum(un_zero, 20)

	atts[atts>0] = un_zero
	atts[atts<0] = 10
	atts = atts.astype(int)
	p = document.add_paragraph('ATE attentions------------------------')
	p = document.add_paragraph('')
	for token, att in zip(tokens, atts):
		run = p.add_run(token+' ')
		run.font.size = Pt(att)
		if att>10:
			run.bold = True

def single_document(tokens, atts, document):
	atts = np.array(atts)
	# print('haha', atts)

	# atts = 10+atts*20


	nz_atts = atts[atts!=0]
	min_size = 10
	max_size = 20
	if len(nz_atts)<2:
		return 
	min_att = min(nz_atts)
	max_att = max(nz_atts)

	if min_att == max_att:
		new_atts = np.array([15]*len(nz_atts))
	else:
		new_atts = min_size+(max_size - min_size)/(max_att - min_att)*(nz_atts - min_att)

	# # print()

	# atts[atts!=0] = new_atts
	atts = atts.astype(np.int32)
	# print(atts)
	# print(atts)

	p = document.add_paragraph('ACD attentions--------------------------')
	p = document.add_paragraph('')

	for token,att in zip(tokens, atts):
		run = p.add_run(token+' ')
		if att>0:
			run.bold = True
			run.font.size = Pt(att)



if __name__ == '__main__':
	vdoc(tokens, ate_alpha, ace_alpha, index)