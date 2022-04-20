# -*- coding: utf-8 -*-

"""
Created on 13/04/2022
@author: Alexandre Drevet & Axel François
github : https://github.com/AxFrancois/Numeric-Comm

"""

from dahuffman import HuffmanCodec
import sk_dsp_comm.fec_block as block
import matplotlib.pylab as plt
import math
#scikit_commpy
import numpy as np

#Q2
f = open("The_Adventures_of_Sherlock_Holmes_A_Scandal_In_Bohemia.txt", "r")
txt = f.read()
char = []
count = []

for letter in txt:
    if letter in char:
        count[char.index(letter)] += 1
    else:
        char.append(letter)
        count.append(1)

charArray = np.array(char)
countArray = np.array(count)
probaArray = countArray/len(txt)

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.85,0.85])
ax.bar(charArray,probaArray)
handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='lower left', ncol=2, frameon=False, bbox_to_anchor=(0.12, 0.88))
fig.savefig('repartition.jpg', bbox_extra_artists=(ax,legend))

#Q3
entropy = -np.sum(np.dot(probaArray, np.log2(probaArray)))
print("Entropy : ", entropy)

#Q4
huffman_codec_data = HuffmanCodec.from_data(txt)
#print(huffman_codec_data)
#huffman_codec_data.print_code_table()
huff_code_table = huffman_codec_data.get_code_table()

#Q5
phrase = """This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever."""
h_enc_data = huffman_codec_data.encode(phrase)
#print(h_enc_data[:30])
h_enc_data_str = ''.join(format(byte,'08b') for byte in h_enc_data)
#print(h_enc_data_str[:30])

#Q6
h_dec_data = huffman_codec_data.decode(h_enc_data)
#print(h_dec_data)

#Q7
encoded_phrase = ""

def huffman_encoder(pTuple):
    intInBin = bin(pTuple[1]).replace("0b", "")
    valren = "0"*(pTuple[0]-len(intInBin)) + intInBin
    return valren

def huffman_decoder_table(huff_code_table):
    dic = {}
    for item in huff_code_table.items():   
        intInBin = bin(item[1][1]).replace("0b", "")
        valbin = "0"*(item[1][0]-len(intInBin)) + intInBin
        dic[valbin] = item[0]
    return dic

actually_useful_table = huffman_decoder_table(huff_code_table)
#print(actually_useful_table)
for letter in phrase:
    encoded_phrase = encoded_phrase + huffman_encoder(huff_code_table[letter])

#print(encoded_phrase[:30])


decoded_phrase = ""
i= 0
while i < len(encoded_phrase):
    for j in range(i,len(encoded_phrase)):
        if encoded_phrase[i:j+1] in actually_useful_table:
            decoded_phrase = decoded_phrase + actually_useful_table[encoded_phrase[i:j+1]]
            i = j+1
            break

#print(decoded_phrase)

#Q8
phrase = """This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever."""
k = 4
encoded_phrase = ""
for letter in phrase:
    encoded_phrase = encoded_phrase + huffman_encoder(huff_code_table[letter])
i=0
while len(encoded_phrase) % k != 0:
    i+=1
    encoded_phrase+= '0'
    
print("Ajout de", i, "padding")

enc_list = []
enc_sub_list = []
for num in encoded_phrase:
    enc_sub_list.append(num)
    if len(enc_sub_list) == 4:
        enc_list.append(enc_sub_list)
        enc_sub_list = []

print(np.matrix(enc_list))