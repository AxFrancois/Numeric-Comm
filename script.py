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
#print(huffman_codec_data[:30])
h_enc_data = huffman_codec_data.encode(txt)
#print(h_enc_data[:30])
h_enc_data_str = ''.join(format(byte,'08b') for byte in h_enc_data)
#print(h_enc_data_str[:30])
