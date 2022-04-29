# -*- coding: utf-8 -*-

"""
Created on 13/04/2022
@author: Alexandre Drevet & Axel François
github : https://github.com/AxFrancois/Numeric-Comm

"""

# Imports
from time import pthread_getcpuclockid
from dahuffman import HuffmanCodec
import sk_dsp_comm.fec_block as block
import matplotlib.pylab as plt
from commpy.channelcoding import cyclic_code_genpoly
import sk_dsp_comm.fec_block as block
import numpy as np

# Functions

#


def bitstring_to_bytes_p3(s):
    """fct : String formatting to bytes, Python3 version using .to_bytes()

    Args:
            s (bitstring): string of bit

    Returns:
            int: bytes
    """
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')

# fct Nberr in a binary array


def nbiterr(bits, detected_bits):
    err_matrix = (bits + detected_bits) % 2
    nb_err = err_matrix.sum()
    # print error
    return nb_err


def homemade_huffman_encoder(pTuple):
    intInBin = bin(pTuple[1]).replace("0b", "")
    valren = "0"*(pTuple[0]-len(intInBin)) + intInBin
    return valren


# Q2
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
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
ax.bar(charArray, probaArray)
handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='lower left', ncol=2,
                    frameon=False, bbox_to_anchor=(0.12, 0.88))
fig.savefig('repartition.jpg', bbox_extra_artists=(ax, legend))

# Q3
entropy = -np.sum(np.dot(probaArray, np.log2(probaArray)))
print("Entropy : ", entropy)

# Q4
huffman_codec_data = HuffmanCodec.from_data(txt)
# print(huffman_codec_data)
# huffman_codec_data.print_code_table()
huff_code_table = huffman_codec_data.get_code_table()

# Q5
phrase = """This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever."""
h_enc_data = huffman_codec_data.encode(phrase)
# print(h_enc_data[:30])
h_enc_data_str = ''.join(format(byte, '08b') for byte in h_enc_data)
# print(h_enc_data_str[:30])

# Q6
h_dec_data = huffman_codec_data.decode(h_enc_data)
# print(h_dec_data)

# Q7
# https://stackoverflow.com/questions/39718576/convert-a-byte-array-to-single-bits-in-a-array-python-3-5

mem = b'\x01\x02\xff'
x = np.fromstring(mem, dtype=np.uint8)
np.unpackbits(x).reshape(3, 8)

# Q8

phrase = """This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever."""
phrasehuff_str = ""
for letter in phrase:
    phrasehuff_str = phrasehuff_str + \
        homemade_huffman_encoder(huff_code_table[letter])
k = 4
Njam = k - len(phrasehuff_str) % k
phrasehuff_str = phrasehuff_str + '0'*Njam
phrasehuff_bit = bitstring_to_bytes_p3(phrasehuff_str)

# TP 2
# https://scikit-dsp-comm.readthedocs.io/en/latest/nb_examples/Block_Codes.html
# Example : create generator polynomial, - to replace by yours !
genpoly_7_4_1 = format(cyclic_code_genpoly(7, 4)[1], 'b')

# Example (7,4) by default (1st poly), to replace by yours
cyccode = block.FECCyclic(genpoly_7_4_1)
