# -*- coding: utf-8 -*-

"""
Created on 13/04/2022
@author: Alexandre Drevet & Axel François
github : https://github.com/AxFrancois/Numeric-Comm

"""

# Imports
# from time import pthread_getcpuclockid
# from operator import length_hint

from codecs import Codec
import random
import math

import commpy.modulation as mod
import matplotlib.pylab as plt
import numpy as np
import sk_dsp_comm.digitalcom as digcom
import sk_dsp_comm.fec_block as block
from commpy.channelcoding import cyclic_code_genpoly
from dahuffman import HuffmanCodec
from scipy import special

# Functions


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


def huffman_decoder_table(huff_code_table):
    dic = {}
    for item in huff_code_table.items():
        intInBin = bin(item[1][1]).replace("0b", "")
        valbin = "0"*(item[1][0]-len(intInBin)) + intInBin
        dic[valbin] = item[0]
    return dic


# TP1
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

fig = plt.figure(0)
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
phrase = """Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyle

This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever.  You may copy it, give it away or
re-use it under the terms of the Project Gutenberg License included
with this eBook or online at www.gutenberg.net


Title: The Adventures of Sherlock Holmes

Author: Arthur Conan Doyle"""
h_enc_data = huffman_codec_data.encode(txt)
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
# print(phrasehuff_str)


# Q9
phrasehuff_str_err = phrasehuff_str
i = 0
while i < len(phrasehuff_str_err):
    randnum = random.randint(0, k-1)
    if phrasehuff_str_err[i+randnum] == '0':
        phrasehuff_str_err = phrasehuff_str_err[:i +
                                                randnum] + '1' + phrasehuff_str_err[i+randnum+1:]
    else:
        phrasehuff_str_err = phrasehuff_str_err[:i +
                                                randnum] + '0' + phrasehuff_str_err[i+randnum+1:]
    i += k

# print(phrasehuff_str_err)


# TP 2
# https://scikit-dsp-comm.readthedocs.io/en/latest/nb_examples/Block_Codes.html

# Q1 et 2
# Example : create generator polynomial, - to replace by yours !
genpoly_7_4_1 = format(cyclic_code_genpoly(7, 4)[1], 'b')

# Q3
# Example (7,4) by default (1st poly), to replace by yours
cyccode = block.FECCyclic(genpoly_7_4_1)

# Q4 : cf Q9 TP1

# Q5
length = int(len(phrasehuff_str) / 4)
# x1 = np.array(list(phrasehuff_str), dtype=int).astype(int)
x1 = np.array(list(h_enc_data_str), dtype=int).astype(int)
# x1 = np.reshape(x1, (k, length)).astype(int)
# print(np.shape(x1))

codewords = cyccode.cyclic_encoder(x1)
# print(codewords)

# Q6
decoded = cyccode.cyclic_decoder(codewords)
decode_byte = ''.join(str(x) for x in decoded)
output = bitstring_to_bytes_p3(decode_byte)

# Q7


# print(huffman_codec_data.decode(output))


# TP3
plt.figure(1)
for M in [4, 16]:
    print("Cas M = ", M)
    k_mod = math.log2(M)
    print("k_mod = ", k_mod)  # faire le bourrage -> fait par huffman
    modem = mod.QAMModem(M)
    modulated = modem.modulate(codewords)
    EbSurN0 = range(1, 10)  # en dB
    valempirique = []
    for isnr in EbSurN0:
        snr = isnr + 10*math.log10(k_mod)
        # bruit gaussien
        bruited = digcom.cpx_awgn(modulated, snr, 1)
        demodulated = modem.demodulate(bruited, 'hard')
        diff = abs(codewords - demodulated)
        unique, counts = np.unique(diff, return_counts=True)
        result = dict(zip(unique, counts))
        try:
            result[1]
        except:
            result[1] = 0
        proba = result[1]/len(demodulated) + 1e-5
        valempirique.append(proba)
        print(isnr, result, round(proba*100, 10), "%")
    if M == 4:
        plt.plot(EbSurN0, valempirique, 'ro--')
    elif M == 16:
        plt.plot(EbSurN0, valempirique, 'bo--')

# car il faut convertir en linéaire
valtheorique = []
for value in EbSurN0:
    valtheorique.append(1/2 * special.erfc(math.sqrt(10**(value/10))))
print(valtheorique)
plt.plot(EbSurN0, valtheorique, 'g^--')
plt.yscale('log')
plt.show()

M = 16
modem = mod.QAMModem(M)
k_mod = math.log2(M)
modulated = modem.modulate(codewords)
EbSurN0 = 8  # en dB
snr = EbSurN0 + 10*math.log10(k_mod)
bruited = digcom.cpx_awgn(modulated, snr, 1)
demodulated = modem.demodulate(bruited, 'hard')
diff1 = abs(codewords - demodulated)
unique, counts = np.unique(diff1, return_counts=True)
result1 = dict(zip(unique, counts))
try:
    result1[1]
except:
    result1[1] = 0
proba1 = result1[1]/len(demodulated)
print(proba1)

decoded = cyccode.cyclic_decoder(demodulated.astype(int))
diff2 = abs(decoded - x1)
unique, counts = np.unique(diff2, return_counts=True)
result2 = dict(zip(unique, counts))
try:
    result2[1]
except:
    result2[1] = 0
proba2 = result2[1]/len(decoded)
print(proba2)
decode_byte = ''.join(str(x) for x in decoded)
output = bitstring_to_bytes_p3(decode_byte)

print(huffman_codec_data.decode(output))
