# -*- coding: utf-8 -*-
"""
Created on 13/04/2022
@author: Alexandre Drevet & Axel François
github : https://github.com/AxFrancois/Numeric-Comm

"""

#----------------------- Imports -----------------------#
# from time import pthread_getcpuclockid
# from operator import length_hint
# from codecs import Codec

import math
import sys
import os
import multiprocessing as mp
import random

import commpy.modulation as mod
import matplotlib.pylab as plt
import numpy as np
import sk_dsp_comm.digitalcom as digcom
import sk_dsp_comm.fec_block as block
from click import progressbar
from commpy.channelcoding import cyclic_code_genpoly
from dahuffman import HuffmanCodec
from scipy import special

from header_tp_eti import bitstring_to_bytes_p3

mysem = mp.Semaphore(0)
valempirique = mp.Array('f', range(13))  # []
valempirique2 = mp.Array('f', range(13))  # []

#----------------------- Functions -----------------------#


# def bitstring_to_bytes_p3(s):
"""String formatting to bytes, Python3 version using .to_bytes()

    Args:
            s (bitstring): string of bit

    Returns:
            int: bytes
    """
# return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')


def homemade_huffman_encoder(pTuple):
    """Encodeur Huffman n'utilisant pas HuffmanCodec.encode()

    Args:
            pTuple (tuple): tuple de la lettre obtenue par HuffmanCodec.get_code_table()

    Returns:
            str: bits de la lettre encodée
    """
    intInBin = bin(pTuple[1]).replace("0b", "")
    valren = "0"*(pTuple[0]-len(intInBin)) + intInBin
    return valren


def huffman_decoder_table(huff_code_table):
    """_summary_

    Args:
            bits (_type_): _description_
            detected_bits (_type_): _description_

    Returns:
            _type_: _description_
    """
    dic = {}
    for item in huff_code_table.items():
        intInBin = bin(item[1][1]).replace("0b", "")
        valbin = "0"*(item[1][0]-len(intInBin)) + intInBin
        dic[valbin] = item[0]
    return dic


def graphLettres(txt):
    # TP1 Q2
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
    print("Entropie calculé : ", entropy)


def source_encode_decode(txt, valempirique, valempirique2, drawgraph=False, addError=False):
    # TP1 Q4
    huffman_codec_data = HuffmanCodec.from_data(txt)
    print("Code table du codec Huffman : ")
    huffman_codec_data.print_code_table()
    print("\n")
    huff_code_table = huffman_codec_data.get_code_table()

    # TP1 Q5
    print("Debut encodage Huffman")
    h_enc_data = huffman_codec_data.encode(txt)
    h_enc_data_str = ''.join(format(byte, '08b') for byte in h_enc_data)
    print("Fin encodage Huffman")

    # Calcul du taux d'erreur sur un code huffman bruité avec 1 erreur par bloc de 4 bits
    print("Calculs de probabilités : ", addError)
    if addError == True:
        h_enc_data_str_err = add_error(h_enc_data_str, 4)
        h_enc_data_err = bitstring_to_bytes_p3(
            h_enc_data_str_err)
        h_dec_err = huffman_codec_data.decode(h_enc_data_err)
        nbError = 0
        for i in range(min(len(h_dec_err), len(txt))):
            if h_dec_err[i] != txt[i]:
                nbError += 1
        print("Taux d'erreur sur un code huffman bruité avec 1 erreur par bloc de 4 bits :", round(
            nbError/max(len(h_dec_err), len(txt))*100, 4), "%.")

        # TP2
    print("\n")
    returned_h_enc_data = channel_encode_decode(
        h_enc_data_str, valempirique, valempirique2, drawgraph, addError)

    # TP1 Q6
    print("Debut décodage Huffman")
    h_dec_data = huffman_codec_data.decode(returned_h_enc_data)
    print("Fin décodage Huffman")

    return h_dec_data


def add_error(txt, k):
    # TP1 Q9
    error_txt = txt
    i = 0
    while i < len(error_txt):
        randnum = random.randint(0, k-1)
        if error_txt[i+randnum] == '0':
            error_txt = error_txt[:i + randnum] + '1' + error_txt[i+randnum+1:]
        else:
            error_txt = error_txt[:i + randnum] + '0' + error_txt[i+randnum+1:]
        i += k
    return error_txt


def channel_encode_decode(ph_enc_data, valempirique, valempirique2, drawgraph=False, addError=False):
    # TP2 Q1 et 2
    # Création polynôme générateur
    genpoly_7_4_1 = format(cyclic_code_genpoly(7, 4)[1], 'b')

    # TP2 Q3
    # Code cyclique associé au polynôme générateur
    cyccode = block.FECCyclic(genpoly_7_4_1)

    # TP2 Q5
    # Conversion str en np array
    h_enc_array = np.array(list(ph_enc_data), dtype=int).astype(int)
    # Encodage cyclique
    print("Debut encodage cyclique")
    codewords = cyccode.cyclic_encoder(h_enc_array)
    print("Fin encodage cyclique")

    # Calcul du taux d'erreur sur un code cyclique bruité avec 1 erreur par bloc de 7 bits
    print("Calculs de probabilités : ", addError)
    if addError == True:
        codewords_str = ''.join(str(x) for x in codewords)
        codewords_str_err = codewords_str
        h_enc_str = ''.join(str(x) for x in h_enc_array)
        for err in range(1, 4):
            for i in range(err):
                codewords_str_err = add_error(codewords_str_err, 7)
            codewords_err = np.array(
                list(codewords_str_err), dtype=int).astype(int)
            decoded_err = cyccode.cyclic_decoder(codewords_err)
            decoded_err_bits_str = ''.join(str(x) for x in decoded_err)

            nbError = 0
            for i in range(len(h_enc_str)):
                if h_enc_str[i] != decoded_err_bits_str[i]:
                    nbError += 1
            print("Taux d'erreur sur un code cyclique bruité avec {} erreur(s) par bloc de 7 bits :".format(
                err), round(nbError/len(codewords_str)*100, 4), "%.")

    # TP3
    print("\n")
    returned_codewords = modulation_demodulation(
        codewords, h_enc_array, cyccode, valempirique, valempirique2, drawgraph, addError)

    # TP2 Q6
    # Décodage cyclique
    print("Debut décodage cyclique")
    decoded = cyccode.cyclic_decoder(returned_codewords.astype('int'))
    print("Fin décodage cyclique")
    decode_bits = ''.join(str(x) for x in decoded)
    # Conversion bitstring en bytes
    output = bitstring_to_bytes_p3(decode_bits)
    # Q7 : cf fin du TP1 !
    return output


def modulation_demodulation(codewords, h_enc_array, cyccode, valempirique, valempirique2, drawgraph=False, addError=False):
    M = [4, 16, 64, 256]
    colors = ['ro--', 'bo--', 'yo--', 'mo--', 'rs--', 'bs--',
              'ys--', 'ms--', 'rD-.', 'bD-.', 'yD-.', 'mD-.']
    EbSurN0 = range(13)  # en dB

    if drawgraph == True:
        code_test = [codewords, h_enc_array]
        for j in range(len(code_test)):
            for i in range(len(M)):
                modem = mod.QAMModem(M[i])
                modulated = modem.modulate(code_test[j])
                print("Calculs pour M = {}...".format(M[i]))
                for isnr in EbSurN0:
                    pid = os.fork()
                    if pid == 0:  # Processus Fils
                        k_mod = math.log2(M[i])
                        snr = isnr + 10*math.log10(k_mod)
                        bruited = digcom.cpx_awgn(modulated, snr, 1)
                        demodulated = modem.demodulate(bruited, 'hard')
                        while np.size(code_test[j]) < np.size(demodulated):
                            demodulated = demodulated[:-1]
                        while np.size(code_test[j]) > np.size(demodulated):
                            demodulated = np.append(demodulated, [0])
                        diff = abs(code_test[j] - demodulated)
                        unique, counts = np.unique(diff, return_counts=True)
                        result = dict(zip(unique, counts))
                        try:
                            result[1]
                        except:
                            result[1] = 0
                        # + 1e-8 Au cas où il y ai 0 erreurs pour ne pas casser l'affichage semilog
                        proba = result[1]/len(demodulated) + 1e-8
                        if j == 0:
                            # valempirique.append(proba)
                            valempirique[isnr] = proba
                            decoded = cyccode.cyclic_decoder(
                                demodulated.astype('int'))
                            diff2 = abs(h_enc_array - decoded)
                            unique2, counts2 = np.unique(
                                diff2, return_counts=True)
                            result2 = dict(zip(unique2, counts2))
                            try:
                                result2[1]
                            except:
                                result2[1] = 0
                            proba2 = result2[1]/len(demodulated) + 1e-8
                            # valempirique2.append(proba2)
                            valempirique2[isnr] = proba2
                        else:
                            # valempirique2.append(proba)
                            valempirique2[isnr] = proba
                        mysem.release()
                        sys.exit(0)
                for val in EbSurN0:
                    mysem.acquire()
                if j == 0:
                    plt.figure(1)
                    data1 = [i for i in valempirique]
                    for index in range(len(data1)):
                        print("Taux d'erreur sur une modulation avec M = {} et un Eb/N0 = {} : {} %".format(
                            M[i], index, round(data1[index]*100, 10)))
                        print("Taux d'erreur après décodage cyclique : {} %".format(
                            round(valempirique2[index]*100, 10)))
                    plt.plot(EbSurN0, data1,
                             colors[i], label="{}-QAM".format(M[i]))
                    mylabel = "avec"
                else:
                    mylabel = "sans"
                plt.figure(2)
                data2 = [i for i in valempirique2]
                plt.plot(EbSurN0, data2, colors[(
                    j+1)*4+i], label="{}-QAM {} cycccode".format(M[i], mylabel))

        # car il faut convertir en linéaire
        valtheorique = []
        for value in EbSurN0:
            valtheorique.append(1/2 * special.erfc(math.sqrt(10**(value/10))))
        print("4-QAM théorique : ", valtheorique)
        plt.figure(1)
        plt.plot(EbSurN0, valtheorique, 'g^-', label="4-QAM théorique")
        plt.legend(loc="lower left")
        plt.yscale('log')
        plt.figure(2)
        plt.legend(loc="lower left")
        plt.yscale('log')
        plt.show()

   # Cas général choisit pour la chaine M[1] = 16 points et EbSurN0[9] = 8 dB
    modem = mod.QAMModem(M[1])
    val = [-3, -1, 1, 3]
    points = np.array([complex(x, y)
                      for x in val for y in val])
    print("Debut modulation")
    modulated = modem.modulate(codewords)
    print("Fin modulation")
    if addError == True:
        snr = EbSurN0[9] + 10*math.log10(math.log2(M[1]))
        bruited = digcom.cpx_awgn(modulated, snr, 1)
        xpoint = points.real
        ypoint = points.imag
        x = bruited.real
        y = bruited.imag
        plt.figure(2)
        plt.plot(x, y, 'rX')
        plt.plot(xpoint, ypoint, 'b*')
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.show()
        print("Debut démodulation")
        demodulated = modem.demodulate(bruited, 'hard')
        print("Fin démodulation")
    else:
        print("Debut démodulation")
        demodulated = modem.demodulate(modulated, 'hard')
        print("Fin démodulation")

    while np.size(codewords) < np.size(demodulated):
        demodulated = demodulated[:-1]
    while np.size(codewords) > np.size(demodulated):
        demodulated = np.append(demodulated, [0])
    return demodulated


#----------------------- TP1 -----------------------#
# Q2 et 3
f = open("The_Adventures_of_Sherlock_Holmes_A_Scandal_In_Bohemia.txt", "r")
txt = f.read()

graphLettres(txt)

# Q4
huffman_codec_data = HuffmanCodec.from_data(txt)
# huffman_codec_data.print_code_table()
huff_code_table = huffman_codec_data.get_code_table()

# Q5
phrase = """Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyle

This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever.  You may copy it, give it away or
re-use it under the terms of the Project Gutenberg License included
with this eBook or online at www.gutenberg.net
"""

returned_txt = source_encode_decode(
    txt, valempirique, valempirique2, True,  True)

myinput = str(input("Souhaitez vous afficher le texte reçu ? (Y/n) > "))
accept = ["Y", "y", "yes", "YES", "Yes"]
if myinput in accept:
    print(returned_txt)

#h_enc_data = huffman_codec_data.encode(phrase)
#h_enc_data_str = ''.join(format(byte, '08b') for byte in h_enc_data)

# Q6
#h_dec_data = huffman_codec_data.decode(h_enc_data)

# Q7
# https://stackoverflow.com/questions/39718576/convert-a-byte-array-to-single-bits-in-a-array-python-3-5

mem = b'\x01\x02\xff'
x = np.fromstring(mem, dtype=np.uint8)
np.unpackbits(x).reshape(3, 8)

# Q8
"""
phrasehuff_str = ""
for letter in phrase:
    phrasehuff_str = phrasehuff_str + \
        homemade_huffman_encoder(huff_code_table[letter])
k = 4
Njam = k - len(phrasehuff_str) % k
phrasehuff_str = phrasehuff_str + '0'*Njam
phrasehuff_bit = bitstring_to_bytes_p3(phrasehuff_str)
"""

#----------------------- TP2 -----------------------#
# https://scikit-dsp-comm.readthedocs.io/en/latest/nb_examples/Block_Codes.html
# cf fonction channel_encode_decode


#----------------------- TP3 -----------------------#
# cf fonction modulation_demodulation
