# -*- coding: utf-8 -*-
"""
Created on 13/04/2022
@author: Alexandre Drevet & Axel François
github : https://github.com/AxFrancois/Numeric-Comm

"""

# ----------------------- Imports ----------------------- #

import math
import multiprocessing as mp
import os
import random
import sys

import commpy.modulation as mod
import matplotlib.pylab as plt
import numpy as np
import sk_dsp_comm.digitalcom as digcom
import sk_dsp_comm.fec_block as block
from commpy.channelcoding import cyclic_code_genpoly
from dahuffman import HuffmanCodec
from scipy import special

from header_tp_eti import bitstring_to_bytes_p3

# ----------------------- Multiprocessing ----------------------- #

mysem = mp.Semaphore(0)  # Permet de synchroniser les processus
# Permet de créer 2 tableaux partagé par les processus. Taille égale au nombre de rapport signal à bruit à tester (oups le nombre magique)
valempirique = mp.Array('f', range(13))
valempirique2 = mp.Array('f', range(13))

# ----------------------- Functions ----------------------- #


def homemade_huffman_encoder(pTuple):
    """Encodeur Huffman n'utilisant pas HuffmanCodec.encode()

    Args:
            pTuple (tuple): tuple de la lettre obtenue par HuffmanCodec.get_code_table()

    Returns:
            str: bits de la lettre encodée
    """
    intInBin = bin(pTuple[1]).replace("0b", "")
    valren = "0" * (pTuple[0] - len(intInBin)) + intInBin
    return valren


def huffman_decoder_table(huff_code_table):
    """Fonction permetttant de générer la table de décodage Huffman à partir de la table d'encodage

    Args:
        huff_code_table (Dictionary): dictionnaire retourné par get_code_table()

    Returns:
        Dictionary: Dictionaire sous forme {"011" : "e", "01010001" : "z"...}
    """
    dic = {}
    # Boucle permettant de remplacer chaque caractère par sa valeur binaire
    for item in huff_code_table.items():
        intInBin = bin(item[1][1]).replace("0b", "")
        valbin = "0" * (item[1][0] - len(intInBin)) + intInBin
        dic[valbin] = item[0]
    return dic


def graphLettres(txt):
    """Fonction permettant de tracer la courbe de probabilité de chaque caractères et l'entropie du texte

    Args:
        txt (string): texte à analyser
    """
    # TP1 Q2
    char = []
    count = []
    # Boucle permettant de compter le nombre de répétition d'un caractère dans le texte
    for letter in txt:
        if letter in char:
            count[char.index(letter)] += 1
        else:
            char.append(letter)
            count.append(1)
    charArray = np.array(char)
    countArray = np.array(count)
    probaArray = countArray / len(txt)
    # Affichage des probabilités d'apparition des caractères dans le texte
    fig = plt.figure(0)
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
    ax.bar(charArray, probaArray)
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower left', ncol=2,
                        frameon=False, bbox_to_anchor=(0.12, 0.88))
    # Q3
    # Calcul de l'entropie de l'alphabet
    entropy = -np.sum(np.dot(probaArray, np.log2(probaArray)))
    print("Entropie calculée : ", entropy)


def source_encode_decode(txt, valempirique, valempirique2, drawgraph=False, addError=False):
    """Fonction du TP1, qui encode un texte grâce à huffman puis le transmet à l'étape codage cyclique, et le récupère pour le décodage huffman. Permet aussi de calculer les erreurs et de tracer les graphs

    Args:
        txt (string): texte à traiter
        valempirique (Array Multiprocessing): Array permettant de faire des graph dans la fonction du TP3 modulation_demodulation()
        valempirique2 (Array Multiprocessing): idem
        drawgraph (bool, optional): Permettant de choisir si l'on veut tracer les graphs ou non. Defaults to False.
        addError (bool, optional): Permettant de chosiir si l'on veut calculer les probabilités assosicées à la fonction. Defaults to False.

    Returns:
        string: texte post traitement après décodage Huffman
    """
    # TP1 Q4
    # Création et affichage de la table du codec Huffman de notre texte
    huffman_codec_data = HuffmanCodec.from_data(txt)
    print("Code table du codec Huffman : ")
    huffman_codec_data.print_code_table()
    print("\n")
    huff_code_table = huffman_codec_data.get_code_table()

    # TP1 Q5
    # Encodage Huffman du texte
    print("Debut encodage Huffman")
    h_enc_data = huffman_codec_data.encode(txt)
    h_enc_data_str = ''.join(format(byte, '08b') for byte in h_enc_data)
    print("Fin encodage Huffman")

    # Calcul du taux d'erreur sur un code Huffman bruité avec 1 erreur par bloc de 4 bits
    print("Calculs de probabilités : ", addError)
    # Condition permettant d'ajouter des erreurs dans le texte encodé
    if addError == True:
        h_enc_data_str_err = add_error(h_enc_data_str, 4)
        h_enc_data_err = bitstring_to_bytes_p3(
            h_enc_data_str_err)
        h_dec_err = huffman_codec_data.decode(h_enc_data_err)
        nbError = 0
        # Boucle permettant de calculer le pourcentage d'erreur ajouté dans le texte
        for i in range(min(len(h_dec_err), len(txt))):
            if h_dec_err[i] != txt[i]:
                nbError += 1
        print("Taux d'erreur sur un code huffman bruité avec 1 erreur par bloc de 4 bits :", round(
            nbError / max(len(h_dec_err), len(txt)) * 100, 4), "%.")

    # TP2 : encodage et décodage cyclique
    print("\n")
    returned_h_enc_data = channel_encode_decode(
        h_enc_data_str, valempirique, valempirique2, drawgraph, addError)

    # TP1 Q6
    # Décodage Huffman de notre texte encodé
    print("Debut décodage Huffman")
    h_dec_data = huffman_codec_data.decode(returned_h_enc_data)
    print("Fin décodage Huffman")

    return h_dec_data


def add_error(txt, k):
    """Fonction permettant d'ajouter des erreurs dans le texte par échange (0->1 et 1->0)

    Args:
        txt (string): texte à modifier sous forme binaire ("0001011011010")
        k (int): ajout d'une erreur par bloc de k bits

    Returns:
        string: texte modifié
    """
    # TP1 Q9
    error_txt = txt
    i = 0
    # Boucle ajoutant des erreurs dans le texte jusqu'à ce que les erreurs dépassent la taille du texte
    while i < len(error_txt):
        randnum = random.randint(0, k - 1)
        if error_txt[i + randnum] == '0':
            error_txt = error_txt[:i + randnum] + \
                '1' + error_txt[i + randnum + 1:]
        else:
            error_txt = error_txt[:i + randnum] + \
                '0' + error_txt[i + randnum + 1:]
        i += k
    return error_txt


def channel_encode_decode(ph_enc_data, valempirique, valempirique2, drawgraph=False, addError=False):
    """Fonction du TP2, qui fait l'encodage cyclique puis transmet à la modulation, et récupère la démodulation pour le décoage cylclique et retourne le résultat. Permet aussi de calculer des statistiques sur les erreurs.

    Args:
        ph_enc_data (string): chaine de caractère binaires issue de l'encodage Huffman
        valempirique (Array multiprocessing): Array permettant de faire des graph dans la fonction du TP3 modulation_demodulation()
        valempirique2 (Array multiprocessing): idem
        drawgraph (bool, optional): Permettant de choisir si l'on veut tracer les graphs ou non. Defaults to False.
        addError (bool, optional): Permettant de chosiir si l'on veut calculer les probabilités assosicées à la fonction. Defaults to False.

    Returns:
        Byte string: données issue du décodage cyclique dans le format adapté à la fonction du codec Huffman
    """
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
        # ajout successif de 1, 2 et 3 erreurs et test pour compter le taux d'erreur post décodage cyclique
        for err in range(1, 4):
            for i in range(err):
                codewords_str_err = add_error(codewords_str_err, 7)
            codewords_err = np.array(
                list(codewords_str_err), dtype=int).astype(int)
            decoded_err = cyccode.cyclic_decoder(codewords_err)
            decoded_err_bits_str = ''.join(str(x) for x in decoded_err)
            # Comptage du nombre d'erreurs
            nbError = 0
            for i in range(len(h_enc_str)):
                if h_enc_str[i] != decoded_err_bits_str[i]:
                    nbError += 1
            print("Taux d'erreur sur un code cyclique bruité avec {} erreur(s) par bloc de 7 bits :".format(
                err), round(nbError / len(codewords_str) * 100, 4), "%.")

    # TP3 : modulation et démodulation
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


def modulation_demodulation(codewords, h_enc_array, cyccode, valempirique, valempirique2, drawgraph=False,
                            addError=False):
    """Fonction du TP3, qui fait ma modulation et la démodulation et retourne le résultat. Permet de tracer les courbes de performance en fonction du rapport signal à bruit.

    Args:
        codewords (np.array): Données encodées par le code cyclique
        h_enc_array (string): Données encodées par Huffman (SANS CODE CYCLIQUE DONC)
        cyccode (FECCyclic): objet du code cylcique, utilisé pour le décodage pour les graphs
        valempirique (Array multiprocessing): Array permettant de faire des graph car partagée par plusieurs processus
        valempirique2 (Array multiprocessing): idem
        drawgraph (bool, optional): Permettant de choisir si l'on veut tracer les graphs ou non. Defaults to False.
        addError (bool, optional): Permettant de chosiir si l'on veut calculer les probabilités assosicées à la fonction. Defaults to False.

    Returns:
        np.array: données démodulées
    """
    M = [4, 16, 64, 256]  # Type de modulation, 4^n
    colors = ['ro--', 'bo--', 'yo--', 'mo--', 'rs--', 'bs--',
              'ys--', 'ms--', 'rD-.', 'bD-.', 'yD-.', 'mD-.']  # Couleurs pour les courbes
    EbSurN0 = range(13)  # rapport signal à bruit en dB

    # Condition et boucle permettant d'afficher les différentes figures demandées
    if drawgraph == True:
        code_test = [codewords, h_enc_array]
        for j in range(len(code_test)):
            for i in range(len(M)):
                modem = mod.QAMModem(M[i])
                modulated = modem.modulate(code_test[j])
                print("Calculs pour M = {}...".format(M[i]))
                for isnr in EbSurN0:
                    pid = os.fork()  # Utilisation de os.fork pour diminuer GRANDEMENT le temps de simulation
                    if pid == 0:  # Processus Fils, le père boucle et attend que tous les fils aient finis
                        k_mod = math.log2(M[i])
                        snr = isnr + 10 * math.log10(k_mod)
                        bruited = digcom.cpx_awgn(modulated, snr, 1)
                        demodulated = modem.demodulate(bruited, 'hard')

                        # Au cas où les messages soient de taille différentes, ce qui arrive pour M = 16 et 254
                        while np.size(code_test[j]) < np.size(demodulated):
                            demodulated = demodulated[:-1]
                        while np.size(code_test[j]) > np.size(demodulated):
                            demodulated = np.append(demodulated, [0])

                        # Calcul des erreurs
                        diff = abs(code_test[j] - demodulated)
                        unique, counts = np.unique(diff, return_counts=True)
                        result = dict(zip(unique, counts))
                        try:
                            result[1]
                        except:
                            result[1] = 0
                        # + 1e-8 Au cas où il y ai 0 erreurs pour ne pas casser l'affichage semilog
                        proba = result[1] / len(demodulated) + 1e-8

                        # si on travaille avec les données encodé cyclique alors il faut aussi les décodé cylcique pour pour la deuxième figure
                        if j == 0:
                            # valempirique.append(proba)
                            valempirique[isnr] = proba
                            # décodage cyclique
                            decoded = cyccode.cyclic_decoder(
                                demodulated.astype('int'))
                            # calcul du second taux d'erreur
                            diff2 = abs(h_enc_array - decoded)
                            unique2, counts2 = np.unique(
                                diff2, return_counts=True)
                            result2 = dict(zip(unique2, counts2))
                            try:
                                result2[1]
                            except:
                                result2[1] = 0
                            proba2 = result2[1] / len(demodulated) + 1e-8
                            # valempirique2.append(proba2)
                            valempirique2[isnr] = proba2
                        else:
                            # si on travaille directement sans encodage cyclique alors pas besoin
                            # valempirique2.append(proba)
                            valempirique2[isnr] = proba
                        # libération d'un sémaphore lors de la fin du process et fermeture du process
                        mysem.release()
                        sys.exit(0)
                # acquisition des sémaphore, autant que le nombre de rapport signal à bruit à tester
                for val in EbSurN0:
                    mysem.acquire()
                # tracé des courbes
                if j == 0:
                    # courbe des performance de la modulation selon le M et le rapport signal à bruit
                    plt.figure(1)
                    data1 = [i for i in valempirique]
                    for index in range(len(data1)):
                        print("Taux d'erreur sur une modulation avec M = {} et un Eb/N0 = {} : {} %".format(
                            M[i], index, round(data1[index] * 100, 10)))
                        print("Taux d'erreur après décodage cyclique : {} %".format(
                            round(valempirique2[index] * 100, 10)))
                    plt.plot(EbSurN0, data1,
                             colors[i], label="{}-QAM".format(M[i]))
                    mylabel = "avec"
                else:
                    mylabel = "sans"
                # Courbe des performance de la chaine en l'absence ou en présence de l'encodage cyclique
                plt.figure(2)
                data2 = [i for i in valempirique2]
                plt.plot(EbSurN0, data2, colors[(
                    j + 1) * 4 + i],
                    label="{}-QAM {} cycccode".format(M[i], mylabel))

        # Conversion en linéaire pour la valeur théorique du 4-QAM
        valtheorique = []
        for value in EbSurN0:
            valtheorique.append(
                1 / 2 * special.erfc(math.sqrt(10 ** (value / 10))))
        print("4-QAM théorique : ", valtheorique)
        plt.figure(1)
        plt.plot(EbSurN0, valtheorique, 'g^-', label="4-QAM théorique")
        # légendes, échelle log et affichage des courbes
        plt.legend(loc="lower left")
        plt.yscale('log')
        plt.figure(2)
        plt.legend(loc="lower left")
        plt.yscale('log')
        plt.show()

    # Cas général choisit pour la chaine M[1] = 16 points et EbSurN0[9] = 8 dB
    modem = mod.QAMModem(M[1])
    val = [-3, -1, 1, 3]
    # Points permettant de tracer la constallation
    points = np.array([complex(x, y)
                       for x in val for y in val])
    # Modulation des différents mots-codes
    print("Debut modulation")
    modulated = modem.modulate(codewords)
    print("Fin modulation")
    # Calcul du SNR et affichage de la constellation de la modulation en fonction des erreurs ajoutées
    if addError == True:
        snr = EbSurN0[9] + 10 * math.log10(math.log2(M[1]))
        bruited = digcom.cpx_awgn(modulated, snr, 1)
        # tracé de la constellation
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
        # Démodulation pour un signal modulé avec des erreurs
        print("Debut démodulation")
        demodulated = modem.demodulate(bruited, 'hard')
        print("Fin démodulation")
    else:
        # Démodulation pour un signal modulé sans erreurs
        print("Debut démodulation")
        demodulated = modem.demodulate(modulated, 'hard')
        print("Fin démodulation")

    # Au cas où les messages soient de taille différentes, ce qui arrive pour M = 16 et 254
    while np.size(codewords) < np.size(demodulated):
        demodulated = demodulated[:-1]
    while np.size(codewords) > np.size(demodulated):
        demodulated = np.append(demodulated, [0])

    return demodulated


# ----------------------- TP1 ----------------------- #
# Q2 et 3 : ouverture du texte et tracé de sa fréquence de symboles
f = open("The_Adventures_of_Sherlock_Holmes_A_Scandal_In_Bohemia.txt", "r")
txt = f.read()

graphLettres(txt)

# Q5
# Bout de texte pris permettant de tester notre programme afin de ne pas tester nos fonctions sur tout le texte faisant perdre du temps
phrase = """Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyle

This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever.  You may copy it, give it away or
re-use it under the terms of the Project Gutenberg License included
with this eBook or online at www.gutenberg.net
"""

# lancement de la chaine de traitement totale
returned_txt = source_encode_decode(
    txt, valempirique, valempirique2, True, True)

myinput = str(input("Souhaitez vous afficher le texte reçu ? (Y/n) > "))
accept = ["Y", "y", "yes", "YES", "Yes"]
# Affichage du texte reçu demandé par l'utilisateur
if myinput in accept:
    print(returned_txt)


# ---------Autres bouts de code pas utilisés --------- #

# h_enc_data = huffman_codec_data.encode(phrase)
# h_enc_data_str = ''.join(format(byte, '08b') for byte in h_enc_data)

# Q6
# h_dec_data = huffman_codec_data.decode(h_enc_data)

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

# ----------------------- TP2 ----------------------- #
# https://scikit-dsp-comm.readthedocs.io/en/latest/nb_examples/Block_Codes.html
# cf fonction channel_encode_decode


# ----------------------- TP3 ----------------------- #
# cf fonction modulation_demodulation
