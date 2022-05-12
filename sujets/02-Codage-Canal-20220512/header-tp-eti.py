
# General
import numpy as np

# Plotting
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 6)

# Huffman source compression
from dahuffman import HuffmanCodec

# Channel coding LBC - Linear Block Code, FEC - Forward Error Correction algos
from commpy.channelcoding import cyclic_code_genpoly
import sk_dsp_comm.fec_block as block # cyclic code is systematic

# Useful functions
# fct : String formatting to bytes, Python3 version using .to_bytes()
def bitstring_to_bytes_p3(s):
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')

# fct Nberr in a binary array
def nbiterr(bits, detected_bits):
    err_matrix = (bits + detected_bits) % 2
    nb_err = err_matrix.sum()
    # print error
    return nb_err

# Cyclic codes parameters
k4 = 4; n7 = 7
k11 = 11; n15 = 15
k26 = 26; n31 = 31

# Example : create generator polynomial, - to replace by yours !
genpoly_7_4_1 = format(cyclic_code_genpoly(7,4)[1],'b')

# Example (7,4) by default (1st poly), to replace by yours
cyccode = block.fec_cyclic(genpoly_7_4_1)

# Open the source file
with open('livre.txt') as fbook:
    phrase = fbook.read()

# Huffman encoder, and phrase encoding and analysis
"""
h_enc_phrase = 
...
"""

# Transform Huffman encoded phrase de 'byte array' into 'str' of (0,1)
h_enc_data_str = ''.join(format(byte, '08b') for byte in h_enc_phrase)
h_enc_arr = np.array(list(h_enc_data_str),dtype=int)


"""
dec_cc_arr = 
...
"""
# Transform 'np.array' into 'str'
decode_cc_str = ''.join(str(x) for x in dec_cc_arr)

"""
...
"""

# Transform 'str' of (0,1) back into byte array before Huffman reconstruction
demod_cc = bitstring_to_bytes_p3(decode_cc_str)

"""
...
"""
