import numpy as np
from os import urandom
# 全局变量b
B = [0,1,2,3,4,5,6,7,8,9,10]

def WORD_SIZE():
    return(32)


def ALPHA():
    return(8)


def BETA():
    return(3)


MASK_VAL = 2 ** WORD_SIZE() - 1

def rol(x, k):
    return (((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))


def ror(x, k):
    return ((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))


def enc_one_round(p, k):
    c0, c1 = p[0], p[1]
    c0 = ror(c0, ALPHA())
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rol(c1, BETA())
    c1 = c1 ^ c0
    return(c0,c1)


def dec_one_round(c,k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, BETA())
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rol(c0, ALPHA())
    return(c0, c1)

def dec_one_round_for_random(c,k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, BETA())
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    return(c0, c1)


def expand_key(k, t):
    ks = [0 for i in range(t)]
    ks[0] = k[len(k) - 1]
    l = list(reversed(k[:len(k) - 1]))
    tmp = len(l)
    for i in range(t - 1):
        l[i % tmp], ks[i + 1] = enc_one_round((l[i % tmp], ks[i]), i)
    return ks


def encrypt(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round((x,y), k)
    return(x, y)


def decrypt(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k)
    return(x,y)

# 验证两种格式的
def check_testvector():
    key = (0x13121110, 0x0b0a0908, 0x03020100)
    pt = (0x74614620, 0x736e6165)
    ks = expand_key(key, 26)
    ct = encrypt(pt, ks)
    if ct == (0x9f7952ec, 0x4175946c):
        print('Testvector of speck64/96 verified.')
    else:
        print('Testvector of speck64/96 not verified.')
        return False

    key = (0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100)
    pt = (0x3b726574, 0x7475432d)
    ks = expand_key(key, 27)
    ct = encrypt(pt, ks)
    if ct == (0x8c6fa548, 0x454e028b):
        print('Testvector of speck64/128 verified.')
    else:
        print('Testvector of speck64/128 not verified.')
        return False

    return True


#convert_to_binary takes as input an array of ciphertext pairs
#where the first row of the array contains the lefthand side of the ciphertexts,
#the second row contains the righthand side of the ciphertexts,
#the third row contains the lefthand side of the second ciphertexts,
#and so on
#it returns an array of bit vectors containing the same data
def convert_to_binary(l):
    n = len(l)
    k = WORD_SIZE() * n
    X = np.zeros((k, len(l[0])), dtype=np.uint8)
    for i in range(k):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - 1 - i % WORD_SIZE()
        X[i] = (l[index] >> offset) & 1
    X = X.transpose()
    return(X)

# 用来取出指定位置的比特并成为新的密钥
# def extract_bits(decimal_number):
#     # 将十进制数转为32位的二进制数
#     binary_repr = np.binary_repr(decimal_number, width=32)
#
#     # 从32位的二进制数中提取第2到第12位
#     extracted_bits = binary_repr[10:22]
#
#     # 将提取的位序列转为十进制数并返回
#     return int(extracted_bits, 2)
def mask_bits(a: int) -> int:
    # 函数内部直接使用全局变量B

    mask = ['0'] * 32

    for bit in B:  # 使用全局变量B
        if 0 <= bit < 32:
            mask[bit] = '1'

    int_mask = int(''.join(mask), 2)

    result = a & int_mask

    return result
# 生成训练数据的函数
# n对数
# nr轮数
# bit 选取出来的优势比特位或者是all
# pair 优势位
# diff 输入差分
# condition diff或者ct
def make_train_data(n, nr, bit, pair=2,diff=(0x80, 0x0)):
  bit = np.array(bit)
  bit = np.sort(bit)
  bitall = np.concatenate(([(i + 3) % 32 for i in bit], [(i + 3) % 32 + 32 for i in bit], [i + 64 for i in bit],[i + 96 for i in bit],[i + 128 for i in bit],[i + 160 for i in bit]))
  Y = np.frombuffer(urandom(n), dtype=np.uint8);
  Y = Y & 1;
  Y1 = np.tile(Y, pair);
  keys = np.frombuffer(urandom(4*3*n), dtype=np.uint32).reshape(3, -1)
  keys = np.tile(keys, pair);
  plain0l = np.frombuffer(urandom(4 * n* pair), dtype=np.uint32)
  plain0r = np.frombuffer(urandom(4 * n* pair), dtype=np.uint32)
  plain1l = plain0l ^ diff[0];
  plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y1 == 0);
  plain1l[Y1 == 0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
  plain1r[Y1 ==0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
  # 这里先将密码扩展成为n+1轮，然后
  ks = expand_key(keys, nr+1);
  ks = np.array(ks)
  last_key = ks[-1, :].reshape(1, -1)

  # 创建一个掩码，其中4到24位为0，其它位为1
  # mask = ((2**7 - 1) | (~0 << 28)) & 0xFFFFFFFF
  #
  # # 使用按位与操作将4到24位替换为0
  # last_key_extract = last_key & mask
  # 将密钥转换成为32位的2进制，将32位中的非指定位都转换成为0
  vfunc = np.vectorize(mask_bits)
  last_key_extract = vfunc(last_key)
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  ctdata0l, ctdata0r = dec_one_round((ctdata0l, ctdata0r), last_key_extract);
  ctdata1l, ctdata1r = dec_one_round((ctdata1l, ctdata1r), last_key_extract);
  ctdata0l = ctdata0l.squeeze()
  ctdata1l = ctdata1l.squeeze()
  R0 = ror(ctdata0l ^ ctdata0r, BETA())
  R1 = ror(ctdata1l ^ ctdata1r, BETA())
  X = [R0, R1, ctdata0l, ctdata0r, ctdata1l, ctdata1r]
  X = np.array(X)
  X = convert_to_binary(X);
  X = X[:,bitall]
  X = X.reshape(pair, n, len(bitall)).transpose((1, 0, 2))
  X = X.reshape(n, 1, -1)
  X = np.squeeze(X)
  return(X,Y);