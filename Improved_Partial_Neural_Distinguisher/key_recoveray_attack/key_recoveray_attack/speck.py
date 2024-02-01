import numpy as np
from os import urandom


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

# 生成训练数据的函数
# n对数
# nr轮数
# bit 选取出来的优势比特位或者是all
# pair 优势位
# diff 输入差分
# condition diff或者ct
def make_train_data(n, nr, bit, pair=2,diff=(0x80, 0x0),isct=True):
  bit = np.array(bit)
  bit = np.sort(bit)
  print(bit)
  if isct:
      bitall = np.concatenate((bit, [i + 32 for i in bit], [i + 64 for i in bit], [i + 64 + 32 for i in bit],[i + 128 - 8 for i in bit], [i + 128 - 8 + 32 for i in bit], [i + 192 - 8 for i in bit],[i + 192 - 8 + 32 for i in bit]))
      resultbitall = np.copy(bitall)
      for i in range(1, pair):
          bitall_plus_n = bitall + 128 * i
          resultbitall = np.concatenate((resultbitall, bitall_plus_n))
  else:
      bitall = np.concatenate((bit, [i + 32 for i in bit], [(i-8)%32+64for i in bit], [(i-8)%32+64+32for i in bit]))
      resultbitall = np.copy(bitall)
      for i in range(1, pair):
          bitall_plus_n = bitall + 64 * i
          resultbitall = np.concatenate((resultbitall, bitall_plus_n))
  print("产生的比特位是",resultbitall)

  # 生成明文并按照优势比特选取出来
  X = []
  Y = np.frombuffer(urandom(n), dtype=np.uint8)
  Y = Y & 1
  keys = np.frombuffer(urandom(4 * 3* n), dtype=np.uint32).reshape(3, n)
  ks = expand_key(keys, nr)
  ks_back = np.frombuffer(urandom(4*3), dtype=np.uint32).reshape(3, -1)
  ks_back = expand_key(ks_back, 1)
  ks_use_back = np.broadcast_to(ks_back, (1, n))
  for i in range(pair):
        plain0l = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        plain0r = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        plain1l = plain0l ^ diff[0]
        plain1r = plain0r ^ diff[1]
        num_rand_samples = np.sum(Y==0)
        plain1l[Y==0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
        plain1r[Y==0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32)
        ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
        ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
        ctdata0l2, ctdata0r2 = dec_one_round((ctdata0l, ctdata0r), ks_use_back)
        ctdata1l2, ctdata1r2 = dec_one_round((ctdata1l, ctdata1r), ks_use_back)
        # 这里表示是使用连接
        if isct:
            X += [ctdata0l,ctdata0r,ctdata1l,ctdata1r,ctdata0l2,ctdata0r2,ctdata1l2,ctdata1r2]
        else:
            X += [ctdata0l^ctdata1l,ctdata0r^ctdata1r,ctdata0l2^ctdata1l2,ctdata0r2^ctdata1r2]
        i += 1
  X = convert_to_binary(X)
  X = np.array(X[:, resultbitall])
  return(X,Y);