import numpy as np
from os import urandom
import binascii
import pandas as pd


def WORD_SIZE():
    return(48)


def ALPHA():
    return(8)


def BETA():
    return(3)


MASK_VAL = 2 ** WORD_SIZE() - 1

def shuffle_together(l):
    state = np.random.get_state()
    for x in l:
        np.random.set_state(state)
        np.random.shuffle(x)

def rol(x, k):
    x = x & 0x0000ffffffffffff      # set 48~63 bits to 0
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))


def ror(x, k):
    x = x & 0x0000ffffffffffff      # set 48~63 bits to 0
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))


def enc_one_round(p, k):
    c0, c1 = p[0], p[1]
    c0 = ror(c0, ALPHA())
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rol(c1, BETA())
    c1 = c1 ^ c0
    return(c0, c1)


def dec_one_round(c,k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, BETA())
    c0 = c0 ^ k
    # 代码是没有错的，加密出来的结果就是48位
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
        x, y = enc_one_round((x, y), k)
    return (x, y)


def decrypt(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round((x, y), k)
    return (x, y)


def check_testvector():
    key = (0x0d0c0b0a0908, 0x050403020100)
    pt = (0x65776f68202c, 0x656761737520)
    ks = expand_key(key, 28)
    ct = encrypt(pt, ks)
    if (ct == (0x9e4d09ab7178, 0x62bdde8f79aa)):
        print("Testvector verified.")
        return(True)
    else:
        print("Testvector not verified.")
        return(False)


def convert_to_binary(arr):
    X = np.zeros((len(arr) * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(len(arr) * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X

def generate_mask(a):
    mask = ['0'] * 48

    for bit in a:  # 使用全局变量B
        if 0 <= bit < 48:
            mask[bit] = '1'

    int_mask = int(''.join(mask), 2)
    return int_mask
def make_train_data(n, nr, diff, bit=[]):
    bit = np.array(bit)
    bit = np.sort(bit)
    bitall = np.concatenate((bit,[i + 48 for i in bit], [i + 96 for i in bit], [i + 96+48 for i in bit]))
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
    keys = np.frombuffer(urandom(16*n), dtype=np.uint64).reshape(2, -1)
    plain0l = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    plain0r = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y == 0)
    plain1l[Y == 0] = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
    plain1r[Y == 0] = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
    ks = expand_key(keys, nr+1)
    ks = np.array(ks)
    last_key = ks[-1, :].reshape(1, -1)
    last_key = np.squeeze(last_key)
    a = [(i + 8) % 48 for i in bit]
    mask = generate_mask(a)
    last_key_extract = np.bitwise_and(last_key, mask)
    c0l, c0r = encrypt((plain0l, plain0r), ks)
    c1l, c1r = encrypt((plain1l, plain1r), ks)
    c0l, c0r = dec_one_round((c0l, c0r), last_key_extract);
    c1l, c1r = dec_one_round((c1l, c1r), last_key_extract);
    X = convert_to_binary([c0l, c0r, c1l, c1r])
    X = X[:, bitall]
    return (X, Y)


if __name__ == '__main__':
    check_testvector()
    # X, Y = make_train_data(10 ** 6, 3, diff=(0x800000, 0))
    # X_front = X[:, :64]
    # X_back = X[:, 64:]
    # print(X_back)
    # X_xor = np.logical_xor(X_front, X_back)
    # X_int = X_xor.astype(int)
    # # 将二维 numpy 数组转换为 DataFrame
    # df = pd.DataFrame(X_int)
    #
    # # 将每行转换为元组，然后统计每个元组的出现次数
    # result = df.apply(tuple, axis=1).value_counts()
    # result = result.head(5)
    # for index, count in result.items():
    #     binary_string = "".join(str(x) for x in index)  # 将每个数组转换为二进制字符串
    #     hex_string = binascii.hexlify(
    #         bytes(int(binary_string[i: i + 8], 2) for i in range(0, len(binary_string), 8)))  # 将二进制字符串转换为十六进制表示
    #     print(f"Array: {hex_string}\nCount: {count}\n")