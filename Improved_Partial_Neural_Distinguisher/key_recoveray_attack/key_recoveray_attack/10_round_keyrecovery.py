import numpy as np
import speck as sp
from os import urandom
from keras.models import load_model
import time
import itertools
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

WORD_SIZE = sp.WORD_SIZE()
MASK_VAL = 2**WORD_SIZE - 1
master_key_bit_length = 96

# 封装函数，用来检测分数
def score_test(ct0_t,ct1_t,ct2_t,ct3_t,bits,nets):
    R0 = sp.ror(ct0_t ^ ct1_t, sp.BETA())
    R1 = sp.ror(ct2_t ^ ct3_t, sp.BETA())
    X = [R0, R1, ct0_t, ct1_t, ct2_t, ct3_t]
    X = np.array(X)
    X = sp.convert_to_binary(X)
    resultbitall = extra_info_bit(bits)
    X = np.array(X[:, resultbitall])
    X = X.reshape(-1, 16 * 12 * 6)
    Z = nets.predict(X, batch_size=1)
    T = nets.predict(X, batch_size=1)
    print(Z)
    z = np.log2(Z / (1 - Z))
    t = np.log2(T / (1 - T))
    print(z)
    # 0表示列方向之和
    s = np.sum(z, axis=0)
    s_t = np.sum(t, axis=0)
    print("最终这个结构生成的分数是")
    print(s)
    print("第二次测试这个结构生成的分数是")
    print(s_t)

# 本函数用来将密钥最低十位的十进制数表示出来
def extract_lowest_10_bits(n):
    # 将数字转换为二进制并去掉前两个字符（'0b'）
    binary_str = bin(n)[2:]

    # 取出最低的10位
    last_10_bits = binary_str[-10:]

    # 将10位的二进制转换回10进制
    decimal_result = int(last_10_bits, 2)

    return decimal_result

# 用来选择分数最高的k个密钥
def select_top_k_candidates(sur_kg, kg_scores, k=3):
    num = len(sur_kg)
    tp = kg_scores.copy()
    tp.sort(reverse=True)
    # print('the top scores are ', tp[:5])
    if num > k:
        base = tp[k]
    else:
        return sur_kg, kg_scores
    filtered_subkey = []
    filtered_score = []
    for i in range(num):
        if kg_scores[i] > base:
            filtered_subkey.append(sur_kg[i])
            filtered_score.append(kg_scores[i])
    return filtered_subkey, filtered_score

# 获取神经区分器的优势比特
def extra_info_bit(bit):
    # return np.concatenate((bit,[i+32 for i in bit]))
    bit = np.array(bit)
    bit = np.sort(bit)
    return np.concatenate(([(i + 3) % 32 for i in bit], [(i + 3) % 32 + 32 for i in bit], [i + 64 for i in bit],[i + 96 for i in bit],[i + 128 for i in bit],[i + 160 for i in bit]))

# 生成挑战
def gen_challenge(pt0,pt1,key,diff=(0x48000, 0x80), neutral_bits=[35,34,33,32,31,30]):
    pt0a, pt1a, pt0b, pt1b = make_plaintext_structure(pt0, pt1, diff=diff, neutral_bits=neutral_bits)
    pt0a, pt1a = sp.dec_one_round((pt0a, pt1a), 0)
    pt0b, pt1b = sp.dec_one_round((pt0b, pt1b), 0)
    ct0a, ct1a = sp.encrypt((pt0a, pt1a), key)
    ct0b, ct1b = sp.encrypt((pt0b, pt1b), key)
    return (ct0a, ct1a, ct0b, ct1b)


def extract_bits(decimal_number,a,b):
    # 将十进制数转为32位的二进制数
    binary_str = format(decimal_number, '032b')

    # 从32位的二进制数中提取第2到第12位
    extracted_bits = binary_str[a:b]
    print(extracted_bits)

    # 将提取的位序列转为十进制数并返回
    return int(extracted_bits, 2)

def attack_with_one_nd(cts, kg_bit_num, kg_offset, sur_kg_low, net, bits, c,tk):
    sur_kg = []
    sur_kg_socres = []
    kg_batch = 2**kg_bit_num
    c0l, c0r, c1l, c1r = cts[0], cts[1], cts[2], cts[3]
    if kg_offset==None:
        key_bit = mask_bits(tk)
        # print("密钥阶段猜测的分数", s1)
        d0l_t, d0r_t = sp.dec_one_round((c0l, c0r), key_bit)
        d1l_t, d1r_t = sp.dec_one_round((c1l, c1r), key_bit)
        R0 = sp.ror(d0l_t ^ d0r_t, sp.BETA())
        R1 = sp.ror(d1l_t ^ d1r_t, sp.BETA())
        X = [R0, R1, d0l_t, d0r_t, d1l_t, d1r_t]
        X = np.array(X)
        X = sp.convert_to_binary(X)
        resultbitall = extra_info_bit(bits)
        T = np.array(X[:, resultbitall])
        X = T.reshape(-1, 16 * len(bits) * 6)
        Z1 = net.predict(X, batch_size=10000)
        Z1 = np.log2(Z1 / (1 - Z1))
        Z1 = Z1.reshape(-1, 64)
        s2 = np.sum(Z1, axis=1)
        print("进入密钥猜测之后的评估", s2)
        n = len(c0l)
        # 扩展成猜测数目额数量
        kg_high = np.arange(kg_batch, dtype=np.uint32)
        vec_rotate_left = np.vectorize(rotate_left)
        kg_high = vec_rotate_left(kg_high, 24)  # 创建向量化函数
    else:
        key_bit = extract_bits(tk,bits[0]+8,bits[-1]+9)
        key_bit = key_bit << kg_offset
        d0l_t, d0r_t = sp.dec_one_round((c0l, c0r),  key_bit)
        d1l_t, d1r_t = sp.dec_one_round((c1l, c1r),  key_bit)
        R0 = sp.ror(d0l_t ^ d0r_t, sp.BETA())
        R1 = sp.ror(d1l_t ^ d1r_t, sp.BETA())
        X = [R0, R1, d0l_t, d0r_t, d1l_t, d1r_t]
        X = np.array(X)
        X = sp.convert_to_binary(X)
        resultbitall = extra_info_bit(bits)
        T = np.array(X[:, resultbitall])
        X = T.reshape(-1, 16 * len(bits) * 6)
        Z1 = net.predict(X, batch_size=5000)
        Z1 = np.log2(Z1 / (1 - Z1))
        Z1 = Z1.reshape(-1, 64)
        s2 = np.sum(Z1, axis=1)
        print("进入密钥猜测之后的评估",s2)
        n = len(c0l)
        kg_high = np.arange(kg_batch, dtype=np.uint32)
        kg_high = kg_high << kg_offset
    # 扩展成猜测数目额数量
    c0l, c0r, c1l, c1r = np.tile(c0l, kg_batch), np.tile(c0r, kg_batch), np.tile(c1l, kg_batch), np.tile(c1r, kg_batch)
    # # 就是这里出现了问题，因为这里猜测的不是全部的密钥，所以没办法用来
    # kg_high = np.array([key_bit])
    print(kg_high)
    kg = kg_high; key_guess = np.repeat(kg, 16*64)
    d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
    d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
    R0 = sp.ror( d0l^d0r, sp.BETA())
    R1 = sp.ror(d1l^d1r, sp.BETA())
    X = [R0, R1,  d0l, d0r, d1l, d1r]
    X = np.array(X)
    X = sp.convert_to_binary(X)
    resultbitall = extra_info_bit(bits)
    print(resultbitall)
    K = np.array(X[:, resultbitall])
    X = K.reshape(-1, 16 * len(bits) * 6)
    Z = net.predict(X, batch_size=10000)
    Z = np.log2(Z / (1 - Z))
    Z = Z.reshape(-1,64)

    s = np.sum(Z, axis=1)
    print(s)
        # 这里有问题，好像是对不上的
    for i in range(kg_batch):
        if s[i] > c:
            sur_kg.append(i)
            sur_kg_socres.append(s[i])
    return sur_kg, sur_kg_socres
def mask_bits(a: int) -> int:
    B = [29, 30, 31, 0, 1, 2, 3, 4, 5, 6,7, 8]
    # 函数内部直接使用全局变量B
    binary_a = format(a, '032b')

    mask = ['0'] * 32

    for bit in B:  # 使用全局变量B
        if 0 <= bit < 32:
            mask[bit] = '1'

    int_mask = int(''.join(mask), 2)

    result = a & int_mask

    return result

def rotate_left(n, r):
    r %= 32
    return (n << r) & 0xFFFFFFFF | ((n & 0xFFFFFFFF) >> (32 - r))
# 生成密文结构
def collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks):
    p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
    p1l, p1r = sp.dec_one_round((p1l, p1r), 0)
    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)

    return c0l, c0r, c1l, c1r
# 检测生成的这个结构每个对是否符合神经网络的输入差分
def isTrueStructure(p0l, p0r, p1l, p1r,ks,diff):
    p0l1, p0r1 = sp.dec_one_round((p0l, p0r), 0)
    p1l1, p1r1 = sp.dec_one_round((p1l, p1r), 0)
    c0l1, c0r1 = sp.encrypt((p0l1, p0r1), [ks[0][0], ks[1][0]])
    c1l1, c1r1 = sp.encrypt((p1l1, p1r1), [ks[0][0], ks[1][0]])
    diffl = c0l1 ^ c1l1
    print("差分是",diffl)
    diffr = c0r1 ^ c1r1
    d0 = (diffl == diff[0])
    d1 = (diffr == diff[1])
    d = d0 * d1
    return d

# 将一对明文按照中性比特扩展成为结构
def make_plaintext_structure(p0l,p0r,diff=(0x48000, 0x80), neutral_bits=[39,38,37,36]):
    p0l = p0l
    p0r = p0r
    for i in neutral_bits:
        if isinstance(i, int):
            i = [i]
        d0 = 0
        d1 = 0
        for j in i:
            d = 1 << j
            d0 |= d >> WORD_SIZE
            d1 |= d & MASK_VAL
        p0l = np.concatenate([p0l, p0l ^ d0])
        p0r = np.concatenate([p0r, p0r ^ d1])
    p1l = p0l ^ diff[0]
    p1r = p0r ^ diff[1]
    return p0l, p0r, p1l, p1r

# 用来生成需要数量的明文对，这里先默认就只生成一对
def gen_plain(n=1):
    pt0 = np.frombuffer(urandom(4), dtype=np.uint32)
    pt1 = np.frombuffer(urandom(4), dtype=np.uint32)
    return (pt0, pt1)
def int_to_bin(x, width):
    """将整数转换为指定宽度的二进制字符串，如果宽度不足，用0填充最高位"""
    return format(x, 'b').zfill(width)

def construct_array(a, b, c):
    # 将输入转换为指定宽度的二进制字符串
    bin_a = int_to_bin(a, 12)
    bin_b = int_to_bin(b, 11)
    bin_c = int_to_bin(c, 11)
    # 根据规则拼接二进制字符串
    d = bin_b[3:] + bin_c + bin_a[1:] + bin_b[1:3]
    return d

def hamming_distance(a, b):
    # 将a转换为32位的二进制字符串
    a_bin = format(a, '032b')
    b = format(b, '032b')
    # 计算汉明距离
    distance = sum(bit1 != bit2 for bit1, bit2 in zip(a_bin, b))
    return distance
def final_select_one_key(cts,final_keys,net):
    keynum = len(final_keys)
    c0l, c0r, c1l, c1r = cts[0], cts[1], cts[2], cts[3]
    key_guess = np.repeat(final_keys, 16 * 64)
    c0l, c0r, c1l, c1r = np.tile(c0l, keynum), np.tile(c0r, keynum), np.tile(c1l, keynum), np.tile(c1r, keynum)
    d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
    d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
    X = sp.convert_to_binary([d0l, d0r, d1l, d1r])
    Z = net.predict(X, batch_size=1000)
    Z = np.log2(Z / (1 - Z))
    Z = Z.reshape(-1,1024)
    s = np.sum(Z, axis=1)
    index = np.argmax(s)
    return final_keys[index]


def attack_with_dual_NDs(t, nr, diffs, NBs,mulpair_NB, nds, bits, c, k):
    #
    assert master_key_bit_length % WORD_SIZE == 0
    m = master_key_bit_length // WORD_SIZE
    assert m == 3 or m == 4
    nets = []
    for nd in nds:
        nets.append(load_model(nd))
    acc = 0
    time_consumption = np.zeros(t)
    data_consumption = np.zeros(t, dtype=np.uint32)
    for i in range(t):
        print('attack index: {}'.format(i))
        data_num = 0
        start = time.time()
        key = np.frombuffer(urandom(m * 4), dtype=np.uint32).reshape(m, 1)
        ks = sp.expand_key(key, nr)
        tk = ks[-1][0]
        print("产生的最后一轮子密钥是：", tk)
        k1,k2,k3 = [],[],[]
        selected_ct0,selected_ct1,selected_ct2,selected_ct3 = [],[],[],[]
        # stage 1
        # 这里表示如果生成16个结构还是没有成功的话就表示攻击失败
        num = 0
        while True:
            if num >= 2 ** 6:
                num = -1
                break
            pt0, pt1 = gen_plain()
            p0l, p0r, p1l, p1r= make_plaintext_structure(pt0, pt1,diff=diffs[0], neutral_bits=NBs[0])
            structureBool = isTrueStructure(p0l, p0r, p1l, p1r, ks, diff=(0x8000, 0))
            print(structureBool)
            # 将明文结构生成多对,并加密成为密文
            ct0, ct1, ct2, ct3 = gen_challenge(p0l, p0r, ks,diff=(0x900000, 0x1000), neutral_bits=mulpair_NB[0])
            sur_kg_1, kg_scores_1 = attack_with_one_nd([ct0, ct1, ct2, ct3], 12, 2, None,nets[0], bits[0], c[0],tk)
            print('猜测的密钥是',sur_kg_1, kg_scores_1)
            kg_1, kg_scores_1 = select_top_k_candidates(sur_kg_1, kg_scores_1, k[0])
            print("猜测的密钥是：", kg_1)
            print("猜测的分数是：", kg_scores_1)
            k1 = kg_1
            num += 1
            data_num += 1
            if len(sur_kg_1) == 0:
                print('\r {} plaintext structures generated'.format(num), end='')
                continue
            else:
                print('')
                print('Stage 1: ', len(sur_kg_1), ' subkeys survive')
                break
        if num == -1:
            print(' ')
            print('this trial fails.')
            print('{} plaintext structures are generated.'.format(data_num))
            print('the time consumption is ', time.time() - start)
            continue
        # stage 2 找的差分路径和以及这条差分路径上概率为一的中性比特
        # 现在中性比特和路径都找好了，可以开始写了
        num = 0
        while True:
            if num >= 2 ** 6:
                num = -1
                break
            # 生成明文
            pt0, pt1 = gen_plain()
            # 使用4个中性比特讲明文扩展成维明文结构
            p0l, p0r, p1l, p1r = make_plaintext_structure(pt0, pt1, diff=diffs[1], neutral_bits=NBs[1])
            # 看看这个结构是不是正确的结构，本函数只是用来辅助测试
            print("运行到了这里")
            structureBool = isTrueStructure(p0l, p0r, p1l, p1r, ks, diff=(0x80,0))
            print(structureBool)
            # 将明文结构生成多对,并加密成为密文
            ct0, ct1, ct2, ct3 = gen_challenge(p0l, p0r, ks, diff=(0x9000, 0x10), neutral_bits=mulpair_NB[1])
            selected_ct0, selected_ct1, selected_ct2, selected_ct3 = ct0, ct1, ct2, ct3
            # 正确的实验证明，如果是正确的对的话，得到的分数应该是接近的，同时每个结构都应该是满足的>10分的，所以应该是以下代码出现了问题
            # 猜测密钥
            # 猜测不了正确的密钥就两个原因，密钥的偏转不正确，还有就是密码的选取不正确
            # 要解决的问题是，即使有时候这个结构能取得很高的分数，但是在密钥攻击的时候却不能将这个对推荐出来，分数不对，代码还是存在问题，和结构分数不一致,即使猜出的是正确的对，但是仍然是偏低的分数
            sur_kg_2, kg_scores_2 = attack_with_one_nd([ct0, ct1, ct2, ct3], 11, None, None, nets[1], bits[1], c[1],tk)
            print('猜测的密钥是', sur_kg_2, kg_scores_2)
            # 按理说，分数应该是一样的，但是却总是负分，需要找到这个bug,应该是结构有问题但是问题究竟在哪里
            # 分数还是偏低的，要准备
            kg_2, kg_scores_2 = select_top_k_candidates(sur_kg_2, kg_scores_2, k[0])
            print("猜测的密钥是：", kg_2)
            print("猜测的分数是：", kg_scores_2)
            k2 = kg_2
            num += 1
            data_num += 1
            if len(sur_kg_2) == 0:
                print('\r {} plaintext structures generated'.format(num), end='')
                continue
            else:
                print(' ')
                print('Stage 2: ', len(sur_kg_2), ' subkeys survive')
                break
        if num == -1:
            print(' ')
            print('this trial fails.')
            print('{} plaintext structures are generated.'.format(data_num))
            print('the time consumption is ', time.time() - start)
            continue

        # 第三阶段，在前面两个阶段都正确的情况下进行第三阶段
        # 最后完成这个阶段的密钥恢复，是否有更好的输入差分的选取
        num = 0
        while True:
            if num >= 2 ** 6:
                num = -1
                break
            # 生成明文
            pt0, pt1 = gen_plain()
            # 使用4个中性比特讲明文扩展成明文结构
            # 这里中性笔特
            p0l, p0r, p1l, p1r = make_plaintext_structure(pt0, pt1, diff=diffs[2], neutral_bits=NBs[2])
            # 看看这个结构是不是正确的结构，本函数只是用来辅助测试
            print("运行到了这里")
            structureBool = isTrueStructure(p0l, p0r, p1l, p1r, ks, diff=(0x800000,0))
            print(structureBool)
            # 将明文结构生成多对,并加密成为密文
            ct0, ct1, ct2, ct3 = gen_challenge(p0l, p0r, ks, diff=(0x90000000, 0x100000), neutral_bits=mulpair_NB[2])
            sur_kg_3, kg_scores_3 = attack_with_one_nd([ct0, ct1, ct2, ct3], 11, 13, None, nets[2], bits[2],c[2], tk)
            print('猜测的密钥是', sur_kg_3, kg_scores_3)
            kg_3, kg_scores_3 = select_top_k_candidates(sur_kg_3, kg_scores_3, k[2])
            print("猜测的密钥是：", kg_scores_3)
            print("猜测的分数是：", kg_scores_3)
            k3 = kg_3
            num += 1
            data_num += 1
            if len(sur_kg_3) == 0:
                print('\r {} plaintext structures generated'.format(num), end='')
                continue
            else:
                print(' ')
                print('Stage 3: ', len(sur_kg_1), ' subkeys survive')
                break
        if num == -1:
            print(' ')
            print('this trial fails.')
            print('{} plaintext structures are generated.'.format(data_num))
            print('the time consumption is ', time.time() - start)
            continue
        # 所有密钥猜测完成
        all_combinations = list(itertools.product(k1, k2, k3))
        final_keys =[]
        minest_distance = 32
        keyfinally = ''
        # 将k1，k2,k3拼成一个新的密钥
        for combination in all_combinations:
        # 将每个元素都转成2进制，其中a为
            final_key_temp = construct_array(combination[0],combination[1],combination[2])
            final_key_temp = int(final_key_temp, 2)
            distance = hamming_distance(tk, final_key_temp)
            final_keys.append(final_key_temp)
            if(minest_distance>distance):
                keyfinally = final_key_temp
                minest_distance = distance
            # 使用全密文的神经区分器去
        print('最小距离是',minest_distance)
        print('最小距离对应的密钥是', keyfinally)
        # 将密文使用完整密钥解密，然后使用神经区分器打分
        resultkey = final_select_one_key([selected_ct0, selected_ct1, selected_ct2, selected_ct3],final_keys,nets[3])
        end = time.time()
        print('the time consumption is ', end - start)
        resultdistance = hamming_distance(tk, resultkey)
        print('实验次数',i)
        print('最终密钥是',resultkey)
        print('推荐密钥和最终密钥之间的距离是', resultdistance)
        print('消耗时间', end - start)
        print('{} plaintext structures are generated.'.format(data_num))
        time_consumption[i] = end - start
        data_consumption[i] = data_num
        with open('output.txt', 'a') as f:
            print('实验次数', i,file=f)
            print('最终密钥是', resultkey,file=f)
            print('推荐密钥和最终密钥之间的距离是', resultdistance,file=f)
            print('the time consumption is ', end - start,file=f)
            print('the data consumption is ', data_num,file=f)
    print('average time consumption is', np.mean(time_consumption))
    print('average structure consumption is', np.mean(data_consumption))




 # 但是没有将神经网络的训练放进这里，这里需要自己去训练适合自己的神经区分器
if __name__ == '__main__':
    # (0x8000,0)
    nd1 = './trained_net/model_7r_diff47ext_depth5_num_epochs5_pair16.h5'
    # (0x80, 0)
    # nd2 = './trained_net/model_7r_diff39ext_depth5_num_epochs5_pair16.h5'
    nd2 = './trained_net/acc64tr_7r_diff39ext_bit2131_depth5_num_epochs5_pair16.h5'
    # (0x800000,0)
    nd3 = './trained_net/acc56_7r_diff56ext_bit010_depth5_num_epochs5_pair16.h5'
    final_nd = './trained_net/7_distinguisher_0x80_acc62.h5'
    selected_bits_1 = [10,11,12,13,14,15,16,17,18,19,20,21]
    # selected_bits_2 = [25,26,27,28,29,30,31,0,1,2,3]
    selected_bits_2 = [21,22,23,24,25,26,27,28,29,30,31]
    # 尽量避免分数过低的成为信息位，能不用随机解密一轮就不用
    selected_bits_3 = [0,1,2,3,4,5,6,7,8,9,10]
    # 这个差分往后一轮有较大的概率成为0000040000000000，即神经区分器的输入差分
    diff_1 = (0x900000, 0x1000)
    diff_2 = (0x9000, 0x10)
    diff_3 = (0x90000000, 0x100000)
    NB_1 = [39,38,37,36]
    NB_2 = [29,28,27,26]
    NB_3 = [0,1,2,3]
    mulpair_NB_1 = [35,34,33,32,31,30]
    mulpair_NB_2 = [25,24,23,22,21,20]
    mulpair_NB_3 = [4,5,6,7,8,9]
    attack_with_dual_NDs(t=100, nr=10, diffs=(diff_1, diff_2, diff_3), NBs=(NB_1, NB_2, NB_3),mulpair_NB=(mulpair_NB_1,mulpair_NB_2,mulpair_NB_3),nds=(nd1, nd2, nd3,final_nd),
                         bits=(selected_bits_1, selected_bits_2, selected_bits_3), c=(15, 15, 15), k=(5, 5, 5))