# while True:
#     try:
#
#         import numpy as np
#         import speck as sp
#         from os import urandom
#         from keras.models import load_model
#         import time
#         import itertools
#
#         word_size = sp.WORD_SIZE()
#         MASK_VAL = 2 ** word_size - 1
#
#
#         def extrs_info_bit(X, bit):
#             bit = np.array(bit)
#             bit = np.sort(bit)
#             bitall = np.concatenate(
#                 ([(i - 8) % 31 for i in bit], [(i - 8) % 31 + 32 for i in bit], [i + 64 for i in bit],
#                  [i + 96 for i in bit],
#                  [i + 128 for i in bit], [i + 160 for i in bit]))
#             resultbitall = np.copy(bitall)
#             X = X[:, resultbitall]
#             return X
#
#
#         def extract_info_bits(bit=None):
#             bitall = np.concatenate(
#                 (bit, [i + 32 for i in bit], [(i - 8) % 32 + 64 for i in bit], [(i - 8) % 32 + 64 + 32 for i in bit]))
#             resultbitall = np.copy(bitall)
#             for i in range(1, 2):
#                 bitall_plus_n = bitall + 64 * i
#                 resultbitall = np.concatenate((resultbitall, bitall_plus_n))
#             return resultbitall
#
#
#         # 同样的，这里使用的方法也是使用神经
#         def extract_sensitive_bits(raw_x, bit=None):
#             bitall = np.concatenate((bit, [i + 48 for i in bit], [i + 96 for i in bit], [i + 96 + 48 for i in bit]))
#             new_x = raw_x[:, bitall]
#             return new_x
#
#
#         # 根据输入差分和中性比特生成明文结构
#         def make_plaintext_structure(diff=(0x211, 0xa04), neutral_bits=None):
#             p0l = np.frombuffer(urandom(8), dtype=np.uint64)
#             p0r = np.frombuffer(urandom(8), dtype=np.uint64)
#             for i in neutral_bits:
#                 if isinstance(i, int):
#                     i = [i]
#                 d0 = 0
#                 d1 = 0
#                 for j in i:
#                     d = 1 << j
#                     d0 |= d >> word_size
#                     d1 |= d & MASK_VAL
#                 p0l = np.concatenate([p0l, p0l ^ d0])
#                 p0r = np.concatenate([p0r, p0r ^ d1])
#             p1l = p0l ^ diff[0]
#             p1r = p0r ^ diff[1]
#             return p0l, p0r, p1l, p1r
#
#
#         # 生成密文结构
#         def collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks):
#             p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
#             p1l, p1r = sp.dec_one_round((p1l, p1r), 0)
#             c0l, c0r = sp.encrypt((p0l, p0r), ks)
#             c1l, c1r = sp.encrypt((p1l, p1r), ks)
#
#             return c0l, c0r, c1l, c1r
#
#
#         def rotate_left(val, r_bits, max_bits):
#             return (val << r_bits % max_bits) & (2 ** max_bits - 1) | \
#                 ((val & (2 ** max_bits - 1)) >> (max_bits - (r_bits % max_bits)))
#
#
#         def attack_with_one_nd(cts, kg_bit_num, kg_offset, sur_kg_low, net, bits, c):
#             sur_kg = []
#             sur_kg_socres = []
#             kg_batch = 2 ** kg_bit_num
#             c0l, c0r, c1l, c1r = cts[0], cts[1], cts[2], cts[3]
#             n = len(c0l)
#             # 扩展成猜测数目额数量
#             c0l, c0r, c1l, c1r = np.tile(c0l, kg_batch), np.tile(c0r, kg_batch), np.tile(c1l, kg_batch), np.tile(c1r,
#                                                                                                                  kg_batch)
#             kg_high = np.arange(kg_batch, dtype=np.uint64)
#             if kg_offset == None:
#                 kg = kg_high;
#                 key_guess = np.repeat(kg, n)
#                 d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
#                 d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
#                 raw_x = sp.convert_to_binary([d0l, d0r, d1l, d1r])
#                 x = extract_sensitive_bits(raw_x, bits)
#                 z = net.predict(x, batch_size=10000)
#                 z = np.log2(z / (1 - z));
#                 z = np.reshape(z, (kg_batch, n))
#                 # 行求和
#                 s = np.sum(z, axis=1)
#                 # 实验证明这里同样是能实现区分，但是为什么依然大部分的分数都是负数？这就意味着，判断的分数仍然是大部分都是不合格的，但是问题不大，猜出来的结果是一样的
#                 # 会不会是训练数据的问题？
#                 for i in range(kg_batch):
#                     if s[i] > 10:
#                         sur_kg.append(i)
#                         sur_kg_socres.append(s[i])
#                 return sur_kg, sur_kg_socres
#             else:
#                 kg = kg_high << kg_offset
#                 key_guess = np.repeat(kg, n)
#                 d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
#                 d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
#                 raw_x = sp.convert_to_binary([d0l, d0r, d1l, d1r])
#                 x = extract_sensitive_bits(raw_x, bits)
#                 z = net.predict(x, batch_size=10000)
#                 z = np.log2(z / (1 - z));
#                 z = np.reshape(z, (kg_batch, n))
#                 # 行求和
#                 s = np.sum(z, axis=1)
#                 # 实验证明这里同样是能实现区分，但是为什么依然大部分的分数都是负数？这就意味着，判断的分数仍然是大部分都是不合格的，但是问题不大，猜出来的结果是一样的
#                 # 会不会是训练数据的问题？
#                 for i in range(kg_batch):
#                     if s[i] > 10:
#                         sur_kg.append(i)
#                         sur_kg_socres.append(s[i])
#                 return sur_kg, sur_kg_socres
#
#
#         def output_sk_kg_diff(sk, sur_kg, kg_scores):
#             for i in range(len(sur_kg)):
#                 print('difference between surviving kg and sk is {}, rank score is {}'.format(
#                     hex(sk ^ np.uint32(sur_kg[i])),
#                     kg_scores[i]))
#
#
#         # 选择排位最高的几个密钥
#         def select_top_k_candidates(sur_kg, kg_scores, k=3):
#             num = len(sur_kg)
#             tp = kg_scores.copy()
#             tp.sort(reverse=True)
#             # print('the top scores are ', tp[:5])
#             if num > k:
#                 base = tp[k]
#             else:
#                 return sur_kg, kg_scores
#             filtered_subkey = []
#             filtered_score = []
#             for i in range(num):
#                 if kg_scores[i] > base:
#                     filtered_subkey.append(sur_kg[i])
#                     filtered_score.append(kg_scores[i])
#             return filtered_subkey, filtered_score
#
#
#         def generate_mask(a):
#             mask = ['0'] * 48
#
#             for bit in a:  # 使用全局变量B
#                 if 0 <= bit < 48:
#                     mask[bit] = '1'
#
#             int_mask = int(''.join(mask), 2)
#             return int_mask
#
#
#         def int_to_bin(x, width):
#             """将整数转换为指定宽度的二进制字符串，如果宽度不足，用0填充最高位"""
#             return format(x, 'b').zfill(width)
#
#
#         def construct_array(a, b, c, d):
#             # 将输入转换为指定宽度的二进制字符串
#             bin_a = int_to_bin(a, 12)
#             bin_b = int_to_bin(b, 12)
#             bin_c = int_to_bin(c, 12)
#             bin_d = int_to_bin(d, 12)
#             # 根据规则拼接二进制字符串
#             e = bin_d + bin_c + bin_b + bin_a
#             return e
#
#
#         def hamming_distance(a, b):
#             # 将a转换为32位的二进制字符串
#             a_bin = format(a, '048b')
#             b = format(b, '048b')
#             # 计算汉明距离
#             distance = sum(bit1 != bit2 for bit1, bit2 in zip(a_bin, b))
#             return distance
#
#
#         def final_select_top_three_keys(cts, final_keys, net):
#             keynum = len(final_keys)
#             c0l, c0r, c1l, c1r = cts[0], cts[1], cts[2], cts[3]
#             key_guess = np.repeat(final_keys, 16 * 64)
#             c0l, c0r, c1l, c1r = np.tile(c0l, keynum), np.tile(c0r, keynum), np.tile(c1l, keynum), np.tile(c1r, keynum)
#             d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
#             d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
#             X = sp.convert_to_binary([d0l, d0r, d1l, d1r])
#             Z = net.predict(X, batch_size=3000)
#             Z = np.log2(Z / (1 - Z))
#             Z = Z.reshape(-1, 1024)
#             s = np.sum(Z, axis=1)
#             top_three_indices = np.argsort(s)[-3:][::-1]  # 反转以得到最大的三个索引
#             top_three_keys = [final_keys[index] for index in top_three_indices]
#             return top_three_keys
#
#
#         def final_select_one_key(cts, final_keys, net):
#             keynum = len(final_keys)
#             c0l, c0r, c1l, c1r = cts[0], cts[1], cts[2], cts[3]
#             key_guess = np.repeat(final_keys, 16 * 64)
#             c0l, c0r, c1l, c1r = np.tile(c0l, keynum), np.tile(c0r, keynum), np.tile(c1l, keynum), np.tile(c1r, keynum)
#             d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
#             d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
#             X = sp.convert_to_binary([d0l, d0r, d1l, d1r])
#             Z = net.predict(X, batch_size=10000)
#             Z = np.log2(Z / (1 - Z))
#             Z = Z.reshape(-1, 1024)
#             s = np.sum(Z, axis=1)
#             index = np.argmax(s)
#             return final_keys[index]
#
#
#         def process_array(a):
#             new = []
#
#             for number in a:
#                 # 首先将原始数添加到新数组中
#                 new.append(number)
#
#                 # 将十进制数转换为二进制字符串
#                 binary_str = format(number, '048b')
#
#                 # 反转一个比特
#                 for i in range(48):
#                     flipped_str = binary_str[:i] + ('0' if binary_str[i] == '1' else '1') + binary_str[i + 1:]
#                     new.append(int(flipped_str, 2))
#
#                 # 反转两个比特
#                 for i in range(48):
#                     for j in range(i + 1, 48):
#                         flipped_str = list(binary_str)
#                         flipped_str[i] = '0' if binary_str[i] == '1' else '1'
#                         flipped_str[j] = '0' if binary_str[j] == '1' else '1'
#                         new.append(int(''.join(flipped_str), 2))
#
#             return new
#         def attack_with_dual_NDs(t, nr, diffs, NBs, nds, bits, c, k):
#             nets = []
#             for nd in nds:
#                 nets.append(load_model(nd))
#             acc = 0
#             time_consumption = np.zeros(t)
#             data_consumption = np.zeros(t, dtype=np.uint32)
#             for i in range(t):
#                 print('attack index: {}'.format(i))
#                 data_num = 0
#                 start = time.time()
#                 key = np.frombuffer(urandom(16), dtype=np.uint64).reshape(2, -1)
#                 ks = sp.expand_key(key, nr)
#                 # print("产生的密钥是", ks)
#                 tk = ks[-1][0]
#                 # print("产生的最后一轮子密钥是：", tk)
#                 selected_ct0, selected_ct1, selected_ct2, selected_ct3 = [], [], [], []
#
#                 # stage 1, guess sk[9~0], diff index is 42
#                 num = 0
#                 while True:
#                     if num >= 2 ** 10:
#                         num = -1
#                         break
#
#                     p0l, p0r, p1l, p1r = make_plaintext_structure(diffs[0], NBs[0])
#                     # p0l1, p0r1 = sp.dec_one_round((p0l, p0r), 0)
#                     # p1l1, p1r1 = sp.dec_one_round((p1l, p1r), 0)
#                     # c0l1, c0r1 = sp.encrypt((p0l1, p0r1), [ks[0][0], ks[1][0]])
#                     # c1l1, c1r1 = sp.encrypt((p1l1, p1r1), [ks[0][0], ks[1][0]])
#                     # diffl = c0l1 ^ c1l1
#                     # diffr = c0r1 ^ c1r1
#                     # diff = (diffl[0], diffr[0])
#                     # print("符合差分？", diff==(0x20,0))
#
#                     c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
#                     selected_ct0, selected_ct1, selected_ct2, selected_ct3 = c0l, c0r, c1l, c1r
#                     # 将此结构使用正确密钥解密之后放进神经区分器里面
#                     # a = [(i + 8) % 48 for i in selected_bits_1]
#                     # bitall = np.concatenate((selected_bits_1, [i + 48 for i in selected_bits_1], [i + 96 for i in selected_bits_1], [i + 96+48 for i in selected_bits_1]))
#                     # mask = generate_mask(a)
#                     # tk1 = np.array(tk)
#                     # key_bit = np.bitwise_and(tk1.item(), mask)
#                     # # key_bit = np.bitwise_and(tk1, mask)
#                     # d0l_t, d0r_t = sp.dec_one_round((c0l, c0r), key_bit)
#                     # d1l_t, d1r_t = sp.dec_one_round((c1l, c1r), key_bit)
#                     # X = sp.convert_to_binary([ d0l_t, d0r_t, d1l_t, d1r_t])
#                     # X = X[:, bitall]
#                     # z = nets[0].predict(X,batch_size=10000)
#                     # z = np.log2(z / (1 - z));
#                     # s = np.sum(z)
#                     # print("分数是",s)
#                     #
#                     sur_kg_1, kg_scores_1 = attack_with_one_nd([c0l, c0r, c1l, c1r], 12, None, None, nets[0], bits[0],
#                                                                c[0])
#                     # print("猜测的密钥是：", sur_kg_1)
#                     # print("对应的分数是：",kg_scores_1)
#                     num += 1
#                     data_num += 1
#                     if len(sur_kg_1) == 0:
#                         print('\r {} plaintext structures generated'.format(num), end='')
#                         continue
#                     else:
#                         print(' ')
#                         print('Stage 1: ', len(sur_kg_1), ' subkeys survive')
#                         break
#                 if num == -1:
#                     print(' ')
#                     print('this trial fails.')
#                     print('{} plaintext structures are generated.'.format(data_num))
#                     print('the time consumption is ', time.time() - start)
#                     continue
#                 kg_1, kg_scores_1 = select_top_k_candidates(sur_kg_1, kg_scores_1, k[0])
#                 # print("猜测的密钥是：", kg_1)
#                 # print("猜测的分数是：", kg_scores_1)
#
#                 # stage 2, guess sk[9~0], diff index is 42
#                 num = 0
#                 while True:
#                     if num >= 2 ** 10:
#                         num = -1
#                         break
#
#                     p0l, p0r, p1l, p1r = make_plaintext_structure(diffs[1], NBs[1])
#                     # p0l1, p0r1 = sp.dec_one_round((p0l, p0r), 0)
#                     # p1l1, p1r1 = sp.dec_one_round((p1l, p1r), 0)
#                     # c0l1, c0r1 = sp.encrypt((p0l1, p0r1), [ks[0][0], ks[1][0]])
#                     # c1l1, c1r1 = sp.encrypt((p1l1, p1r1), [ks[0][0], ks[1][0]])
#                     # diffl = c0l1 ^ c1l1
#                     # diffr = c0r1 ^ c1r1
#                     # diff = (diffl[1], diffr[1])
#                     # print("符合差分？", diff == (0x20000, 0))
#
#                     c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
#                     # 将此结构使用正确密钥解密之后放进神经区分器里面
#                     # a = [(i + 8) % 48 for i in selected_bits_2]
#                     # bitall = np.concatenate((selected_bits_2, [i + 48 for i in selected_bits_2],
#                     #                          [i + 96 for i in selected_bits_2], [i + 96 + 48 for i in selected_bits_2]))
#                     # mask = generate_mask(a)
#                     # tk1 = np.array(tk)
#                     # key_bit = np.bitwise_and(tk1.item(), mask)
#                     # # key_bit = np.bitwise_and(tk1, mask)
#                     # d0l_t, d0r_t = sp.dec_one_round((c0l, c0r), key_bit)
#                     # d1l_t, d1r_t = sp.dec_one_round((c1l, c1r), key_bit)
#                     # X = sp.convert_to_binary([d0l_t, d0r_t, d1l_t, d1r_t])
#                     # X = X[:, bitall]
#                     # z = nets[1].predict(X, batch_size=10000)
#                     # z = np.log2(z / (1 - z));
#                     # s = np.sum(z)
#                     # print("分数是", s)
#                     #
#                     sur_kg_2, kg_scores_2 = attack_with_one_nd([c0l, c0r, c1l, c1r], 12, 12, None, nets[1], bits[1],
#                                                                c[1])
#                     # print("猜测的密钥是：", sur_kg_2)
#                     # print("对应的分数是：", kg_scores_2)
#                     num += 1
#                     data_num += 1
#                     if len(sur_kg_2) == 0:
#                         print('\r {} plaintext structures generated'.format(num), end='')
#                         continue
#                     else:
#                         print(' ')
#                         print('Stage 2: ', len(sur_kg_2), ' subkeys survive')
#                         break
#                 if num == -1:
#                     print(' ')
#                     print('this trial fails.')
#                     print('{} plaintext structures are generated.'.format(data_num))
#                     print('the time consumption is ', time.time() - start)
#                     continue
#                 kg_2, kg_scores_2 = select_top_k_candidates(sur_kg_2, kg_scores_2, k[1])
#                 # print("猜测的密钥是：", kg_2)
#                 # print("猜测的分数是：", kg_scores_2)
#
#                 #     # stage 3, guess sk[31, 22], diff index is 33
#                 num = 0
#                 while True:
#                     if num >= 2 ** 10:
#                         num = -1
#                         break
#
#                     p0l, p0r, p1l, p1r = make_plaintext_structure(diffs[2], NBs[2])
#                     # p0l1, p0r1 = sp.dec_one_round((p0l, p0r), 0)
#                     # p1l1, p1r1 = sp.dec_one_round((p1l, p1r), 0)
#                     # c0l1, c0r1 = sp.encrypt((p0l1, p0r1), [ks[0][0], ks[1][0]])
#                     # c1l1, c1r1 = sp.encrypt((p1l1, p1r1), [ks[0][0], ks[1][0]])
#                     # diffl = c0l1 ^ c1l1
#                     # diffr = c0r1 ^ c1r1
#                     # diff = (diffl[2], diffr[2])
#                     # print("符合差分？", diff == (0x20000000, 0))
#
#                     c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
#                     # 将此结构使用正确密钥解密之后放进神经区分器里面
#                     # a = [(i + 8) % 48 for i in selected_bits_3]
#                     # bitall = np.concatenate((selected_bits_3, [i + 48 for i in selected_bits_3],
#                     #                          [i + 96 for i in selected_bits_3], [i + 96 + 48 for i in selected_bits_3]))
#                     # mask = generate_mask(a)
#                     # tk1 = np.array(tk)
#                     # key_bit = np.bitwise_and(tk1.item(), mask)
#                     # # key_bit = np.bitwise_and(tk1, mask)
#                     # d0l_t, d0r_t = sp.dec_one_round((c0l, c0r), key_bit)
#                     # d1l_t, d1r_t = sp.dec_one_round((c1l, c1r), key_bit)
#                     # X = sp.convert_to_binary([d0l_t, d0r_t, d1l_t, d1r_t])
#                     # X = X[:, bitall]
#                     # z = nets[2].predict(X, batch_size=10000)
#                     # z = np.log2(z / (1 - z));
#                     # s = np.sum(z)
#                     # print("分数是", s)
#                     #
#                     sur_kg_3, kg_scores_3 = attack_with_one_nd([c0l, c0r, c1l, c1r], 12, 24, None, nets[2], bits[2],
#                                                                c[2])
#                     # print("猜测的密钥是：", sur_kg_3)
#                     # print("对应的分数是：", kg_scores_3)
#                     num += 1
#                     data_num += 1
#                     if len(sur_kg_3) == 0:
#                         print('\r {} plaintext structures generated'.format(num), end='')
#                         continue
#                     else:
#                         print(' ')
#                         print('Stage 3: ', len(sur_kg_3), ' subkeys survive')
#                         break
#                 if num == -1:
#                     print(' ')
#                     print('this trial fails.')
#                     print('{} plaintext structures are generated.'.format(data_num))
#                     print('the time consumption is ', time.time() - start)
#                     continue
#                 kg_3, kg_scores_3 = select_top_k_candidates(sur_kg_3, kg_scores_3, k[2])
#                 # print("猜测的密钥是：", kg_3)
#                 # print("猜测的分数是：", kg_scores_3)
#                 #     # stage 3, guess sk[31, 22], diff index is 33
#                 num = 0
#                 while True:
#                     if num >= 2 ** 10:
#                         num = -1
#                         break
#
#                     p0l, p0r, p1l, p1r = make_plaintext_structure(diffs[3], NBs[3])
#                     # p0l1, p0r1 = sp.dec_one_round((p0l, p0r), 0)
#                     # p1l1, p1r1 = sp.dec_one_round((p1l, p1r), 0)
#                     # c0l1, c0r1 = sp.encrypt((p0l1, p0r1), [ks[0][0], ks[1][0]])
#                     # c1l1, c1r1 = sp.encrypt((p1l1, p1r1), [ks[0][0], ks[1][0]])
#                     # diffl = c0l1 ^ c1l1
#                     # diffr = c0r1 ^ c1r1
#                     # diff = (diffl[3], diffr[3])
#                     # print("符合差分？", diff == (0x20000000000, 0))
#
#                     c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
#                     # 将此结构使用正确密钥解密之后放进神经区分器里面
#                     # a = [(i + 8) % 48 for i in selected_bits_4]
#                     # bitall = np.concatenate((selected_bits_4, [i + 48 for i in selected_bits_4],
#                     #                          [i + 96 for i in selected_bits_4], [i + 96 + 48 for i in selected_bits_4]))
#                     # mask = generate_mask(a)
#                     # tk1 = np.array(tk)
#                     # key_bit = np.bitwise_and(tk1.item(), mask)
#                     # # key_bit = np.bitwise_and(tk1, mask)
#                     # d0l_t, d0r_t = sp.dec_one_round((c0l, c0r), key_bit)
#                     # d1l_t, d1r_t = sp.dec_one_round((c1l, c1r), key_bit)
#                     # X = sp.convert_to_binary([d0l_t, d0r_t, d1l_t, d1r_t])
#                     # X = X[:, bitall]
#                     # z = nets[3].predict(X, batch_size=10000)
#                     # z = np.log2(z / (1 - z));
#                     # s = np.sum(z)
#                     # print("分数是", s)
#                     #
#                     sur_kg_4, kg_scores_4 = attack_with_one_nd([c0l, c0r, c1l, c1r], 12, 36, None, nets[3], bits[3],
#                                                                c[3])
#                     # print("猜测的密钥是：", sur_kg_4)
#                     # print("对应的分数是：", kg_scores_4)
#                     num += 1
#                     data_num += 1
#                     if len(sur_kg_4) == 0:
#                         print('\r {} plaintext structures generated'.format(num), end='')
#                         continue
#                     else:
#                         print(' ')
#                         print('Stage 4: ', len(sur_kg_4), ' subkeys survive')
#                         break
#                 if num == -1:
#                     print(' ')
#                     print('this trial fails.')
#                     print('{} plaintext structures are generated.'.format(data_num))
#                     print('the time consumption is ', time.time() - start)
#                     continue
#                 kg_4, kg_scores_4 = select_top_k_candidates(sur_kg_4, kg_scores_4, k[3])
#                 # print("猜测的密钥是：", kg_4)
#                 # print("猜测的分数是：", kg_scores_4)
#                 #
#                 all_combinations = list(itertools.product(kg_1, kg_2, kg_3, kg_4))
#                 final_keys = []
#                 minest_distance = 48
#                 keyfinally = ''
#                 for combination in all_combinations:
#                     # 将每个元素都转成2进制，其中a为
#                     final_key_temp = construct_array(combination[0], combination[1], combination[2], combination[3])
#                     final_key_temp = int(final_key_temp, 2)
#                     distance = hamming_distance(tk, final_key_temp)
#                     final_keys.append(final_key_temp)
#                     if (minest_distance > distance):
#                         keyfinally = final_key_temp
#                         minest_distance = distance
#                 # print('最小距离是', minest_distance)
#                 # print('最小距离对应的密钥是', keyfinally)
#                 # 将密文使用完整密钥解密，然后使用神经区分器打分
#                 final_keys = np.array(final_keys)
#                 print(len(final_keys))
#                 final_keys = final_keys.astype(np.uint64)
#                 print(len(final_keys))
#                 resultkey = final_select_top_three_keys([selected_ct0, selected_ct1, selected_ct2, selected_ct3],
#                                                         final_keys,
#                                                         nets[4])
#                 # 得到最高分的三个密钥之后对三个密钥分别做以下操作：
#                 # 反转密钥的一位两位，然后再选取最后的密钥
#                 flipkey=process_array(resultkey)
#                 print(flipkey)
#                 resultkey = final_select_top_three_keys([selected_ct0, selected_ct1, selected_ct2, selected_ct3],
#                                                         flipkey,
#                                                         nets[4])
#                 end = time.time()
#                 print('the time consumption is ', end - start)
#                 resultdistance = []
#                 for resultkeysub in resultkey:
#                     resultdistance.append(hamming_distance(tk, resultkeysub))
#                 resultdistancesmallest = min(resultdistance)
#                 # print('实验次数', i)
#                 # print('最终密钥是', resultkey)
#                 # print('推荐密钥和最终密钥之间的距离是', resultdistancesmallest)
#                 # print('消耗时间', end - start)
#                 # print('{} plaintext structures are generated.'.format(data_num))
#                 time_consumption[i] = end - start
#                 data_consumption[i] = data_num
#                 with open('outputnew5.txt', 'a') as f:
#                     print('实验次数', i, file=f)
#                     print('最终密钥是', resultkey, file=f)
#                     print('推荐密钥和最终密钥之间的距离是', resultdistancesmallest, file=f)
#                     print('the time consumption is ', end - start, file=f)
#                     print('the data consumption is ', data_num, file=f)
#             print('average time consumption is', np.mean(time_consumption))
#             print('average structure consumption is', np.mean(data_consumption))
#             #     resultkey = final_select_one_key([selected_ct0, selected_ct1, selected_ct2, selected_ct3],final_keys,nets[4])
#             #     end = time.time()
#             #     print('the time consumption is ', end - start)
#             #     resultdistance = hamming_distance(tk, resultkey)
#             #     print('实验次数', i)
#             #     print('最终密钥是', resultkey)
#             #     print('推荐密钥和最终密钥之间的距离是', resultdistance)
#             #     print('消耗时间', end - start)
#             #     print('{} plaintext structures are generated.'.format(data_num))
#             #     time_consumption[i] = end - start
#             #     data_consumption[i] = data_num
#             #     with open('output.txt', 'a') as f:
#             #         print('实验次数', i, file=f)
#             #         print('最终密钥是', resultkey, file=f)
#             #         print('推荐密钥和最终密钥之间的距离是', resultdistance, file=f)
#             #         print('the time consumption is ', end - start, file=f)
#             #         print('the data consumption is ', data_num, file=f)
#             # print('average time consumption is', np.mean(time_consumption))
#             # print('average structure consumption is', np.mean(data_consumption))
#
#
#         # 但是没有将神经网络的训练放进这里，这里需要自己去训练适合自己的神经区分器
#         if __name__ == '__main__':
#             nd1 = './trained_net/model_7r_diff53_acc63_depth1_num_epochs5.h5'
#             nd2 = './trained_net/model_7r_diff65_acc58_depth1_num_epochs5.h5'
#             nd3 = './trained_net/model_7r_diff77_acc59_depth1_num_epochs5.h5'
#             nd4 = './trained_net/model_7r_diff89_acc60_depth1_num_epochs5.h5'
#             nd5 = './trained_net/model_7r_diff47_depth1_num_epochs5.h5'  # 最终的筛选器
#             selected_bits_1 = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
#             selected_bits_2 = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
#             selected_bits_3 = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#             selected_bits_4 = [0, 1, 2, 3, 40, 41, 42, 43, 44, 45, 46, 47]
#
#             # 这个差分往后一轮有较大的概率成为0000040000000000，即神经区分器的输入差分
#             diff_1 = (0x2400, 0x4)
#             diff_2 = (0x2400000, 0x4000)
#             diff_3 = (0x2400000000, 0x4000000)
#             diff_4 = (0x400000000002, 0x4000000000)
#             NB_1 = [25 - i for i in range(10)]
#             NB_2 = [37 - i for i in range(10)]
#             NB_3 = [49 - i for i in range(10)]
#             NB_4 = [61 - i for i in range(10)]
#             attack_with_dual_NDs(t=100, nr=10, diffs=(diff_1, diff_2, diff_3, diff_4), NBs=(NB_1, NB_2, NB_3, NB_4),
#                                  nds=(nd1, nd2, nd3, nd4, nd5),
#                                  bits=(selected_bits_1, selected_bits_2, selected_bits_3, selected_bits_4),
#                                  c=(10, 15, 15, 15), k=(3, 10, 10, 10))
    # except Exception as e:
    #     print("发生错误：", e)
    #     # 错误处理代码
while True:
    try:
        import numpy as np
        import speck as sp
        from os import urandom
        from keras.models import load_model
        import time
        import itertools

        word_size = sp.WORD_SIZE()
        MASK_VAL = 2 ** word_size - 1


        def extrs_info_bit(X, bit):
            bit = np.array(bit)
            bit = np.sort(bit)
            bitall = np.concatenate(
                ([(i - 8) % 31 for i in bit], [(i - 8) % 31 + 32 for i in bit], [i + 64 for i in bit],
                 [i + 96 for i in bit],
                 [i + 128 for i in bit], [i + 160 for i in bit]))
            resultbitall = np.copy(bitall)
            X = X[:, resultbitall]
            return X


        def extract_info_bits(bit=None):
            bitall = np.concatenate(
                (bit, [i + 32 for i in bit], [(i - 8) % 32 + 64 for i in bit], [(i - 8) % 32 + 64 + 32 for i in bit]))
            resultbitall = np.copy(bitall)
            for i in range(1, 2):
                bitall_plus_n = bitall + 64 * i
                resultbitall = np.concatenate((resultbitall, bitall_plus_n))
            return resultbitall


        # 同样的，这里使用的方法也是使用神经
        def extract_sensitive_bits(raw_x, bit=None):
            bitall = np.concatenate((bit, [i + 48 for i in bit], [i + 96 for i in bit], [i + 96 + 48 for i in bit]))
            new_x = raw_x[:, bitall]
            return new_x


        # 根据输入差分和中性比特生成明文结构
        def make_plaintext_structure(diff=(0x211, 0xa04), neutral_bits=None):
            p0l = np.frombuffer(urandom(8), dtype=np.uint64)
            p0r = np.frombuffer(urandom(8), dtype=np.uint64)
            for i in neutral_bits:
                if isinstance(i, int):
                    i = [i]
                d0 = 0
                d1 = 0
                for j in i:
                    d = 1 << j
                    d0 |= d >> word_size
                    d1 |= d & MASK_VAL
                p0l = np.concatenate([p0l, p0l ^ d0])
                p0r = np.concatenate([p0r, p0r ^ d1])
            p1l = p0l ^ diff[0]
            p1r = p0r ^ diff[1]
            return p0l, p0r, p1l, p1r


        # 生成密文结构
        def collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks):
            p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
            p1l, p1r = sp.dec_one_round((p1l, p1r), 0)
            c0l, c0r = sp.encrypt((p0l, p0r), ks)
            c1l, c1r = sp.encrypt((p1l, p1r), ks)

            return c0l, c0r, c1l, c1r


        def rotate_left(val, r_bits, max_bits):
            return (val << r_bits % max_bits) & (2 ** max_bits - 1) | \
                ((val & (2 ** max_bits - 1)) >> (max_bits - (r_bits % max_bits)))


        def attack_with_one_nd(cts, kg_bit_num, kg_offset, sur_kg_low, net, bits, c):
            sur_kg = []
            sur_kg_socres = []
            kg_batch = 2 ** kg_bit_num
            c0l, c0r, c1l, c1r = cts[0], cts[1], cts[2], cts[3]
            n = len(c0l)
            # 扩展成猜测数目额数量
            c0l, c0r, c1l, c1r = np.tile(c0l, kg_batch), np.tile(c0r, kg_batch), np.tile(c1l, kg_batch), np.tile(c1r,
                                                                                                                 kg_batch)
            kg_high = np.arange(kg_batch, dtype=np.uint64)
            if kg_offset == None:
                kg = kg_high;
                key_guess = np.repeat(kg, n)
                d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
                d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
                raw_x = sp.convert_to_binary([d0l, d0r, d1l, d1r])
                x = extract_sensitive_bits(raw_x, bits)
                z = net.predict(x, batch_size=10000)
                z = np.log2(z / (1 - z));
                z = np.reshape(z, (kg_batch, n))
                # 行求和
                s = np.sum(z, axis=1)
                # 实验证明这里同样是能实现区分，但是为什么依然大部分的分数都是负数？这就意味着，判断的分数仍然是大部分都是不合格的，但是问题不大，猜出来的结果是一样的
                # 会不会是训练数据的问题？
                for i in range(kg_batch):
                    if s[i] > 10:
                        sur_kg.append(i)
                        sur_kg_socres.append(s[i])
                return sur_kg, sur_kg_socres
            else:
                kg = kg_high << kg_offset
                key_guess = np.repeat(kg, n)
                d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
                d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
                raw_x = sp.convert_to_binary([d0l, d0r, d1l, d1r])
                x = extract_sensitive_bits(raw_x, bits)
                z = net.predict(x, batch_size=10000)
                z = np.log2(z / (1 - z));
                z = np.reshape(z, (kg_batch, n))
                # 行求和
                s = np.sum(z, axis=1)
                # 实验证明这里同样是能实现区分，但是为什么依然大部分的分数都是负数？这就意味着，判断的分数仍然是大部分都是不合格的，但是问题不大，猜出来的结果是一样的
                # 会不会是训练数据的问题？
                for i in range(kg_batch):
                    if s[i] > 10:
                        sur_kg.append(i)
                        sur_kg_socres.append(s[i])
                return sur_kg, sur_kg_socres


        def output_sk_kg_diff(sk, sur_kg, kg_scores):
            for i in range(len(sur_kg)):
                print('difference between surviving kg and sk is {}, rank score is {}'.format(
                    hex(sk ^ np.uint32(sur_kg[i])),
                    kg_scores[i]))


        # 选择排位最高的几个密钥
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


        def generate_mask(a):
            mask = ['0'] * 48

            for bit in a:  # 使用全局变量B
                if 0 <= bit < 48:
                    mask[bit] = '1'

            int_mask = int(''.join(mask), 2)
            return int_mask


        def int_to_bin(x, width):
            """将整数转换为指定宽度的二进制字符串，如果宽度不足，用0填充最高位"""
            return format(x, 'b').zfill(width)


        def construct_array(a, b, c, d):
            # 将输入转换为指定宽度的二进制字符串
            bin_a = int_to_bin(a, 12)
            bin_b = int_to_bin(b, 12)
            bin_c = int_to_bin(c, 12)
            bin_d = int_to_bin(d, 12)
            # 根据规则拼接二进制字符串
            e = bin_d + bin_c + bin_b + bin_a
            return e


        def hamming_distance(a, b):
            # 将a转换为32位的二进制字符串
            a_bin = format(a, '048b')
            b = format(b, '048b')
            # 计算汉明距离
            distance = sum(bit1 != bit2 for bit1, bit2 in zip(a_bin, b))
            return distance


        def final_select_top_three_keys(cts, final_keys, net):
            keynum = len(final_keys)
            c0l, c0r, c1l, c1r = cts[0], cts[1], cts[2], cts[3]
            key_guess = np.repeat(final_keys, 16 * 64)
            c0l, c0r, c1l, c1r = np.tile(c0l, keynum), np.tile(c0r, keynum), np.tile(c1l, keynum), np.tile(c1r, keynum)
            d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
            d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
            X = sp.convert_to_binary([d0l, d0r, d1l, d1r])
            Z = net.predict(X, batch_size=3000)
            Z = np.log2(Z / (1 - Z))
            Z = Z.reshape(-1, 1024)
            s = np.sum(Z, axis=1)
            top_three_indices = np.argsort(s)[-3:][::-1]  # 反转以得到最大的三个索引
            top_three_keys = [final_keys[index] for index in top_three_indices]
            return top_three_keys


        def final_select_one_key(cts, final_keys, net):
            keynum = len(final_keys)
            c0l, c0r, c1l, c1r = cts[0], cts[1], cts[2], cts[3]
            key_guess = np.repeat(final_keys, 16 * 64)
            c0l, c0r, c1l, c1r = np.tile(c0l, keynum), np.tile(c0r, keynum), np.tile(c1l, keynum), np.tile(c1r, keynum)
            d0l, d0r = sp.dec_one_round((c0l, c0r), key_guess)
            d1l, d1r = sp.dec_one_round((c1l, c1r), key_guess)
            X = sp.convert_to_binary([d0l, d0r, d1l, d1r])
            Z = net.predict(X, batch_size=10000)
            Z = np.log2(Z / (1 - Z))
            Z = Z.reshape(-1, 1024)
            s = np.sum(Z, axis=1)
            index = np.argmax(s)
            return final_keys[index]


        def process_array(a):
            new = []

            for number in a:
                # 首先将原始数添加到新数组中
                new.append(number)

                # 将十进制数转换为二进制字符串
                binary_str = format(number, '048b')

                # 反转一个比特
                for i in range(48):
                    flipped_str = binary_str[:i] + ('0' if binary_str[i] == '1' else '1') + binary_str[i + 1:]
                    new.append(int(flipped_str, 2))

                # 反转两个比特
                for i in range(48):
                    for j in range(i + 1, 48):
                        flipped_str = list(binary_str)
                        flipped_str[i] = '0' if binary_str[i] == '1' else '1'
                        flipped_str[j] = '0' if binary_str[j] == '1' else '1'
                        new.append(int(''.join(flipped_str), 2))

            return new


        def attack_with_dual_NDs(t, nr, diffs, NBs, nds, bits, c, k):
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
                key = np.frombuffer(urandom(16), dtype=np.uint64).reshape(2, -1)
                ks = sp.expand_key(key, nr)
                # print("产生的密钥是", ks)
                tk = ks[-1][0]
                # print("产生的最后一轮子密钥是：", tk)
                selected_ct0, selected_ct1, selected_ct2, selected_ct3 = [], [], [], []

                # stage 1, guess sk[9~0], diff index is 42
                num = 0
                while True:
                    if num >= 2 ** 10:
                        num = -1
                        break

                    p0l, p0r, p1l, p1r = make_plaintext_structure(diffs[0], NBs[0])
                    # p0l1, p0r1 = sp.dec_one_round((p0l, p0r), 0)
                    # p1l1, p1r1 = sp.dec_one_round((p1l, p1r), 0)
                    # c0l1, c0r1 = sp.encrypt((p0l1, p0r1), [ks[0][0], ks[1][0]])
                    # c1l1, c1r1 = sp.encrypt((p1l1, p1r1), [ks[0][0], ks[1][0]])
                    # diffl = c0l1 ^ c1l1
                    # diffr = c0r1 ^ c1r1
                    # diff = (diffl[0], diffr[0])
                    # print("符合差分？", diff==(0x20,0))

                    c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
                    selected_ct0, selected_ct1, selected_ct2, selected_ct3 = c0l, c0r, c1l, c1r
                    # 将此结构使用正确密钥解密之后放进神经区分器里面
                    # a = [(i + 8) % 48 for i in selected_bits_1]
                    # bitall = np.concatenate((selected_bits_1, [i + 48 for i in selected_bits_1], [i + 96 for i in selected_bits_1], [i + 96+48 for i in selected_bits_1]))
                    # mask = generate_mask(a)
                    # tk1 = np.array(tk)
                    # key_bit = np.bitwise_and(tk1.item(), mask)
                    # # key_bit = np.bitwise_and(tk1, mask)
                    # d0l_t, d0r_t = sp.dec_one_round((c0l, c0r), key_bit)
                    # d1l_t, d1r_t = sp.dec_one_round((c1l, c1r), key_bit)
                    # X = sp.convert_to_binary([ d0l_t, d0r_t, d1l_t, d1r_t])
                    # X = X[:, bitall]
                    # z = nets[0].predict(X,batch_size=10000)
                    # z = np.log2(z / (1 - z));
                    # s = np.sum(z)
                    # print("分数是",s)
                    #
                    sur_kg_1, kg_scores_1 = attack_with_one_nd([c0l, c0r, c1l, c1r], 12, None, None, nets[0], bits[0],
                                                               c[0])
                    # print("猜测的密钥是：", sur_kg_1)
                    # print("对应的分数是：",kg_scores_1)
                    num += 1
                    data_num += 1
                    if len(sur_kg_1) == 0:
                        print('\r {} plaintext structures generated'.format(num), end='')
                        continue
                    else:
                        print(' ')
                        print('Stage 1: ', len(sur_kg_1), ' subkeys survive')
                        break
                if num == -1:
                    print(' ')
                    print('this trial fails.')
                    print('{} plaintext structures are generated.'.format(data_num))
                    print('the time consumption is ', time.time() - start)
                    continue
                kg_1, kg_scores_1 = select_top_k_candidates(sur_kg_1, kg_scores_1, k[0])
                # print("猜测的密钥是：", kg_1)
                # print("猜测的分数是：", kg_scores_1)

                # stage 2, guess sk[9~0], diff index is 42
                num = 0
                while True:
                    if num >= 2 ** 10:
                        num = -1
                        break

                    p0l, p0r, p1l, p1r = make_plaintext_structure(diffs[1], NBs[1])
                    # p0l1, p0r1 = sp.dec_one_round((p0l, p0r), 0)
                    # p1l1, p1r1 = sp.dec_one_round((p1l, p1r), 0)
                    # c0l1, c0r1 = sp.encrypt((p0l1, p0r1), [ks[0][0], ks[1][0]])
                    # c1l1, c1r1 = sp.encrypt((p1l1, p1r1), [ks[0][0], ks[1][0]])
                    # diffl = c0l1 ^ c1l1
                    # diffr = c0r1 ^ c1r1
                    # diff = (diffl[1], diffr[1])
                    # print("符合差分？", diff == (0x20000, 0))

                    c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
                    # 将此结构使用正确密钥解密之后放进神经区分器里面
                    # a = [(i + 8) % 48 for i in selected_bits_2]
                    # bitall = np.concatenate((selected_bits_2, [i + 48 for i in selected_bits_2],
                    #                          [i + 96 for i in selected_bits_2], [i + 96 + 48 for i in selected_bits_2]))
                    # mask = generate_mask(a)
                    # tk1 = np.array(tk)
                    # key_bit = np.bitwise_and(tk1.item(), mask)
                    # # key_bit = np.bitwise_and(tk1, mask)
                    # d0l_t, d0r_t = sp.dec_one_round((c0l, c0r), key_bit)
                    # d1l_t, d1r_t = sp.dec_one_round((c1l, c1r), key_bit)
                    # X = sp.convert_to_binary([d0l_t, d0r_t, d1l_t, d1r_t])
                    # X = X[:, bitall]
                    # z = nets[1].predict(X, batch_size=10000)
                    # z = np.log2(z / (1 - z));
                    # s = np.sum(z)
                    # print("分数是", s)
                    #
                    sur_kg_2, kg_scores_2 = attack_with_one_nd([c0l, c0r, c1l, c1r], 12, 12, None, nets[1], bits[1],
                                                               c[1])
                    # print("猜测的密钥是：", sur_kg_2)
                    # print("对应的分数是：", kg_scores_2)
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
                kg_2, kg_scores_2 = select_top_k_candidates(sur_kg_2, kg_scores_2, k[1])
                # print("猜测的密钥是：", kg_2)
                # print("猜测的分数是：", kg_scores_2)

                #     # stage 3, guess sk[31, 22], diff index is 33
                num = 0
                while True:
                    if num >= 2 ** 10:
                        num = -1
                        break

                    p0l, p0r, p1l, p1r = make_plaintext_structure(diffs[2], NBs[2])
                    # p0l1, p0r1 = sp.dec_one_round((p0l, p0r), 0)
                    # p1l1, p1r1 = sp.dec_one_round((p1l, p1r), 0)
                    # c0l1, c0r1 = sp.encrypt((p0l1, p0r1), [ks[0][0], ks[1][0]])
                    # c1l1, c1r1 = sp.encrypt((p1l1, p1r1), [ks[0][0], ks[1][0]])
                    # diffl = c0l1 ^ c1l1
                    # diffr = c0r1 ^ c1r1
                    # diff = (diffl[2], diffr[2])
                    # print("符合差分？", diff == (0x20000000, 0))

                    c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
                    # 将此结构使用正确密钥解密之后放进神经区分器里面
                    # a = [(i + 8) % 48 for i in selected_bits_3]
                    # bitall = np.concatenate((selected_bits_3, [i + 48 for i in selected_bits_3],
                    #                          [i + 96 for i in selected_bits_3], [i + 96 + 48 for i in selected_bits_3]))
                    # mask = generate_mask(a)
                    # tk1 = np.array(tk)
                    # key_bit = np.bitwise_and(tk1.item(), mask)
                    # # key_bit = np.bitwise_and(tk1, mask)
                    # d0l_t, d0r_t = sp.dec_one_round((c0l, c0r), key_bit)
                    # d1l_t, d1r_t = sp.dec_one_round((c1l, c1r), key_bit)
                    # X = sp.convert_to_binary([d0l_t, d0r_t, d1l_t, d1r_t])
                    # X = X[:, bitall]
                    # z = nets[2].predict(X, batch_size=10000)
                    # z = np.log2(z / (1 - z));
                    # s = np.sum(z)
                    # print("分数是", s)
                    #
                    sur_kg_3, kg_scores_3 = attack_with_one_nd([c0l, c0r, c1l, c1r], 12, 24, None, nets[2], bits[2],
                                                               c[2])
                    # print("猜测的密钥是：", sur_kg_3)
                    # print("对应的分数是：", kg_scores_3)
                    num += 1
                    data_num += 1
                    if len(sur_kg_3) == 0:
                        print('\r {} plaintext structures generated'.format(num), end='')
                        continue
                    else:
                        print(' ')
                        print('Stage 3: ', len(sur_kg_3), ' subkeys survive')
                        break
                if num == -1:
                    print(' ')
                    print('this trial fails.')
                    print('{} plaintext structures are generated.'.format(data_num))
                    print('the time consumption is ', time.time() - start)
                    continue
                kg_3, kg_scores_3 = select_top_k_candidates(sur_kg_3, kg_scores_3, k[2])
                # print("猜测的密钥是：", kg_3)
                # print("猜测的分数是：", kg_scores_3)
                #     # stage 3, guess sk[31, 22], diff index is 33
                num = 0
                while True:
                    if num >= 2 ** 10:
                        num = -1
                        break

                    p0l, p0r, p1l, p1r = make_plaintext_structure(diffs[3], NBs[3])
                    # p0l1, p0r1 = sp.dec_one_round((p0l, p0r), 0)
                    # p1l1, p1r1 = sp.dec_one_round((p1l, p1r), 0)
                    # c0l1, c0r1 = sp.encrypt((p0l1, p0r1), [ks[0][0], ks[1][0]])
                    # c1l1, c1r1 = sp.encrypt((p1l1, p1r1), [ks[0][0], ks[1][0]])
                    # diffl = c0l1 ^ c1l1
                    # diffr = c0r1 ^ c1r1
                    # diff = (diffl[3], diffr[3])
                    # print("符合差分？", diff == (0x20000000000, 0))

                    c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
                    # 将此结构使用正确密钥解密之后放进神经区分器里面
                    # a = [(i + 8) % 48 for i in selected_bits_4]
                    # bitall = np.concatenate((selected_bits_4, [i + 48 for i in selected_bits_4],
                    #                          [i + 96 for i in selected_bits_4], [i + 96 + 48 for i in selected_bits_4]))
                    # mask = generate_mask(a)
                    # tk1 = np.array(tk)
                    # key_bit = np.bitwise_and(tk1.item(), mask)
                    # # key_bit = np.bitwise_and(tk1, mask)
                    # d0l_t, d0r_t = sp.dec_one_round((c0l, c0r), key_bit)
                    # d1l_t, d1r_t = sp.dec_one_round((c1l, c1r), key_bit)
                    # X = sp.convert_to_binary([d0l_t, d0r_t, d1l_t, d1r_t])
                    # X = X[:, bitall]
                    # z = nets[3].predict(X, batch_size=10000)
                    # z = np.log2(z / (1 - z));
                    # s = np.sum(z)
                    # print("分数是", s)
                    #
                    sur_kg_4, kg_scores_4 = attack_with_one_nd([c0l, c0r, c1l, c1r], 12, 36, None, nets[3], bits[3],
                                                               c[3])
                    # print("猜测的密钥是：", sur_kg_4)
                    # print("对应的分数是：", kg_scores_4)
                    num += 1
                    data_num += 1
                    if len(sur_kg_4) == 0:
                        print('\r {} plaintext structures generated'.format(num), end='')
                        continue
                    else:
                        print(' ')
                        print('Stage 4: ', len(sur_kg_4), ' subkeys survive')
                        break
                if num == -1:
                    print(' ')
                    print('this trial fails.')
                    print('{} plaintext structures are generated.'.format(data_num))
                    print('the time consumption is ', time.time() - start)
                    continue
                kg_4, kg_scores_4 = select_top_k_candidates(sur_kg_4, kg_scores_4, k[3])
                # print("猜测的密钥是：", kg_4)
                # print("猜测的分数是：", kg_scores_4)
                #
                all_combinations = list(itertools.product(kg_1, kg_2, kg_3, kg_4))
                final_keys = []
                minest_distance = 48
                keyfinally = ''
                for combination in all_combinations:
                    # 将每个元素都转成2进制，其中a为
                    final_key_temp = construct_array(combination[0], combination[1], combination[2], combination[3])
                    final_key_temp = int(final_key_temp, 2)
                    distance = hamming_distance(tk, final_key_temp)
                    final_keys.append(final_key_temp)
                    if (minest_distance > distance):
                        keyfinally = final_key_temp
                        minest_distance = distance
                # print('最小距离是', minest_distance)
                # print('最小距离对应的密钥是', keyfinally)
                # 将密文使用完整密钥解密，然后使用神经区分器打分
                final_keys = np.array(final_keys)
                print(len(final_keys))
                final_keys = final_keys.astype(np.uint64)
                print(len(final_keys))
                resultkey = final_select_top_three_keys([selected_ct0, selected_ct1, selected_ct2, selected_ct3],
                                                        final_keys,
                                                        nets[4])
                resultdistance = []
                for resultkeysub in resultkey:
                    resultdistance.append(hamming_distance(tk, resultkeysub))
                resultdistancesmallest = min(resultdistance)
                with open('outputnew6.txt', 'a') as f:
                    print('使用之前的最小距离是',resultdistancesmallest,file=f)
                # 得到最高分的三个密钥之后对三个密钥分别做以下操作：
                # 反转密钥的一位两位，然后再选取最后的密钥
                flipkey = process_array(resultkey)
                print(flipkey)
                flipkey = np.array(flipkey)
                flipkey = flipkey.astype(np.uint64)
                resultkey = final_select_top_three_keys([selected_ct0, selected_ct1, selected_ct2, selected_ct3],flipkey,nets[4])
                end = time.time()
                print('the time consumption is ', end - start)
                resultdistance = []
                for resultkeysub in resultkey:
                    resultdistance.append(hamming_distance(tk, resultkeysub))
                resultdistancesmallest = min(resultdistance)
                with open('outputnew6.txt', 'a') as f:
                    print('使用之后的最小距离是', resultdistancesmallest,file=f)
                # print('实验次数', i)
                # print('最终密钥是', resultkey)
                # print('推荐密钥和最终密钥之间的距离是', resultdistancesmallest)
                # print('消耗时间', end - start)
                # print('{} plaintext structures are generated.'.format(data_num))
                time_consumption[i] = end - start
                data_consumption[i] = data_num
                with open('outputnew6.txt', 'a') as f:
                    print('实验次数', i, file=f)
                    print('最终密钥是', resultkey, file=f)
                    print('推荐密钥和最终密钥之间的距离是', resultdistancesmallest, file=f)
                    print('the time consumption is ', end - start, file=f)
                    print('the data consumption is ', data_num, file=f)
            print('average time consumption is', np.mean(time_consumption))
            print('average structure consumption is', np.mean(data_consumption))
            #     resultkey = final_select_one_key([selected_ct0, selected_ct1, selected_ct2, selected_ct3],final_keys,nets[4])
            #     end = time.time()
            #     print('the time consumption is ', end - start)
            #     resultdistance = hamming_distance(tk, resultkey)
            #     print('实验次数', i)
            #     print('最终密钥是', resultkey)
            #     print('推荐密钥和最终密钥之间的距离是', resultdistance)
            #     print('消耗时间', end - start)
            #     print('{} plaintext structures are generated.'.format(data_num))
            #     time_consumption[i] = end - start
            #     data_consumption[i] = data_num
            #     with open('output.txt', 'a') as f:
            #         print('实验次数', i, file=f)
            #         print('最终密钥是', resultkey, file=f)
            #         print('推荐密钥和最终密钥之间的距离是', resultdistance, file=f)
            #         print('the time consumption is ', end - start, file=f)
            #         print('the data consumption is ', data_num, file=f)
            # print('average time consumption is', np.mean(time_consumption))
            # print('average structure consumption is', np.mean(data_consumption))


        # 但是没有将神经网络的训练放进这里，这里需要自己去训练适合自己的神经区分器
        if __name__ == '__main__':
            nd1 = './trained_net/model_7r_diff53_acc63_depth1_num_epochs5.h5'
            nd2 = './trained_net/model_7r_diff65_acc58_depth1_num_epochs5.h5'
            nd3 = './trained_net/model_7r_diff77_acc59_depth1_num_epochs5.h5'
            nd4 = './trained_net/model_7r_diff89_acc60_depth1_num_epochs5.h5'
            nd5 = './trained_net/model_7r_diff47_depth1_num_epochs5.h5'  # 最终的筛选器
            selected_bits_1 = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
            selected_bits_2 = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
            selected_bits_3 = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            selected_bits_4 = [0, 1, 2, 3, 40, 41, 42, 43, 44, 45, 46, 47]

            # 这个差分往后一轮有较大的概率成为0000040000000000，即神经区分器的输入差分
            diff_1 = (0x2400, 0x4)
            diff_2 = (0x2400000, 0x4000)
            diff_3 = (0x2400000000, 0x4000000)
            diff_4 = (0x400000000002, 0x4000000000)
            NB_1 = [25 - i for i in range(10)]
            NB_2 = [37 - i for i in range(10)]
            NB_3 = [49 - i for i in range(10)]
            NB_4 = [61 - i for i in range(10)]
            attack_with_dual_NDs(t=100, nr=10, diffs=(diff_1, diff_2, diff_3, diff_4), NBs=(NB_1, NB_2, NB_3, NB_4),
                                 nds=(nd1, nd2, nd3, nd4, nd5),
                                 bits=(selected_bits_1, selected_bits_2, selected_bits_3, selected_bits_4),
                                 c=(10, 15, 15, 15), k=(3, 17, 17, 17))
    except Exception as e:
        print("发生错误：", e)
        # 错误处理代码
