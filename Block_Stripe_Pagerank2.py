"""
Author: Sakura, Michelle
Date: 4/21/2021
"""

import pickle as pkl
import numpy as np


FMTO_GRAPH_PATH = ".\\WikiData.txt"
LINK_MATRIX_PREFIX = ".\\block_stripe\\data\\Link_Matrix_"
LINK_MATRIX_SUFFIX = ".matrix"
R_VECTOR_PREDIX = ".\\block_stripe\\data\\R_Vector_"
R_VECTOR_SUFFIX = ".vector"
RESULT_OUTPUT_PATH = ".\\block_stripe\\output\\result.txt"
CHECKPOINT_OUTPUT_PREFIX = ".\\block_stripe\\checkpoints\\iterate_"
CHECKPOINT_OUTPUT_SUFFIX = "_times.checkpoint"

NEW_VECTOR_PREFIX = "_new"

PRINT_NUM = 100
SAVE_CHECKPOINT_INTERVAL = 10
RANDOM_WALK_PROBABILITY = 0.15
BLOCK_NUM = 10  # identify the num of block-stripes
MAX_NODE_INDEX = 8297  # max node index process before
THRESHOLD = 1e-8

Node_Num = -1

# R = [0] * 8298
# mindex = 0


class IndexTransfer:
    block_num = BLOCK_NUM
    # max_node_index = 0
    node_num = 0
    num_in_group = 0
    num_in_last_group = 0

    def __init__(self, node_num):
        # self.block_num = block_num
        # self.max_node_index = max_node_index
        self.node_num = node_num
        self.num_in_group = self.node_num // self.block_num + 1
        self.num_in_last_group = node_num % self.num_in_group

    def calc_aim_block_index(self, aim):
        return aim // self.num_in_group

    def dest2stripedest(self, dest):
        return dest % self.num_in_group

    def stripedest2dest(self, sd, sno):
        return sd + sno * self.num_in_group

    def init_rlist(self, node_dict, sno):
        r_ret = [0] * self.num_in_group
        # for key in node_dict.keys():
        #     if self.calc_aim_block_index(key) == sno:
        #         r_ret[self.dest2stripedest(key)] = RANDOM_WALK_PROBABILITY / self.node_num
        return r_ret


def judge_block_similar(r_new, r_old):
    return (np.abs(r_new - r_old)).sum() < THRESHOLD * len(r_old)


def dump_vector(transfer, block_index, r_, new=False):
    if block_index == BLOCK_NUM - 1:
        r_ = r_[:transfer.num_in_last_group]
    if new == False:
        f_name = R_VECTOR_PREDIX + str(block_index) + R_VECTOR_SUFFIX
    else:
        f_name = R_VECTOR_PREDIX + str(block_index) + NEW_VECTOR_PREFIX + R_VECTOR_SUFFIX
    with open(f_name, "wb") as wf:
        pkl.dump(r_, wf)
    # for i in range(0, len(r_)):
    #     # print(block_index, ' ', i, '  ', block_index * 830 + i, ' ', new,' ',len(r_))
    #     R[block_index * 830 + i] = r_[i]


def load_vector(block_index, new=False):
    if new == False:
        f_name = R_VECTOR_PREDIX + str(block_index) + R_VECTOR_SUFFIX

    else:
        f_name = R_VECTOR_PREDIX + str(block_index) + NEW_VECTOR_PREFIX + R_VECTOR_SUFFIX
    with open(f_name, "rb") as f:
        r = pkl.load(f)

    return r


def load_matrix_stripe(index):
    f_name = LINK_MATRIX_PREFIX + str(index) + LINK_MATRIX_SUFFIX
    f = open(f_name, "rb")
    stripe = pkl.load(f)
    return stripe


def dump_matrix_stripe(index, stripe):
    f_name = LINK_MATRIX_PREFIX + str(index) + LINK_MATRIX_SUFFIX
    f = open(f_name, "wb")
    pkl.dump(stripe, f)


def read_graph(node_num):
    Link_Matrix = {}

    with open(FMTO_GRAPH_PATH) as file:
        lines = file.readlines()
        for line in lines:
            split = line.split()
            fm, to = eval(split[0]), eval(split[1])
            node_num = max(node_num, fm, to)

            if fm not in Link_Matrix:
                Link_Matrix[fm] = [1, [to]]
            else:
                Link_Matrix[fm][0] += 1
                Link_Matrix[fm][1].append(to)

    print("load data finish")
    return dict(sorted(Link_Matrix.items(), key=lambda item: item[0])), node_num


def dump_link_matrix(Link_Matrix, transfer):
    for i in range(0, BLOCK_NUM):
        Link_Matrix_List = {}
        for fm in Link_Matrix:
            Link_Matrix_List[fm] = [Link_Matrix[fm][0],
                                    [elem for elem in Link_Matrix[fm][1] if
                                     elem < (i + 1) * transfer.num_in_group and elem >= i * transfer.num_in_group]]
            if len(Link_Matrix_List[fm][1]) == 0:
                del Link_Matrix_List[fm]
        dump_matrix_stripe(i, Link_Matrix_List)


def load_data():
    Node_Num = -1
    Link_Matrix, Node_Num = read_graph(Node_Num)
    Node_Num += 1  # ?????????index=0??????
    transfer = IndexTransfer(Node_Num)
    dump_link_matrix(Link_Matrix, transfer)

    return transfer


def normalize_list_randomwalk(vector_sum, r_random, transfer):  # ?????????
    flag = 1
    for block_index in range(0, BLOCK_NUM):
        r_new = load_vector(block_index, True)
        r_new = (1 - RANDOM_WALK_PROBABILITY) * r_new / vector_sum + r_random[:len(r_new)]
        r_old = load_vector(block_index)
        if flag == 1:
            flag = judge_block_similar(r_new, r_old)
        if block_index < BLOCK_NUM:
            dump_vector(transfer, block_index, r_new)
    return flag


def normalize_list_randomwalk2(vector_sum, r_random, transfer):
    flag = 1
    for block_index in range(0, BLOCK_NUM):
        r_new = load_vector(block_index, True)
        r_new = (1 - RANDOM_WALK_PROBABILITY) * (r_new + (1 - vector_sum) / transfer.node_num) + r_random[:len(r_new)]
        r_old = load_vector(block_index)
        if flag == 1:
            flag = judge_block_similar(r_new, r_old)
        if block_index < BLOCK_NUM:
            dump_vector(transfer, block_index, r_new)
    return flag


def matrix_block_multiple(matrix_stripe, block_index, transfer, r_new):  # stripe???v????????????
    r_old = load_vector(block_index)

    # ??????
    for i in dict(filter(
            lambda x: x[0] >= block_index * transfer.num_in_group and x[0] < (block_index + 1) * transfer.num_in_group,
            matrix_stripe.items())):  # ??????????????????stripe
        if i >= (block_index + 1) * transfer.num_in_group:
            break
        for to in matrix_stripe[i][1]:
            to_block_index = transfer.dest2stripedest(to)
            fm_block_index = transfer.dest2stripedest(i)
            r_new[to_block_index] += r_old[fm_block_index] / matrix_stripe[i][0]
    return r_new


def matrix_stripe_multiple(stripe_index, transfer):  # stripe???v???
    matrix_stripe = load_matrix_stripe(stripe_index)
    r_new = np.zeros(transfer.num_in_group)
    for block_index in range(0, BLOCK_NUM):
        r_new = matrix_block_multiple(matrix_stripe, block_index, transfer, r_new)

    dump_vector(transfer, stripe_index, r_new, new=True)
    return r_new.sum()


def matrix_multiple(transfer):
    sum_list = np.zeros(BLOCK_NUM)
    for stripe_index in range(0, BLOCK_NUM):
        sum_list[stripe_index] = matrix_stripe_multiple(stripe_index, transfer)
    return sum_list.sum()


def initialize(transfer):
    if transfer.node_num == 0:
        return
    r = np.ones(transfer.num_in_group) / transfer.node_num
    for block_index in range(0, BLOCK_NUM):
        dump_vector(transfer, block_index, r)


def output_result_list(transfer):
    # ????????????????????????100???
    print("start calculating final rank")
    results = {}
    for i in range(0, BLOCK_NUM):
        result = load_vector(i)
        sort_result = dict(zip(np.argsort(-result)[:PRINT_NUM] + i * transfer.num_in_group,
                               sorted(result, reverse=True)[:PRINT_NUM]))
        results.update(sort_result)
    results = dict(sorted(results.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
    print("start ouput")
    with open(RESULT_OUTPUT_PATH, "w") as f:
        for i, key in enumerate(results):
            f.write(str(key)+ '\t'+str(results[key])+'\n')
            if i == 99:
                break

    # for i, key in enumerate(results):
    #     if i ==PRINT_NUM:
    #         break
    #     print(str(key) + '\t' + str(results[key]))


def block_stripe_pagerank(transfer):
    print("basic pangerank")
    initialize(transfer)
    r_random = np.ones(transfer.num_in_group) / \
               transfer.node_num * RANDOM_WALK_PROBABILITY
    print("initialize finish")
    flag = 0
    round = 0
    while not flag:  # ??????????????????
        # print(round, '\t', R[6634])
        vector_sum = matrix_multiple(transfer)  # ???????????????????????????
        flag = normalize_list_randomwalk2(vector_sum, r_random, transfer)
        round += 1

    print("multiple finish")


if __name__ == '__main__':
    print("### Block-Stripe Version running.. ###")
    transfer = load_data()
    block_stripe_pagerank(transfer)
    output_result_list(transfer)
