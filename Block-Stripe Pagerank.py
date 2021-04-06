"""
Author: Sakura
Date: 4/6/2021
"""
import sys
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

FMTO_GRAPH_PATH = ".\\WikiData.txt"
LINK_MATRIX_PREFIX = ".\\block_stripe\\data\\Link_Matrix_"
LINK_MATRIX_SUFFIX = ".matrix"
R_VECTOR_PREDIX = ".\\block_stripe\\data\\R_Vector_"
R_VECTOR_SUFFIX = ".vector"
RESULT_OUTPUT_PATH = ".\\block_stripe\\output\\result.txt"
CHECKPOINT_OUTPUT_PREFIX = ".\\block_stripe\\checkpoints\\iterate_"
CHECKPOINT_OUTPUT_SUFFIX = "_times.checkpoint"

PRINT_NUM = 100
SAVE_CHECKPOINT_INTERVAL = 10
RANDOM_WALK_PROBABILITY = 0.15
BLOCK_NUM = 10  # identify the num of block-stripes
MAX_NODE_INDEX = 8297  # max node index process before
THRESHOLD = 1e-8


class IndexTransfer:
    block_num = 0
    max_node_index = 0
    node_num = 0
    num_in_group = 0

    def __init__(self, block_num, max_node_index, node_num):
        self.block_num = block_num
        self.max_node_index = max_node_index
        self.node_num = node_num
        self.num_in_group = self.max_node_index // self.block_num + 1

    def calc_aim_block_index(self, aim):
        return aim // self.num_in_group

    def dest2stripedest(self, dest):
        return dest - dest // self.num_in_group * self.num_in_group

    def stripedest2dest(self, sd, sno):
        return sd + sno * self.num_in_group

    def init_rlist(self, node_dict, sno):
        r_ret = [0] * self.num_in_group
        for key in node_dict.keys():
            if self.calc_aim_block_index(key) == sno:
                r_ret[self.dest2stripedest(key)] = RANDOM_WALK_PROBABILITY / self.node_num
        return r_ret


def calc_aim_block_index(item, max_index, block_num):
    return item // (((max_index + 1) // block_num) + 1)


def load_data():
    max_nodenum, Link_Matrix, temp_dict = 0, [[i, 0, []] for i in range(0, MAX_NODE_INDEX + 1)], {}
    with open(FMTO_GRAPH_PATH) as file:
        lines = file.readlines()
        for line in lines:
            split = line.split()
            fm, to = eval(split[0]), eval(split[1])
            max_nodenum = max(max_nodenum, max(fm, to))
            Link_Matrix[fm][1] += 1
            Link_Matrix[fm][2].append(to)
            temp_dict[fm] = temp_dict[to] = 1
    print('load data finish..')

    Link_Matrix_List = [
        [[i, 0, []] for i in range(0, MAX_NODE_INDEX + 1)]
        for j in range(0, BLOCK_NUM)]
    for entry in Link_Matrix:
        for item in entry[2]:
            aim_block_index = calc_aim_block_index(item, MAX_NODE_INDEX, BLOCK_NUM)
            # print(aim_block_index)
            Link_Matrix_List[aim_block_index][entry[0]][1] = entry[1]
            Link_Matrix_List[aim_block_index][entry[0]][2].append(item)

    for i in range(0, BLOCK_NUM):
        f_name = LINK_MATRIX_PREFIX + str(i) + LINK_MATRIX_SUFFIX
        f = open(f_name, "wb")
        pkl.dump(Link_Matrix_List[i], f)
    print('punch data into blocks finish..')

    return temp_dict, max_nodenum, len(temp_dict.keys())


def normalize_list(list_to_norm):
    norm1 = 0
    for item in list_to_norm:
        norm1 += item
    list_to_return = list_to_norm
    for i in range(0, len(list_to_return)):
        list_to_return[i] /= norm1
    return list_to_return


def calc_2listdist(list1, list2):
    if len(list1) != len(list2):
        raise Exception("lists' length don't figure: ", len(list1), "!=", len(list2))
    ret = 0
    for i in range(0, len(list1)):
        ret += abs(list1[i] - list2[i])
    return ret


def init_rlist(max_node_index, node_num, node_dict):
    r_ret = [0] * (max_node_index + 1)
    for key in node_dict.keys():
        r_ret[key] = RANDOM_WALK_PROBABILITY / node_num
    return r_ret


def block_stripe_pagerank(max_node_index, node_num, node_dict, transfer):
    r_old = init_rlist(max_node_index, node_num, node_dict)
    round = 0

    while True:
        for block_index in range(0, transfer.block_num):
            f_name = R_VECTOR_PREDIX + str(block_index) + R_VECTOR_SUFFIX
            r_stripe = transfer.init_rlist(node_dict, block_index)
            with open(f_name, "wb") as f:
                pkl.dump(r_stripe, f)

        for block_index in range(0, transfer.block_num):
            rdata_name = R_VECTOR_PREDIX + str(block_index) + R_VECTOR_SUFFIX
            r_now_stripe = []
            with open(rdata_name, "rb") as rf:
                r_now_stripe = pkl.load(rf)
                f_name = LINK_MATRIX_PREFIX + str(block_index) + LINK_MATRIX_SUFFIX
                with open(f_name, "rb") as f:
                    link_matrix_stripe = pkl.load(f)
                    for entry in link_matrix_stripe:
                        for destination in entry[2]:
                            r_now_stripe[transfer.dest2stripedest(destination)] \
                                += (1 - RANDOM_WALK_PROBABILITY) * r_old[entry[0]] / entry[1]

            with open(rdata_name, "wb") as wf:
                pkl.dump(r_now_stripe, wf)

        r_newly = []
        for block_index in range(0, transfer.block_num):
            rdata_name = R_VECTOR_PREDIX + str(block_index) + R_VECTOR_SUFFIX
            with open(rdata_name, "rb") as rf:
                r_stripe = pkl.load(rf)
                if block_index != transfer.block_num - 1:
                    r_newly += r_stripe
                else:
                    r_newly += r_stripe[0:transfer.dest2stripedest(transfer.max_node_index) + 1]

        r_newly = normalize_list(r_newly)

        # make checkpoints
        if not round % SAVE_CHECKPOINT_INTERVAL:
            checkpoint_save_path = CHECKPOINT_OUTPUT_PREFIX + str(round) + CHECKPOINT_OUTPUT_SUFFIX
            with open(checkpoint_save_path, 'wb') as f:
                pkl.dump(r_newly, f)
            print("make checkpoints no.", round, " finish")

        # check if it comes to convergence
        if calc_2listdist(r_newly, r_old) < THRESHOLD:
            print("convergence at round", round)
            return r_newly
        else:
            r_old = r_newly
            round += 1


def output_result_list(results):
    checksum, dic = 0, {}

    for index in range(0, len(results)):
        if results[index] != 0:
            dic[index] = results[index]
            checksum += results[index]

    if abs(checksum - 1) > THRESHOLD:
        raise Exception("calc error..")

    with open(RESULT_OUTPUT_PATH, "w") as f:
        tot = 0
        for entry in sorted([(value, key) for (key, value) in dic.items()], reverse=True):
            f.write(str(entry[1]) + '\t' + str(entry[0]) + '\t\n')
            tot += 1
            if tot >= PRINT_NUM:
                print("write finish..")
                return


if __name__ == '__main__':
    print("### Block-Stripe Version running.. ###")
    Node_dict, Max_Node_Index, Node_Num = load_data()
    # print(Max_Node_Index, Node_Num)
    transfer = IndexTransfer(BLOCK_NUM, Max_Node_Index, Node_Num)
    r_new = block_stripe_pagerank(Max_Node_Index, Node_Num, Node_dict, transfer)
    output_result_list(r_new)
