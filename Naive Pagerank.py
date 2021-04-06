"""
Author: Sakura
Date: 4/6/2021
"""
import sys
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

FMTO_GRAPH_PATH = ".\\WikiData.txt"
LINK_MATRIX_PREFIX = ".\\naive\\data\\Link_Matrix_"
LINK_MATRIX_SUFFIX = ".matrix"
R_VECTOR_PREDIX = ".\\naive\\data\\R_Vector_"
R_VECTOR_SUFFIX = ".vector"
RESULT_OUTPUT_PATH = ".\\naive\\output\\result.txt"
CHECKPOINT_OUTPUT_PREFIX = ".\\naive\\checkpoints\\iterate_"
CHECKPOINT_OUTPUT_SUFFIX = "_times.checkpoint"

PRINT_NUM = 100
SAVE_CHECKPOINT_INTERVAL = 10
RANDOM_WALK_PROBABILITY = 0.15
BLOCK_NUM = 10  # identify the num of block-stripes
MAX_NODE_INDEX = 8297  # max node index process before
THRESHOLD = 1e-8


def calc_aim_block_index(item, max_index, block_num):
    return item // (((max_index + 1) // block_num) + 1);


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

    f_name = LINK_MATRIX_PREFIX + LINK_MATRIX_SUFFIX
    f = open(f_name, "wb")
    pkl.dump(Link_Matrix, f)

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
        raise Exception("lists' length don't figure")
    ret = 0
    for i in range(0, len(list1)):
        ret += abs(list1[i] - list2[i])
    return ret


def init_rlist(max_node_index, node_num, node_dict):
    r_ret = [0] * (max_node_index + 1)
    for key in node_dict.keys():
        r_ret[key] = RANDOM_WALK_PROBABILITY / node_num
    return r_ret


def naive_pagerank(max_node_index, node_num, node_dict):
    r_old = init_rlist(max_node_index, node_num, node_dict)
    round = 0

    while True:
        r_newly = init_rlist(max_node_index, node_num, node_dict)
        f_name = LINK_MATRIX_PREFIX + LINK_MATRIX_SUFFIX
        with open(f_name, "rb") as f:
            link_matrix_stripe = pkl.load(f)
            for entry in link_matrix_stripe:
                for destination in entry[2]:
                    r_newly[destination] += (1 - RANDOM_WALK_PROBABILITY) * r_old[entry[0]] / entry[1]
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
    print("### Naive Version(without block-stripe) running.. ###")
    Node_dict, Max_Node_Index, Node_Num = load_data()
    r_new = naive_pagerank(Max_Node_Index, Node_Num, Node_dict)
    output_result_list(r_new)