from Block_Stripe_Pagerank import *

MAX_NODE_NUM = MAX_NODE_INDEX + 1


def initialiaze():
    initial_rate = [1 / float(MAX_NODE_NUM)] * MAX_NODE_NUM
    for block_index in range(0, transfer.block_num):
        f_name = R_VECTOR_PREDIX + str(block_index) + R_VECTOR_SUFFIX
        with open(f_name, "wb") as f:
            pkl.dump(initial_rate, f)
    return


def sparse_matrix_multiple():  # 待写,取出
    r_new = []
    return r_new
    pass


def block_stripe_pagerank2(max_node_index, node_num, node_dict, transfer):
    # r_old初始化
    initialiaze()
    # 矩阵乘法
    r_new


if __name__ == '__main__':
    print("### Block-Stripe Version 2 running.. ###")
    Node_dict, Max_Node_Index, Node_Num = load_data()
    # print(Max_Node_Index, Node_Num)
    transfer = IndexTransfer(BLOCK_NUM, Max_Node_Index, Node_Num)
