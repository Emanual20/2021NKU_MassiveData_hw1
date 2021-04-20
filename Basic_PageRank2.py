from Basic_Pagerank import *

THRESHOLD = 1e-6
MAX_ROUND = 100


def initialiaze(Node_Num):
    r_ret = [1 / float(Node_Num)] * Node_Num
    return r_ret


def matrix_multiple(Node_Num,r_old):
    r_newly=[0]* Node_Num
    f_name = LINK_MATRIX_PREFIX + LINK_MATRIX_SUFFIX
    with open(f_name, "rb") as f:
        link_matrix_stripe = pkl.load(f)
        for entry in link_matrix_stripe:
            for destination in entry[2]:
                r_newly[destination] += (1 - RANDOM_WALK_PROBABILITY) * r_old[entry[0]] / entry[1]

    return r_newly


def random_walk(r, Node_Num):
    r_ret = [i * (1 - RANDOM_WALK_PROBABILITY) + RANDOM_WALK_PROBABILITY / float(Node_Num) for i in r]
    return r_ret


def basic_pagerank(Node_Num, Node_dict):
    # 初始化
    r_old = initialiaze(Node_Num)
    round = 0
    while True:
        # 矩阵乘法得到r_new
        r_new = matrix_multiple(Node_Num,r_old)
        # 标准化
        r_new = normalize_list(r_new)
        # r_new*beta+（1-beta）/N
        r_new = random_walk(r_new, Node_Num)
        # 比较
        if calc_2listdist(r_old, r_new) <= THRESHOLD * Node_Num or round == MAX_ROUND:
            print("convergence at round", round)
            return r_new
        else:
            r_old = r_new
            round += 1

def output_result(result):
    result=np.array(result)
    sort_result=dict(zip(np.argsort(-result)[:PRINT_NUM+1], sorted(result, reverse=True)[:PRINT_NUM+1]))
    # print(sort_result)
    for key in sort_result:
        print(key,sort_result[key])

if __name__ == '__main__':
    print("### Naive Version(without block-stripe) running.. ###")
    Node_dict, Max_Node_Index, Node_Num = load_data()
    r_new = basic_pagerank(Max_Node_Index + 1, Node_dict)
    output_result(r_new)
    # print(r_new)
