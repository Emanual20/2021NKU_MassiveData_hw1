# Edit Log
## 2021/4/6 Sakura
## 2021/4/19 Michelle

### 修改normalize_list
原：
```python
def normalize_list(list_to_norm):
    norm1 = 0
    for item in list_to_norm: # 使用求和函数
        norm1 += item
    list_to_return = list_to_norm # list等号是浅拷贝
    for i in range(0, len(list_to_return)):
        list_to_return[i] /= norm1
    return list_to_return
```
修改为
```python
def normalize_list(list_to_norm):
    list_to_norm=[i/sum(list_to_norm) for i in list_to_norm]
    return list_to_norm
```

### 修改 block_stripe_pagerank

```python
for block_index in range(0, transfer.block_num):
    rdata_name = R_VECTOR_PREDIX + str(block_index) + R_VECTOR_SUFFIX
    with open(rdata_name, "rb") as rf:
        r_stripe = pkl.load(rf)
        if block_index != transfer.block_num - 1:
            r_newly += r_stripe
        else:
            r_newly += r_stripe[0:transfer.dest2stripedest(transfer.max_node_index) + 1]


r_newly=normalize_list(r_newly)
```
修改为
```python

for block_index in range(0, transfer.block_num):
    rdata_name = R_VECTOR_PREDIX + str(block_index) + R_VECTOR_SUFFIX
    with open(rdata_name, "rb") as rf:
        r_stripe = pkl.load(rf)
        r_newly += r_stripe

r_newly=normalize_list(r_newly[:transfer.max_node_index+1])
```

官方库没有未出现的点