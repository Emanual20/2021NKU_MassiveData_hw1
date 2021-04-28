package = ".\\block_stripe\\output\\result_with_package.txt"
block = "./block_stripe/output/result.txt"
basic = "./basic/output/result.txt"
results = [[], [], [], []]
name = [package, block, basic]

for i in range(0, 3):
    f = open(name[i], 'rb')
    p = f.readlines()

    for line in p:
        results[i].append(line.split()[0])

for i in range(0, 100):
    print(results[0][i], '\t', results[1][i], '\t', results[2][i], '\t',
          (results[1][i] == results[2][i]) + (results[0][i] == results[1][i] + results[2][i]))
