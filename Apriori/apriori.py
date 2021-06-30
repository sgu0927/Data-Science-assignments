import sys
import time
from itertools import combinations


def generate_candidates(k, previous):
    itemset = []
    for p in previous:
        itemset.extend(p)
    # 길이 2 itemset
    if k == 1:
        comb = combinations(itemset, 2)
        candidates = []
        for c in comb:
            candidates.append(list(c))
        return candidates
    # 길이 3 이상 itemset
    else:
        candidates = []
        itemset = list(set(itemset))
        comb = combinations(itemset, k+1)
        for c in comb:
            candidates.append(list(c))
        return candidates


def calc_confidence(k, itemset, support):
    ret = []
    for i in range(1, k+1):
        comb = list(combinations(itemset, i))
        for j in range(len(comb)):
            left = list(comb[j])
            right = list(set(itemset) - set(left))
            freq_left = 0
            cnt = 0

            for y in range(total_transaction):
                countable = True
                for x in range(i):
                    if left[x] not in transactions[y]:
                        countable = False
                        break
                if countable:
                    freq_left += 1
                    for x in range(k+1 - i):
                        if right[x] not in transactions[y]:
                            countable = False
                            break
                if countable:
                    cnt += 1
            ret.append([set(left), set(right), support,
                        round((cnt/freq_left)*100, 2)])
    return ret


def pruning(k, candidates, itemset):
    # length k subset for Test length k+1
    test_itemset = []
    for candidate in candidates:
        test_itemset.append([candidate])
        comb = combinations(candidate, k)
        for c in comb:
            test_itemset[-1].append(list(c))
    check = [True for _ in range(len(test_itemset))]
    # return info with support, confidence
    info = []
    next_candidate = []

    # subset이 previous에 있는지 확인 + min_support 적용
    for i, test in enumerate(test_itemset):
        T = test[1:]
        for j in range(len(T)):
            if T[j] not in itemset:
                check[i] = False
                break

        if check[i]:
            min_cnt = min_support * total_transaction
            cur = test[0]
            cnt = 0

            for j in range(total_transaction):
                transaction = transactions[j]
                has_all = True
                for x in range(k+1):
                    if cur[x] not in transaction:
                        has_all = False
                        break
                if has_all:
                    cnt += 1

            if cnt < min_cnt:
                check[i] = False
            else:
                info.extend(calc_confidence(
                    k, cur, round((cnt/total_transaction)*100, 2)))
                next_candidate.append(cur)

    return next_candidate, info


def write_output_txt(info):
    for i in range(len(info)):
        line = str(info[i][0]) + '\t' + str(info[i][1]) + \
            '\t' + str('%.2f' % info[i][2]) + '\t' + \
            str('%.2f' % info[i][3])
        print(line, file=output_txt)


if __name__ == '__main__':
    start = time.time()
    # getting parameters
    min_support = float(sys.argv[1]) * (0.01)
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    # check correct parameter
    if len(sys.argv) != 4:
        print("Insufficient arguments")
        sys.exit()

    # Read from input text
    transactions = []
    all_items = []
    input_text = open(input_path, 'r')
    lines = input_text.readlines()
    total_transaction = len(lines)
    input_text.close()

    for line in lines:
        items = list(map(int, line.rstrip().split("\t")))
        all_items.extend(items)
        transactions.append(items)

    nums = list(set(all_items))

    # Find 1-itemset
    # There is no duplication of items in each transaction
    one_itemset = []
    for num in nums:
        if all_items.count(num) >= min_support * total_transaction:
            one_itemset.append([num])

    # Apriori Algorithm
    k = 1
    previous_itemset = one_itemset
    output_txt = open(output_path, 'w')
    while True:
        # length k+1 구하는 과정
        # self-joining
        C = generate_candidates(k, previous_itemset)

        if not C:
            break

        # pruning (association rule support,confidence와 함께 return / next candidate return)
        previous_itemset, info = pruning(k, C, previous_itemset)

        if info:
            write_output_txt(info)
        if previous_itemset:
            k += 1
        else:
            break

    output_txt.close()
    print("time: ", time.time()-start)
