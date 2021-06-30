import sys
import numpy as np
import pandas as pd


class Node:
    def __init__(self, data):
        self.data = data
        self.parent = None
        self.test_attr = None
        self.test_info = None
        self.classification = None
        self.children = []

    def print_Node(self):
        print("data", self.data)
        print("parent", self.parent)
        print("test_attr", self.test_attr)
        print("test_info", self.test_info)
        print("classification", self.classification)
        print("children", self.children)


def information_gain(data, class_label, attr):
    total_entropy = 0
    labels, label_cnts = np.unique(data[class_label], return_counts=True)
    for i in range(len(labels)):
        if label_cnts[i] != 0:
            total_entropy += -((label_cnts[i]/np.sum(label_cnts)) *
                               np.log2(label_cnts[i]/np.sum(label_cnts)))

    weighted = 0
    attr_samples, attr_cnts = np.unique(data[attr], return_counts=True)

    for i in range(len(attr_samples)):
        for j in range(len(labels)):
            p_ij = (len(data[((data[class_label] == labels[j]) & (
                data[attr] == attr_samples[i]))])) / attr_cnts[i]
            if p_ij != 0:
                weighted += \
                    - (attr_cnts[i]/np.sum(attr_cnts)) * p_ij * np.log2(p_ij)

    return total_entropy - weighted


def gain_ratio(data, class_label, attr):
    gain = information_gain(data, class_label, attr)

    attr_samples, attr_cnts = np.unique(data[attr], return_counts=True)
    split_info = 0
    for i in range(len(attr_samples)):
        if attr_cnts[i] != 0:
            split_info += -((attr_cnts[i]/np.sum(attr_cnts)) *
                            np.log2(attr_cnts[i]/np.sum(attr_cnts)))

    return gain / split_info


def build_tree(cur_node, data, class_label, rest_attr):
    global root

    # sample이 모두 같은 class label
    if len(np.unique(data[class_label])) == 1:
        if cur_node.parent:
            cur_node.test_attr = cur_node.parent.test_attr
        cur_node.classification = np.unique(data[class_label])[0]
        return root
    # sample이 없을 때
    elif len(data) == 0:
        labels, counts = np.unique(
            cur_node.parent.data[class_label], return_counts=True)
        cur_node.classification = labels[counts == max(counts)][0]
        return root
    # 남은 attr이 없을 때
    elif len(rest_attr) == 0:
        labels, counts = np.unique(data[class_label], return_counts=True)
        cur_node.classification = labels[counts == max(counts)][0]
        return root

    gain_ratios = [[gain_ratio(data, class_label, rest_attr[i]), rest_attr[i]]
                   for i in range(len(rest_attr))]
    test_attr = max(gain_ratios)[1]
    cur_node.test_attr = test_attr

    divided = np.unique(data[test_attr])
    rest_attr = rest_attr.drop(test_attr)

    for i in range(len(divided)):
        child_Node = Node(data[data[test_attr] == divided[i]])
        child_Node.test_info = divided[i]
        child_Node.parent = cur_node
        cur_node.children.append(child_Node)
        build_tree(child_Node, child_Node.data, class_label, rest_attr)

    return root


def print_tree(cur_node):
    cur_node.print_Node()

    for i in range(len(cur_node.children)):
        print_tree(cur_node.children[i])


def print_tree2(cur_node):
    if len(cur_node.children) == 0 and cur_node.classification == 'good':
        cur_node.print_Node()

    for i in range(len(cur_node.children)):
        print_tree2(cur_node.children[i])


def tree_test(df_test):
    global root

    for i in range(len(df_test)):
        cur_node = root
        sample = df_test.iloc[i]

        while True:
            if cur_node.classification:
                df_test[class_label][i] = cur_node.classification
                break
            else:
                test_attr = cur_node.test_attr
                done = False
                for j in range(len(cur_node.children)):
                    if sample[test_attr] == cur_node.children[j].test_info:
                        cur_node = cur_node.children[j]
                        break
                    # 분류 안되는 것
                    elif j == len(cur_node.children)-1 and sample[test_attr] != cur_node.children[j].test_info:
                        if len(cur_node.data[class_label]) > 0:
                            labels, counts = np.unique(
                                cur_node.data[class_label], return_counts=True)
                        else:
                            labels, counts = np.unique(
                                df_train[class_label], return_counts=True)
                        df_test[class_label][i] = labels[counts == max(
                            counts)][0]
                        done = True

                if done:
                    break


if __name__ == '__main__':
    # getting parameters
    training_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]

    # check correct parameter
    if len(sys.argv) != 4:
        print("Insufficient arguments")
        sys.exit()

    df_train = pd.read_table(training_path)
    attrs = df_train.columns
    class_label = attrs[-1]
    attrs = attrs[:-1]

    root = Node(df_train)
    build_tree(root, df_train, class_label, attrs)

    df_test = pd.read_table(test_path)
    df_test[class_label] = np.nan

    tree_test(df_test)
    df_test.to_csv(output_path, sep='\t', index=False)
