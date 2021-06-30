import sys
import time
import numpy as np


class DBSCAN:
    def __init__(self, data, Eps, MinPts):
        self.data_cnt = len(data)
        self.epsilon = Eps
        self.MinPts = MinPts
        self.visited = np.full((self.data_cnt), False)
        dist = np.zeros((self.data_cnt, self.data_cnt))
        for i in range(self.data_cnt):
            print(str(i) + "  is done !!!")
            for j in range(self.data_cnt):
                dist[i][j] = np.sqrt((data[i][0]-data[j][0])**2 +
                                     (data[i][1]-data[j][1])**2)
        self.dist = dist
        self.label = np.full((self.data_cnt), 0)
        # 현재 cluster index
        self.index = 0

    def get_clusters(self):
        for idx in range(self.data_cnt):
            if not self.visited[idx]:
                self.visited[idx] = True
                neighborhood = np.where(self.dist[idx] <= self.epsilon)[0]
                if len(neighborhood) < self.MinPts:
                    self.label[idx] = -1
                else:
                    # print(neighborhood)
                    self.index += 1
                    self.label[idx] = self.index
                    self.expand(neighborhood.tolist())

    def expand(self, neighborhood):
        for n in neighborhood:
            if self.label[n] == 0:
                self.label[n] = self.index

            if not self.visited[n]:
                self.visited[n] = True
                cur_neighborhood = np.where(
                    self.dist[n] <= self.epsilon)[0].tolist()
                if len(cur_neighborhood) >= self.MinPts:
                    neighborhood.extend(cur_neighborhood)


if __name__ == '__main__':
    start = time.time()
    # check correct parameter
    if len(sys.argv) != 5:
        print("Insufficient arguments")
        sys.exit()

    data_path = sys.argv[1]
    input_num = int(data_path[5])
    n_clusters = int(sys.argv[2])
    Eps = float(sys.argv[3])
    MinPts = int(sys.argv[4])

    data_file = open(data_path, 'r')
    lines = data_file.readlines()
    data_file.close()

    data = []
    for line in lines:
        _idx, x, y = line.split('\t')
        data.append([float(x), float(y)])
        print(str(_idx) + "  is done !!!")

    clustering_method = DBSCAN(data, Eps, MinPts)
    clustering_method.get_clusters()
    labels, counts = np.unique(clustering_method.label, return_counts=True)
    label_cnt = sorted(list(zip(labels, counts)),
                       key=lambda x: x[1], reverse=True)
    k = 0
    for i in range(n_clusters+1):
        if label_cnt[i][0] == -1:
            continue
        else:
            output_file = open("input"+str(input_num) +
                               "_cluster_" + str(k) + ".txt", 'w')
            points = np.where(clustering_method.label == label_cnt[i][0])[0]
            for point in points:
                print(point, file=output_file)
            output_file.close()
            k += 1

    print("time: ", time.time()-start)
