from numpy import *
import operator

def create_dataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataset,labels,k):
    dataset_size=dataset.shape[0]
    print("dataset_size: {0}".format(dataset_size))
    diff_mat=tile(inX,(dataset_size,1)) - dataset
    sq_diff_mat=diff_mat**2
    sq_distances=sq_diff_mat.sum(axis=1)
    distances=sq_distances**0.5
    sorted_dist_indicies=distances.argsort()
    class_count={}

    for i in range(k):
        vote_i_label=labels[sorted_dist_indicies[i]]
        class_count[vote_i_label]=class_count.get(vote_i_label,0)+1
    sorted_class_count=sorted(class_count.iteritems(),
                              key=operator.itemgetter(1),reverse=True)
    return sorted_class_count[0][0]
