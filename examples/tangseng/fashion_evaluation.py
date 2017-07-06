# -*- coding: utf-8 -*-
"""Fashion evaluation module.

Example:
    The following illustrates the basic usage.

        import numpy as np
        truth = np.array([[142, 23], [124, 34], [342, 31]])
        prediction = np.array([[132, 30], [134, 35], [322, 21]])
        print evaluate_pck(truth, prediction)

"""

import numpy as np
import sklearn.metrics


def compute_confusion(annotation, prediction, labels):
    """
    Compute a confusion matrix.
    """
    truth = np.array(annotation, dtype=np.int32).flatten()
    prediction = np.array(prediction, dtype=np.int32).flatten()
    assert len(truth) == len(prediction)
    return sklearn.metrics.confusion_matrix(truth, prediction,
                                            [i for i in xrange(len(labels))])

def compute_IOU(annotation, prediction,labels):
    """
    Compute class IOU of each class, and mean IOU
    Return an array of length number-of-labels +1
    First number-of-labels elements is class IOU of each class
    The last elements is mean IOU of all class
    """
    truth = np.array(annotation, dtype=np.int32)
    labelmap = np.array(prediction, dtype=np.int32)#.argmax(axis = 0)
    nlabel = len(labels)

    intersectCount = np.zeros(nlabel)
    unionCount = np.zeros(nlabel)
    classIOU = np.zeros(nlabel)
    for i in xrange(1,nlabel):
        segClass = labelmap==i        
        truthClass = truth==i
        intersect = np.logical_and(segClass,truthClass) 
        intersectCount[i] = int(sum(intersect[:]))
        union = np.logical_or(segClass,truthClass)
        unionCount[i] = int(sum(union[:]))
        if unionCount[i]!=0:
            classIOU[i] = intersectCount[i]/unionCount[i]
    meanIOU = sum(intersectCount)/sum(unionCount)
    #print meanIOU
    classIOU = classIOU*100
    meanIOU = meanIOU*100
    return np.append(classIOU,meanIOU)


def evaluate_fashion_confusion(confusion):
    """
    Evaluate various performance measures from the confusion matrix.
    """
    accuracy = (np.sum(np.diag(confusion)).astype(np.float64) /
                np.sum(confusion.flatten()))
    precision = np.divide(np.diag(confusion).astype(np.float64),
                          np.sum(confusion, axis=0))
    recall = np.divide(np.diag(confusion).astype(np.float64),
                       np.sum(confusion, axis=1))
    f1 = np.divide(2 * np.diag(confusion).astype(np.float64),
                   np.sum(confusion, axis=0) + np.sum(confusion, axis=1))
    # Remove the background from consideration.
    fg_accuracy = (np.sum(np.diag(confusion)[1:]).astype(np.float64) /
                   np.sum(confusion[1:,:].flatten()))
    result = {
        'accuracy': float(accuracy),
        'average_precision': float(np.nanmean(precision)),
        'average_recall': float(np.nanmean(recall)),
        'average_f1': float(np.nanmean(f1)),
        'fg_accuracy': float(fg_accuracy),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist()
        }
    return result


def find_keypoints_in_posemap(posemap):
    """Find the keypoint-locations in posemap.

    Args:
        posemap (numpy.array): P-by-H-by-W array of pose map, where `P` is the
            number of keypoints, `H` is the height, and `W` is the width.

    Returns:
        numpy.array: P-by-2 array of keypoint locations.

    """
    y = posemap.max(axis=2).argmax(axis=1)
    x = posemap.max(axis=1).argmax(axis=1)
    return np.array([x, y], dtype=np.float32).transpose()


def evaluate_pck(truth, prediction, threshold=0.1):
    """Compute PCK measure between ground truth and prediction.

    Args:
        truth (numpy.array): P-by-2-by-N array of ground truth pose, where `P`
            is the number of keypoints and `N` is the number of images.
        prediction (numpy.array): P-by-2-by-N array of predicted pose.
        threshold (float): Optional threshold to decide tolerance. Smaller
            threshold makes prediction harder. Default to 0.1.

    Returns:
        numpy.array: P-by-1 array of probability for each keypoints.

    """
    scale = np.max(truth.max(axis=0) - truth.min(axis=0) + 1, axis=0)
    distance = np.sqrt(np.sum((prediction - truth) ** 2, axis=1))
    true_positive = distance <= threshold * scale
    if true_positive.ndim == 1:
        return true_positive
    else:
        return true_positive.astype(np.float).mean(axis=1)


def _test_evaluate_pck():
    """Test `evaluate_pck` functionality.
    """
    size = (14, 2, 100)
    truth = np.random.randint(10, size=size)
    prediction = truth + np.random.randn(size[0], size[1], size[2])
    print evaluate_pck(truth, prediction)


def _test_evaluate_pck2():
    """Test `evaluate_pck` functionality.
    """
    size = (14, 2)
    truth = find_keypoints_in_posemap(np.random.rand(14, 300, 200))
    prediction = truth + np.random.randn(size[0], size[1])
    print evaluate_pck(truth, prediction)


if __name__ == "__main__":
    _test_evaluate_pck()
    _test_evaluate_pck2()
