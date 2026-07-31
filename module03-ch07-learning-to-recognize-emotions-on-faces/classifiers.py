from abc import ABCMeta, abstractmethod
import numpy as np
import cv2


class Classifier(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass
    @abstractmethod
    def evaluate(self, X_test, y_test, visualize=False):
        pass


class MultiLayerPerceptron(Classifier):
    def __init__(self, layer_sizes, class_labels, params=None):
        self.num_features = int(layer_sizes[0])
        self.num_classes = int(layer_sizes[-1])
        self.class_labels = np.array(class_labels)
        self.params = params or dict()
        self.model = cv2.ml.ANN_MLP_create()
        self.model.setLayerSizes(np.int32(layer_sizes))
        self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)

    def _labels_str_to_num(self, labels):
        return np.array([int(np.where(self.class_labels == l)[0][0]) for l in labels])

    def _labels_num_to_str(self, labels):
        return self.class_labels[labels.astype(int)]

    def _one_hot(self, labels_num):
        n = len(labels_num)
        one_hot = np.zeros((n, self.num_classes), dtype=np.float32)
        for i, lbl in enumerate(labels_num):
            one_hot[i, int(lbl)] = 1.0
        return one_hot

    def load(self, filepath):
        self.model = cv2.ml.ANN_MLP_load(filepath)

    def save(self, filepath):
        self.model.save(filepath)

    def fit(self, X_train, y_train, params=None):
        if params is None:
            params = self.params
        X_train = np.array(X_train, dtype=np.float32)
        if X_train.ndim == 1:
            X_train = X_train.reshape(1, -1)
        if isinstance(y_train[0], str) if len(y_train) > 0 else False:
            y_train_num = self._labels_str_to_num(y_train)
        else:
            y_train_num = np.array(y_train, dtype=int)
        y_train_onehot = self._one_hot(y_train_num)

        self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        if 'bp_dw_scale' in params:
            self.model.setBackpropWeightScale(float(params['bp_dw_scale']))
        if 'bp_moment_scale' in params:
            self.model.setBackpropMomentumScale(float(params['bp_moment_scale']))
        term_crit = params.get('term_crit',
            (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 300, 0.01))
        self.model.setTermCriteria(term_crit)
        self.model.train(X_train, cv2.ml.ROW_SAMPLE, y_train_onehot)

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=np.float32)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        ret, Y_vote = self.model.predict(X_test)
        y_hat = np.argmax(Y_vote, axis=1)
        return self._labels_num_to_str(y_hat)

    def _accuracy(self, y_true, y_pred):
        y_pred_labels = np.argmax(y_pred, axis=1)
        return np.mean(y_true.astype(int) == y_pred_labels)

    def _precision(self, y_true, y_pred):
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true = y_true.astype(int)
        precisions = []
        for c in range(self.num_classes):
            tp = np.sum((y_pred_labels == c) & (y_true == c))
            fp = np.sum((y_pred_labels == c) & (y_true != c))
            if tp + fp > 0:
                precisions.append(tp / (tp + fp))
        return np.mean(precisions) if precisions else 0.0

    def _recall(self, y_true, y_pred):
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true = y_true.astype(int)
        recalls = []
        for c in range(self.num_classes):
            tp = np.sum((y_pred_labels == c) & (y_true == c))
            fn = np.sum((y_pred_labels != c) & (y_true == c))
            if tp + fn > 0:
                recalls.append(tp / (tp + fn))
        return np.mean(recalls) if recalls else 0.0

    def evaluate(self, X_test, y_test, visualize=False):
        X_test = np.array(X_test, dtype=np.float32)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        ret, Y_vote = self.model.predict(X_test)
        if isinstance(y_test[0], str) if len(y_test) > 0 else False:
            y_test_num = self._labels_str_to_num(y_test)
        else:
            y_test_num = np.array(y_test, dtype=int)
        return (self._accuracy(y_test_num, Y_vote),
                self._precision(y_test_num, Y_vote),
                self._recall(y_test_num, Y_vote))
