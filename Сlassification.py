import numpy as np
import pandas as pd
import sklearn
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class Сlassification:
    def logistic(x_, y_):
        start = time.time()

        x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.2, random_state=0)
        x = x_
        y = y_

        cls = LogisticRegression(max_iter=10000)
        cls.fit(x_train, y_train)
        y_pred = cls.predict(x_test)

        print('Accuracy = ', metrics.accuracy_score(y_test, y_pred))
        print('Precision = ', metrics.precision_score(y_test, y_pred, average='macro'))
        print('Recall = ', metrics.recall_score(y_test, y_pred, average='macro'))
        print('Коэффициенты регрессии: ', cls.coef_)

        fig, ax = plt.subplots()
        dicc = {}

        print('\n Массив предсказанных значений:\n')
        for i in range(len(x_test)):
            dicc[f'{x_test[i]}'] = y_pred[i]
        print(dicc)

        plt.ylabel('Feature №1')
        plt.xlabel('Feature №0')

        z = [(-0.7, -0.3, 0.33)]
        xfit = np.linspace(-4, 2.5)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.viridis, edgecolors="k")
        for m, b, d in z:
            yfit = m * xfit + b

        plt.plot(xfit, yfit, '-k')
        stop = time.time()

        #metrics.confusion_matrix(y_test, y_pred)
        #class_names = [0, 1]
        #fig, ax = plt.subplots()
        #ticks = np.arange(len(class_names))
        #plt.xticks(ticks, class_names)
        #plt.yticks(ticks, class_names)
        #sns.heatmap(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred)), annot=True)
        #plt.ylabel('Действительные значения')
        #plt.xlabel('Предсказанные значения')

        print('Время работы алгоритма: ', stop - start)
        plt.show()

    def logistic_rbf(x_, y_):

        """
        Функция, реализующая модель классификации на два класса на основе логистической регрессии.

        Parameters
        ----------
        x_, y_: numpy array
            Массивы обучающей переменной и предсказываемой


        x_train, x_test, y_train, y_test: numpy array
            Массивы обучающей переменной и предсказываемой, разделенные на обучающую и тестовую выборки

        y_pred: numpy array
            Массив предсказанных значений

        Returns
        -----------
        dicc: dict
            Словарь вида {x_test: y_pred}, тестовое значение переменной х сопоставляется с предсказанным значением переменной y


        """
        start = time.time()

        x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.2, random_state=0)
        x = x_
        y = y_

        cls = SVC(kernel='rbf')
        cls.fit(x_train, y_train)
        y_pred = cls.predict(x_test)

        print('Accuracy = ', metrics.accuracy_score(y_test, y_pred))
        print('Precision = ', metrics.precision_score(y_test, y_pred, average='macro'))
        print('Recall = ', metrics.recall_score(y_test, y_pred, average='macro'))

        fig, ax = plt.subplots()
        dicc = {}

        print('\n Массив предсказанных значений:\n')
        for i in range(len(x_test)):
            dicc[f'{x_test[i]}'] = y_pred[i]
        print(dicc)

        plt.ylabel('Feature №1')
        plt.xlabel('Feature №0')

        z = [(-0.7, -0.3, 0.33)]
        xfit = np.linspace(-4, 2.5)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.viridis, edgecolors="k")
        for m, b, d in z:
            yfit = m * xfit + b

        plt.plot(xfit, yfit, '-k')
        stop = time.time()

        #metrics.confusion_matrix(y_test, y_pred)
        #class_names = [0, 1]
        #fig, ax = plt.subplots()
        #ticks = np.arange(len(class_names))
        #plt.xticks(ticks, class_names)
        #plt.yticks(ticks, class_names)
        #sns.heatmap(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred)), annot=True)
        #plt.ylabel('Действительные значения')
        #plt.xlabel('Предсказанные значения')

        print('Время работы алгоритма: ', stop - start)
        plt.show()

    def lin_with_l1(x_, y_):

        """
        Функция, реализующая модель классификации на два класса основе логистической регрессии.

        Parameters
        ----------
        x_, y_: numpy array
            Массивы обучающей переменной и предсказываемой


        x_train, x_test, y_train, y_test: numpy array
            Массивы обучающей переменной и предсказываемой, разделенные на обучающую и тестовую выборки

        y_pred: numpy array
            Массив предсказанных значений

        Returns
        -----------
        dicc: dict
            Словарь вида {x_test: y_pred}, тестовое значение переменной х сопоставляется с предсказанным значением переменной y


        """
        start = time.time()

        x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.2, random_state=0)
        x = x_
        y = y_

        cls = LogisticRegression(penalty='l1', solver='liblinear')
        cls.fit(x_train, y_train)
        y_pred = cls.predict(x_test)

        print('Accuracy = ', metrics.accuracy_score(y_test, y_pred))
        print('Precision = ', metrics.precision_score(y_test, y_pred, average='macro'))
        print('Recall = ', metrics.recall_score(y_test, y_pred, average='macro'))

        fig, ax = plt.subplots()
        dicc = {}
        print('Коэффииценты регрессии: ', cls.coef_)

        print('\n Массив предсказанных значений:\n')
        for i in range(len(x_test)):
            dicc[f'{x_test[i]}'] = y_pred[i]
        print(dicc)

        plt.ylabel('Feature №1')
        plt.xlabel('Feature №0')

        z = [(-0.7, -0.3, 0.33)]
        xfit = np.linspace(-4, 2.5)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.viridis, edgecolors="k")
        for m, b, d in z:
            yfit = m * xfit + b

        plt.plot(xfit, yfit, '-k')
        stop = time.time()

        #metrics.confusion_matrix(y_test, y_pred)
        #class_names = [0, 1]
        #fig, ax = plt.subplots()
        #ticks = np.arange(len(class_names))
        #plt.xticks(ticks, class_names)
        #plt.yticks(ticks, class_names)
        #sns.heatmap(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred)), annot=True)
        #plt.ylabel('Действительные значения')
        #plt.xlabel('Предсказанные значения')

        print('Время работы алгоритма: ', stop - start)
        plt.show()

    def opornieVectora(dictionary):

        """
        Функция, реализующая модель классификации на два класса на основе метода опорных векторов.

        Parameters
        ----------
        dictionary: dictionary
            Словарь со всеми переменными, которые ранее были введены в функции vvod


        Returns
        -----------
        dicc: dict
            Словарь вида {x_test: y_pred}, тестовое значение переменной х сопоставляется с предсказанным значением переменной y


        """
        start = time.time()

        X_train = dictionary['X_train']
        X_test = dictionary['X_test']
        y_train = dictionary['y_train']
        y_test = dictionary['y_test']
        X = dictionary['X']
        y = dictionary['y']

        clf = SVC().fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print('Accuracy = ', metrics.accuracy_score(y_test, y_pred))
        print('Precision = ', metrics.precision_score(y_test, y_pred, average='macro'))
        print('Recall = ', metrics.recall_score(y_test, y_pred, average='macro'))

        fig, ax = plt.subplots()
        dicc = {}
        print('Коэффииценты регрессии: ', clf.dual_coef_)
        print('\n Массив предсказанных значений:\n')
        for i in range(len(X_test)):
            dicc[f'{X_test[i]}'] = y_pred[i]
        print(dicc)

        plt.ylabel('Feature №1')
        plt.xlabel('Feature №0')

        z = [(-0.7, -0.3, 0.33)]
        xfit = np.linspace(-4, 2.5)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis, edgecolors="k")
        for m, b, d in z:
            yfit = m * xfit + b

        plt.plot(xfit, yfit, '-k')

        stop = time.time()

        print('Время работы алгоритма: ', stop - start)
        plt.show()

    def funcForInput():

        """
        Функция для ввода данных, с помощью которых будет проводиться классификация

        Parameters
        ----------
        X: numpy.ndarray
            Массив значений переменной x
        y: numpy.ndarray
            Массив значений переменной y
        X_train: numpy.ndarray
            Массив обучающей выборки переменной x
        X_test: numpy.ndarray
            Массив тестовой выборки переменной x
        y_train: numpy.ndarray
            Массив обучающей выборки переменной y
        y_test: numpy.ndarray
            Массив тестовой выборки переменной y

        Returns
        -----------
        Final: dictionary
            Словарь со всеми переменными, которые ранее были введены, нужен для ввода в основную функцию

        """

        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        X, y = make_classification(n_samples=100, n_features=4, class_sep=0.98, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        Final = {'X_train': X_train,
                 'X_test': X_test,
                 'y_train': y_train,
                 'y_test': y_test,
                 'X': X,
                 'y': y, }
        return Final




# x1, y1 = make_classification(n_samples = 1000, n_features = 4, class_sep = 0.5, random_state = 0)
# x2, y2 = make_classification(n_samples = 10000, n_features = 4, class_sep = 0.5, random_state = 0)
# x3, y3 = make_classification(n_samples = 5000, n_features = 4, class_sep = 0.5, random_state = 0)
# x4, y4 = make_classification(n_samples = 2500, n_features = 4, class_sep = 0.5, random_state = 0)
# x5, y5 = make_classification(n_samples = 500, n_features = 4, class_sep = 0.5, random_state = 0)
# Задание 1
# print(x1, y1)
# Сlassification.logistic(x1, y1)

# # Задание 2
# Сlassification.logistic_rbf(x1, y1)
# # Задание 3
# Сlassification.lin_with_l1(x1, y1)
# # Задание 4

# x = Сlassification.funcForInput()
# print(x)
# Сlassification.opornieVectora(x)
