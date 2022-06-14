from lab6.Method6 import *
import copy



class User:
    def userAnswer(self):
        """
        Функция создана дл упрощения работы пользователя с данной программой, тут представлены подсказки и премеры ввода
        данных.
        Returns
        ===========
        Обращается к нужной функции метода и передает ей необходимые параметры.
        """
        print(
            "Каким методом хотите воспользоваться?\n"
            "1 - Функция, реализующая модель классификации на два класса на основе логистической регрессии \n"
            "2 - Функция, реализующая модель классификации на два класса на основе логистической регрессии с "
            "радиальными базисными функциями\n "
            "3 - Функция, реализующая модель классификации на два класса на основе логистической регрессии с "
            "регуляризацией L1.\n "
            "4 - Функция, реализующая модель классификации на два класса на основе метода опорных векторов.\n")

        user_answer = int(input())

        # Функция, реализующая модель классификации на два класса на основе логистической регрессии
        if user_answer == 1:
            print("Мы будем составлять массив предсказываемых значений и массив обучающей выборки с помощью функции"
                  "make_classification. Для этого нужно ввести ее параметры. Пример параметров: "
                  "make_classification(n_samples = 1000, n_features = 4, class_sep = 0.5, random_state = 0) ")
            print("Введите параметр n_samples. Например: 1000")
            n_samp = int(input())
            print("Введите параметр n_features. Например: 4")
            n_feat = int(input())
            print("Введите параметр class_sep. Например: 0.5")
            class_s = float(input())
            print("Введите параметр random_state. Например: 0")
            random_s = int(input())

            x1, y1 = make_classification(n_samples=n_samp, n_features=n_feat, class_sep=class_s, random_state=random_s)

            Method6.logistic(x1, y1)

        # Функция, реализующая модель классификации на два класса на основе логистической регрессии с
        # радиальными базисными функциями

        elif user_answer == 2:
            print("Мы будем составлять массив предсказываемых значений и массив обучающей выборки с помощью функции"
                  "make_classification. Для этого нужно ввести ее параметры. Пример параметров: "
                  "make_classification(n_samples = 1000, n_features = 4, class_sep = 0.5, random_state = 0) ")
            print("Введите параметр n_samples. Например: 1000")
            n_samp = int(input())
            print("Введите параметр n_features. Например: 4")
            n_feat = int(input())
            print("Введите параметр class_sep. Например: 0.5")
            class_s = float(input())
            print("Введите параметр random_state. Например: 0")
            random_s = int(input())

            x1, y1 = make_classification(n_samples=n_samp, n_features=n_feat, class_sep=class_s, random_state=random_s)

            Method6.logistic_rbf(x1, y1)

        # Функция, реализующая модель классификации на два класса на основе логистической регрессии с "
        # регуляризацией L1

        elif user_answer == 3:
            print("Мы будем составлять массив предсказываемых значений и массив обучающей выборки с помощью функции"
                  "make_classification. Для этого нужно ввести ее параметры. Пример параметров: "
                  "make_classification(n_samples = 1000, n_features = 4, class_sep = 0.5, random_state = 0) ")
            print("Введите параметр n_samples. Например: 1000")
            n_samp = int(input())
            print("Введите параметр n_features. Например: 4")
            n_feat = int(input())
            print("Введите параметр class_sep. Например: 0.5")
            class_s = float(input())
            print("Введите параметр random_state. Например: 0")
            random_s = int(input())

            x1, y1 = make_classification(n_samples=n_samp, n_features=n_feat, class_sep=class_s, random_state=random_s)

            Method6.logistic_rbf(x1, y1)

        # Функция, реализующая модель классификации на два класса на основе метода опорных векторов

        elif user_answer == 4:
            print("Эта функция запускается на автоматически генерируемых данных")

            x = Method6.funcForInput()
            print("Сгенерироавнные данные: ")
            print(x)
            Method6.opornieVectora(x)

        else:
            print('Введен неверный номер')


functionss = User()
functionss.userAnswer()
