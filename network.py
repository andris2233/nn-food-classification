import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime


# чтение всех списков ингр и ответов
def read_test(dat):
    x1 = []
    y1 = []
    for i in dat:
        s = []
        for j in i['ingredients']:
            s.append(j)
        y1.append(i['cuisine'])
        x1.append(s)
    return x1, y1


# формирование списка ответов и множеств ингр и кухонь
def read_recipes(dat, ing, ans, cui):
    for i in dat:
        for j in i['ingredients']:
            ing.add(j)
        ans[i['id']] = i['cuisine']
        cui.add(i['cuisine'])


# активационная функция нейрона
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# формирование списка входных векторов
def train_set(dat):
    x1 = []
    for i in dat:
        s = []
        for j in i['ingredients']:
            s.append(j)
        x1.append(s)
    return x1


# конвертирование списка векторов ответов в формат, выдаваемый сетью
def convert_y_to_vect(y, n):
    y_vect = np.zeros((len(y), n)) + 0.01
    for i in range(len(y)):
        y_vect[i, y[i]] = 0.99
    return y_vect


# конвертирование списка инг (вектор Х) во входной вектор сети
def convert_x_to_vect(x, ing):
    vec = []
    for i in ing:
        if i in x:
            vec.append(1.0)
        else:
            vec.append(0.01)
    return vec


class neural_network:
    # инициализация
    def __init__(self, inputnodes, outputnodes, hiddennodes_1, hiddennodes_2, learningrate):
        # формирование структуры сети
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.hnodes1 = hiddennodes_1
        self.hnodes2 = hiddennodes_2
        # скорость обучения
        self.lr = learningrate
        # инициализация весовых коэффициентов сети
        self.wih = (np.random.rand(self.hnodes1, self.inodes) - 0.5)
        self.whh = (np.random.rand(self.hnodes2, self.hnodes1) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes2) - 0.5)
        # активатор нейрона
        self.activation_function = lambda x: sigmoid(x)
        # начальное нулевое значение средней ошибки
        self.avg = 0

        pass

    # тренировка
    def train(self, inputs_list, targets_list, iterat, all_iterat):
        # преобразование списка входных значений в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # сигнал через первый скрытый
        ih_inputs = np.dot(self.wih, inputs)
        ih_outputs = self.activation_function(ih_inputs)

        # сигнал через второй скрытый
        hh_inputs = np.dot(self.whh, ih_outputs)
        hh_outputs = self.activation_function(hh_inputs)

        # сигнал к выходному слою
        final_inputs = np.dot(self.who, hh_outputs)
        final_outputs = self.activation_function(final_inputs)

        # ошибки выходного слоя
        output_errors = targets - final_outputs
        # подсчет ошибки, для последующего отображения на графике
        if iterat == 0:
            self.avg = 0
        else:
            self.avg += np.linalg.norm(output_errors)
        # алгоритм обратного распространения
        hidden_errors_2 = np.dot(self.who.T, output_errors)
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hh_outputs))

        hidden_errors_1 = np.dot(self.whh.T, hidden_errors_2)
        self.whh += self.lr * np.dot((hidden_errors_2 * hh_outputs * (1.0 - hh_outputs)), np.transpose(ih_outputs))
        self.wih += self.lr * np.dot((hidden_errors_1 * ih_outputs * (1.0 - ih_outputs)), np.transpose(inputs))
        if iterat % 1000 == 0:
            print('Iteration {} of {}'.format(iterat, all_iterat))
        pass

    # опрос
    def query(self, inputs_list):
        # преобразование списка входных значений в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        # сигнал через первый скрытый
        ih_inputs = np.dot(self.wih, inputs)
        ih_outputs = self.activation_function(ih_inputs)

        # сигнал через второй скрытый
        hh_inputs = np.dot(self.whh, ih_outputs)
        hh_outputs = self.activation_function(hh_inputs)

        # сигнал к выходному слою
        final_inputs = np.dot(self.who, hh_outputs)
        final_otputs = self.activation_function(final_inputs)
        return final_otputs


if __name__ == "__main__":
    ingredients = set()  # множество всех ингредиентов
    answ = {}  # ответы id -- кухня
    cuisines = set()  # множество всех ингредиентов
    with open("data/datapackage.json".replace('/', '\\'), "r") as r_json:
        data = json.load(r_json)
    read_recipes(data, ingredients, answ, cuisines)
    X = train_set(data)
    y = []
    cuisine_vect = []
    for i in cuisines:
        cuisine_vect.append(i)
    for i in answ:
        for j in range(len(cuisine_vect)):
            if answ[i] == cuisine_vect[j]:
                y.append(j)
    ingredients_vect = []
    for i in ingredients:
        ingredients_vect.append(i)
    del ingredients
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    y_v_train = convert_y_to_vect(y_train, len(cuisines))
    y_v_test = convert_y_to_vect(y_test, len(cuisines))

    # инициализация экземпляра нейросети
    inputnodes = len(ingredients_vect)
    outputnodes = len(cuisines)
    hiddennodes_1 = 100
    hiddennodes_2 = 40
    learningrate = 0.1
    n = neural_network(inputnodes, outputnodes, hiddennodes_1, hiddennodes_2, learningrate)
    avg_cost_func = []

    # обучение НС
    now = datetime.datetime.now()
    print()
    print('start learning at: {}'.format(now.time()))
    epoch = 1
    for j in range(epoch):
        print('Epoch {} of {}'.format(j + 1, epoch))
        for i in range(len(X_train)):
            n.train(convert_x_to_vect(X_train[i], ingredients_vect), y_v_train[i], i, len(X_train))
        avg_cost_func.append(1.0/len(y_v_train) * n.avg)
    now = datetime.datetime.now()
    print('the end of learning at: {} \n'.format(now.time()))

    # запись результатов обучения в файлы
    np.savetxt('input_hidden_weights.txt', np.c_[n.wih], delimiter='\t')
    np.savetxt('hidden_hidden_weights.txt', np.c_[n.whh], delimiter='\t')
    np.savetxt('hidden_output_weights.txt', np.c_[n.who], delimiter='\t')
    np.savetxt('hidden_output_weights.txt', np.c_[n.who], delimiter='\t')
    with open("structure_input.txt", "w") as file:
        for i in ingredients_vect:
            file.write(str(i) + '\n')
    with open("structure_neural_network.txt", "w") as file:
        file.write(str(n.inodes) + '\n')
        file.write(str(n.hnodes1) + '\n')
        file.write(str(n.hnodes2) + '\n')
        file.write(str(n.onodes) + '\n')
        file.write(str(n.lr) + '\n')

    # построение графика
    plt.plot(avg_cost_func)
    plt.ylabel('Average J')
    plt.xlabel('Epoch number')
    plt.show()
    predict = []
    score = 0

    # пробуем на своих рецептах
    b = True
    while b:
        n1 = input('Do you want to check your recipes? y or n \n')
        if n1 == 'y':
            with open("data/test_two.json".replace('/', '\\'), "r") as r_json:
                data = json.load(r_json)
            x_test_two, y_test_two = read_test(data)
            y_test_two_to = []
            for i in range(len(y_test_two)):
                for j in range(len(cuisine_vect)):
                    if y_test_two[i] == cuisine_vect[j]:
                        y_test_two_to.append(j)
            y_test_two_to = convert_y_to_vect(y_test_two_to, len(cuisines))
            print('Answers:')
            for i in range(len(x_test_two)):
                predict = n.query(convert_x_to_vect(x_test_two[i], ingredients_vect))
                print(str(i + 1) + ') ' + cuisine_vect[np.argmax(predict)])
            print('End of task \n')
            b = False
        elif n1 == 'n':
            print('End of task')
            b = False
        else:
            print('Incorrect input. Try again')

    # подсчет точности и сохранение ответов сети для проверки
    now = datetime.datetime.now()
    print('Start testing set at: {}'.format(now.time()))
    with open("predict.txt".replace("/", "\\"), "w") as file:
        for i in range(len(X_test)):
            if i%1000 == 0:
                print('Iteration {} of {}'.format(i, len(X_test)))
            predict = n.query(convert_x_to_vect(X_test[i], ingredients_vect))
            file.write(str(predict))
            if np.argmax(predict) == np.argmax(y_v_test[i]):
                score += 1

    # вывод точности классификации
    print('\nPrediction accuracy is {}%'.format(score/len(X_test) * 100))
    now = datetime.datetime.now()
    print('End of testing at: {}'.format(now.time()))
