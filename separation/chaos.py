import numpy as np
import os
from scipy.integrate import solve_ivp
from sklearn import preprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from typing import cast
import threading

class Progressbar:
    def __init__(self):
        self.lock = threading.Lock()
        self.progressbar_count = 0
        self.progressbar_bar=0

    def count(self):
        with self.lock:
            self.progressbar_count += 1
        print(self.progressbar_count)
        if (self.progressbar_count // 100) != self.progressbar_bar:
            self.progressbar_bar += 1
            print(self.progressbar_bar)

class ChaosGenerator:

    def __init__(self, start_time, end_time, step, function = None, args = (), method='RK45'):
        self.__start_time = start_time
        self.__end_time = end_time
        self.__step = step
        self.__t_span = [start_time, end_time]
        self.__t_eval = np.linspace(*self.__t_span, step)
        if function == None:
            self.__function = self.lorenz
        else:
            self.__function = function
        self.__args = tuple(args)
        self.__method = method

    def lorenz(self,t,X,p,r,b):
        x,y,z = X
        return [-p*x+p*y, -x*z+r*x-y, x*y-b*z]

    def task_solve_ivp(self, init):
        ivp_ans: np.ndarray = cast(np.ndarray, solve_ivp(self.__function, self.__t_span, init, method=self.__method,
                                                     t_eval=self.__t_eval).y)
        normalize_ivp_ans = preprocessing.minmax_scale(ivp_ans, axis=1)
        return ivp_ans, normalize_ivp_ans

    def calculate(self, init_list):
        """
        calculate chaotic signals using multiprocessing from initial values
        # Returns
        result: ndarray
            chaotic signals
        normalize_result: ndarray
            normalized chaotic signals
        """
        chunk_size = 100
        self.result = []
        self.normalize_result = []

        with ProcessPoolExecutor(max_workers=13) as executor:
            results = executor.map(self.task_solve_ivp, init_list, chunksize=chunk_size)

        results_list = list(results)
        self.result = np.array(
            list(map(lambda x: x[0], results_list))).transpose(0, 2, 1)
        self.normalize_result = np.array(
            list(map(lambda x: x[1], results_list))).transpose(0, 2, 1)

        return self.result, self.normalize_result

    @staticmethod
    def Normalize(value_list: np.array):
        """
        normalize the value_list using min-max scaling
        """
        result = []
        wave_count = value_list.shape[0]
        for i in range(wave_count):
            normalized_value = preprocessing.minmax_scale(value_list[i])
            result.append(normalized_value)
        return np.array(result)

    @staticmethod
    def Standardize(value_list: np.array):
        """
        standardize the value_list using z-score normalization
        """
        result = []
        wave_count = value_list.shape[0]
        for i in range(wave_count):
            normalized_value = preprocessing.scale(value_list[i])
            result.append(normalized_value)
        return np.array(result)

    def infomation(self):
        """
        情報を表示
        """
        return f'''
        start_time: {self.start_time}
        end_time = {self.end_time}
        step = {self.step}
        function = {self.function.__name__}
        args = {self.args}
        '''



###########
class Rossler:
  """Rossler equation class"""

  def __init__(self, a, b, c):
    """
    rossler equation
    """
    self.a = a
    self.b = b
    self.c = c

  def __call__(self, t, x):

    return [-x[1]-x[2], x[0] + self.a * x[1], self.b + x[0] * x[2] - self.c * x[2]]


class Lorenz:
  """Lorenz equation class"""

  def __init__(self, sigma, beta, rho):
    """
    lorenz equation
    """
    self.sigma = sigma
    self.beta = beta
    self.rho = rho

  def __call__(self, t, x):
    return [self.sigma * (x[1] - x[0]), x[0] * (self.rho - x[2]) - x[1], x[0] * x[1] - self.beta * x[2]]   
