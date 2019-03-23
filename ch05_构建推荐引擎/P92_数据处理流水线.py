import functools

import numpy as np


def add3(input_array):
    """将数组的每一个元素加3"""
    return map(lambda x: x+3, input_array)


def mul2(input_array):
    """将数组的每一个元素乘以2"""
    return map(lambda x: x*2, input_array)


def sub5(input_array):
    """将数组的每一个元素减去5"""
    return map(lambda x: x-5, input_array)


def function_composer(*args):
    """定义一个函数组合器, 函数作为参数输入, 返回一个组合函数.
    执行顺序保留输入顺序
    """
    return functools.reduce(lambda f, g: lambda x: f(g(x)), args)


if __name__ == "__main__":

    arr = np.array([2, 5, 4, 7])

    print("\nOperation: add3(mul2(sub5(arr)))")
    arr1 = add3(arr)
    arr2 = mul2(arr1)
    arr3 = sub5(arr2)
    print("Output using the lengthy way:", list(arr3))

    func_composed = function_composer(sub5, mul2, add3)
    print("Output using funciton composition:", list(func_composed(arr)))

    print("Operation: sub5(add3(mul2(sub5(mul2(arr)))))", list(sub5(add3(mul2(sub5(mul2(arr)))))))

