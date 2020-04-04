def doc_sample1():
    q = [0, 0, 0, 1, -1]
    a = [[1, 0, 0, 1, -2],
         [0, 1, 0, -2, 1],
         [0, 0, 1, 3, 1]]
    b = [1, 2, 3]
    return q, a, b


def doc_sample2():
    q = [-1, -1, 0, 0]
    a = [[1, -1, -1, 0],
         [-1, 2, 0, -1]]
    b = [-1, -2]
    return q, a, b


def youtube_sample1():  # https://www.youtube.com/watch?v=9iYWYxSPCZs&vl=uk
    q = [-7, -3, -5, 0, 0, 0]
    a = [[1, 1, 1, 1, 0, 0],
         [1, 2, 3, 0, 1, 0],
         [3, 2, 1, 0, 0, 1]]
    b = [50, 73, 75]
    return q, a, b


def variants_sample1():
    q = [1, -3, 2, 1, 4]
    a = [[1, 2, -1, 2, 4],
         [0, -1, 2, 1, 3],
         [1, -3, 2, 2, 0]]
    b = [1, 3, 4]
    return q, a, b


def variants_sample2():
    q = [-1, -3, 2, 1, 4]
    a = [[-1, 3, 0, 2, 1],
         [2, -1, 1, 2, 3],
         [1, -1, 2, 1, 0]]
    b = [1, 2, 4]
    return q, a, b


def variants_sample3():
    q = [-1, 0, -2, 5, 4]
    a = [[-1, 3, 0, 2, 1],
         [2, -1, 1, 2, 3],
         [1, -1, 2, 1, 0]]
    b = [1, 4, 5]
    return q, a, b


def variants_sample4():
    a = [[2, 3, 1, 2, 1],
         [2, 1, -3, 2, 1],
         [2, 1, 2, 1, 0]]
    q = [-1, 1, -2, 1, 5]
    b = [1, 3, 1]
    return q, a, b


def variants_sample5():
    a = [[2, 1, 3, 4],
         [1, -1, 2, 1],
         [0, 0, 1, 3]]
    q = [-2, 3, 4, -1]
    b = [2, 4, 1]
    return q, a, b


def variants_sample6():
    a = [[2, 3, 1, 2],
         [2, -1, 2, 1],
         [1, 1, 0, -1]]
    q = [-2, 1, -1, 3]
    b = [3, 4, 1]
    return q, a, b


def variants_sample7():
    a = [[2, 3, -1, 2],
         [1, 1, 1, 1],
         [2, -1, 0, 2]]
    q = [-2, 3, 4, -1]
    b = [1, 1, 2]
    return q, a, b


def variants_sample8():
    a = [[2, 1, 3, 4],
         [2, -1, 2, 1],
         [0, 0, 1, 2]]
    q = [-2, 3, 4, -1]
    b = [1, 2, 4]
    return q, a, b


def variants_sample9():
    a = [[1, 2, 3, 1, 2, 5],
         [2, -3, 1, 2, 1, 4]]
    q = [-2, 3, 4, -1, 2, 1]
    b = [1, 2]
    return q, a, b


def variants_sample10():
    a = [[3, 2, 1, -3, 2, 1],
         [1, 1, 0, 0, 1, 1]]
    q = [-2, 3, 1, 2, 0, 1]
    b = [3, 2]
    return q, a, b


def variants_sample11():
    a = [[1, 2, 3, 4, 5, 6],
         [2, 1, -3, 2, 1, -3]]
    q = [1, -1, 2, 3, 1, 0]
    b = [1, 4]
    return q, a, b


def variants_sample12():
    a = [[2, 3, -1, 0, 2, 1],
         [2, 0, 3, 0, 1, 1]]
    q = [-2, 3, 4, -1, 2, 1]
    b = [1, 2]
    return q, a, b


def py_sample1():
    # 317.5
    c = [4, 8, 3, 0, 0, 0]
    A_eq = [
        [2, 5, 3, -1, 0, 0],
        [3, 2.5, 8, 0, -1, 0],
        [8, 10, 4, 0, 0, -1]]
    b_eq = [185, 155, 600]
    return c, A_eq, b_eq


def bad_not_antony():
    a = [[-1, -2, -2, -1, -1, - 1],
         [-1, -1, -2, -2, -1, -1],
         [-1, -1, -1, -2, -2, -1]]
    b = [-1, -1, -1]
    c = [1, 1, 1, 1, 1, 1]
    return c, a, b


def bad_frogsrop():
    a = [[-1, 3, 0, 2, 1],
         [2, -1, 1, 2, 3],
         [1, -1, 2, 1, 0],
         [-1, 0, 0, 0, 0]]
    b = [1, 2, 4, -2]
    c = [-1, -3, 2, 1, 4]
    return c, a, b
