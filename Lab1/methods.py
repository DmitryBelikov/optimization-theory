from Lab1.utils import f


def fibonacci(a, b, eps):
    fib = [0.0, 1.0]
    while fib[-1] <= (b - a) / eps:
        fib.append(fib[-2] + fib[-1])

    n = len(fib) - 3

    x1 = a + (fib[n] / fib[-1]) * (b - a)
    x2 = a + (fib[n + 1] / fib[-1]) * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    while n > 0:
        n -= 1
        if f(x1) < f(x2):
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (fib[n] / fib[-1]) * (b - a)
            f1 = f(x1)
        elif f(x1) > f(x2):
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (fib[n + 1] / fib[-1]) * (b - a)
            f2 = f(x2)
        else:
            a = x1
            b = x2
    return a, b
