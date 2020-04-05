from lab2.utils import *
import numpy as np


def main():
    for i in range(-10, 10):
        c1 = i / 10
        for j in range(-10, 10):
            c2 = j / 10
            for k in range(-10, 10):
                a11 = k / 10
                for t in range(-10, 10):
                    a12 = t / 10
                    for p in range(-10, 10):
                        a21 = p / 10
                        for q in range(-10, 10):
                            a22 = q / 10
                            for w in range(-10, 10):
                                b1 = w / 10
                                for e in range(-10, 10):
                                    b2 = e / 10
                                    c_ = [c1, c2]
                                    b_ = [b1, b2]
                                    a_ = [[a11, a12], [a21, a22]]
                                    res1 = simplex_method_ub(c_, a_, b_)
                                    res2 = bnb_method_my(c_, a_, b_)
                                    dx = np.any(np.abs(np.array(res1.x) - res2.x) > 2)
                                    if res1.success and dx:
                                        print(c_)
                                        print(a_)
                                        print(b_)


if __name__ == '__main__':
    main()
