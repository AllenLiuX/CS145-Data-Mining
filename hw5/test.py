import time





if __name__ == '__main__':
    start_time = time.time()
    a = frozenset(['a', 'b'])
    print(a)
    for i in a:
        i = frozenset(i)
        print(i)
        print(type(i))
        print(i.issubset(a))
    print(subsets(a))
    end_time = time.time()
    print('======= Time taken: %f =======' %(end_time - start_time))