from entropy import *
from generate_random_quantum import *
import time

# TEST
def test_non_shannon(gen_func, test_func, lim):
    start_time = time.clock()
    for i in range(lim):
        p4 = gen_func(2**4) # 4-qubit state
        q4 = gen_func(3**4) # 4-qutrit state

        res, _ = test_func(p4, 2)
        res, _ = test_func(q4, 3)
        assert res == True

    print("%d of each of 4 qubit and qutrit states tested" % (lim))
    print("%s: --- PASSED in %s seconds ---" % (test_func, time.clock() - start_time))
