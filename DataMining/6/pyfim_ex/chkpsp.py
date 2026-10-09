#!/usr/bin/python3
#-----------------------------------------------------------------------
# File    : chkpsp.py
# Contents: check pattern spectrum generation
# Author  : Christian Borgelt
# History : 2024.12.13 file created
#-----------------------------------------------------------------------
from sys        import argv
from os         import remove
from random     import seed, random
from fim        import estpsp, genpsp

#-----------------------------------------------------------------------

if __name__ == '__main__':
    runs   = int(argv[1]) if len(argv) > 1 else 1
    tracts = [[i+1 for i in range(100) if random() < 0.1]
                   for k in range(1000)]
    with open('data.txt', 'w') as out:
        for t in tracts:
            for i in t: out.write('%d ' % i)
            out.write('\n')
    psp = estpsp(tracts)
    print(psp)
    psp = genpsp(tracts)
    print(psp)
    remove('data.txt')
