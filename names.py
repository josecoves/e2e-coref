import sys
import pickle

test_num = sys.argv[1]
files = [x + '.friends.v4_gold_conll' for x in ['train', 'dev', 'test']]
files.append('output_{}.txt'.format(test_num))

with open("id2names.txt", "rb") as fp:
    Names = pickle.load(fp)


def process(e):
    a, b = e[0] == '(', e[-1] == ')'
    if a and b:
        return '({})'.format(Names[int(e[1:-1])])
    if a:
        return '(' + Names[int(e[1:])]
    else:
        return Names[int(e[:-1])] + ')'

for fname in files:
    with open(fname, 'r') as fin:
        with open(fname + '.named', 'w') as fout:
            for line in fin:
                row = [x.rstrip() for x in line.split('\t')]
                if len(row) > 8 and row[-1] != '-':
                    ent = [process(x) for x in row[-1].split('|')]
                    row[-1] = '|'.join(ent)
                fout.write('\t'.join(row) + '\n')


