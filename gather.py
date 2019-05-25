"""
Clusters: 615
Mentions: 4981
Unique Mentions: 4140
Mention Detection - 0.9541/0.9380/0.9460
BCube - 0.6976/0.7544/0.7249
Ceafe - 0.5993/0.5126/0.5526
Blanc - 0.7566/0.7825/0.7666
"""
from collections import defaultdict
import subprocess

experiments = ['lncomb', 'ln', 'ori', 'oris', 'orip', 'pluants', 'sing_max', 'sing_min', 'sing_none' ]
extras = ['ssing_min', 'ssing_max', 'ssing_none']

experiments += extras

experiments = ['lncomb', 'ln', 'pluants', 'pmany', 'ori', 'sing_max', 'sing_min', 'sing_none'][::-1]
sexperiments = ['s'+x for x in experiments]
#experiments += ['pmany']
keys = ['Mention Detection', 'BCube', 'Ceafe', 'Blanc']
res = defaultdict(list)

ex2n = {'lncomb':'New Plural + order', 'ln':'New Plural', 'pluants':'Many + order', 'pmany':'Plural + many',
        'ori':'Base + plural', 'sing_max':'Singular + most', 'sing_min':'Singular + least', 'sing_none':'Singular + none'}

def gname(ex):
    return '\\textit{' + ex2n.get(ex, ex2n.get(ex[1:],ex2n.get(ex[:-1], 'NA: '+ex))) + '}'

def formated(res):
    x = '%.1f' % (res*100)
    return x

def pbest(experiments, latest=False, sing=False):
    print(experiments, latest, sing)
    for ex in experiments:
        #if 'many' in exp and sing==True: continue
        best, ans = 0.0, []
        done = False
        with open(ex + '.out', 'r') as fin:
            for line in fin:
                for key in keys:
                    if len(line) >= len(key) and line[:len(key)] == key:
                        line = line[line.find('-') + 2:].rstrip()
                        # print(line, line.split('/'))
                        res[key] = [str(float(x[:-1]) * 100)[:4] for x in line.split('/')]
                if ('f1_max' in line or 'max_f1' in line) and '=' in line:
                    # print(line)
                    a, b = line.split(',')
                    a = a[a.find('=') + 1:]
                    b = b[b.find('=') + 1:-1]
                    # print(a,b, a==b)
                    if a == b and float(a) > best:
                        best = float(a)
                        ans = [gname(ex)]
                        for key in keys:
                            # ans.append(key)
                            ans.extend(res[key])
                            ans.append(formated(best))
                elif latest and 'Average F1 (conll):' in line:
                    ans = [gname(ex)]
                    for key in keys:
                        # ans.append(key)
                        ans.extend(res[key])
                    ans.append(formated(best))
                    done = True
        if latest and not done: continue
        print(' & '.join(ans) + '\\\\')

def eval(exp):
    print('eval', exp)
    subprocess.run("nohup python3 -u evaluate.py {} &>> {}.out &".format(exp,exp), shell=True)

def show(exp):
    subprocess.run("tail {}.out".format(exp), shell=True)

if True:
    for exp in experiments:
        _ = None
        #eval(exp)
        show(exp)
    pbest(experiments, latest=False)
    pbest(experiments, latest=True)
    pbest(sexperiments, latest=True, sing=True)
    #for exp in ['lncomb', 'ln', 'pluants', 'pmany', 'ori']:  eval(exp + '2')
    pbest(['lncombm', 'lnm'], latest=True)
    pbest([x + '2' for x in ['lncomb', 'ln', 'pluants', 'pmany', 'ori']], latest=True)

    #eval("lnm")
    #eval("lncombm")




