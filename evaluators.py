import itertools
from abc import *
import numpy as np
from pydash import flatten_deep
from sklearn.utils.linear_assignment_ import linear_assignment
import collections
import sys

Gold_Path = 'dev.friends.v4_gold_conll'
Test_Path = 'test.friends.v4_gold_conll'

def path2docs(path):
    #mentions = set()
    docs = []
    with open(path, 'r') as fin:
        for line in fin:
            if len(line) < 2: continue
            if line[1] == 'b':
                clusters = collections.defaultdict(list)
                opened = 0
            elif line[1] == 'e':
                docs.append(list(list(set(cluster)) for cluster in clusters.values()))
                assert(opened == 0)
                #print('clusters',clusters)
                #print('docs',docs)
                #exit(1)
            elif line[0] == '#': continue # scores
            else:
                row = line.split('\t')
                key = row[0] + ':' + row[2]
                cs = row[-1].rstrip()
                if cs != '-':
                    for c in row[-1].split('|'):
                        c = c.rstrip()
                        a, b = c[0] == '(', c[-1] == ')'
                        if a and b:
                            clusters[c[1:-1]].append(key)
                        elif a:
                            opened += 1
                            #clusters[c[1:]].append(key)
                        else:
                            opened -=1
                            #print(row)
                            #print('clusters[c[:-1]]',clusters[c[:-1]])
                            #clusters[c[:-1]][-1] += ',' + row[2]
                            clusters[c[:-1]].append(key)

    print('Converted {} to Cluster Doc'.format(path))
    print('Docs: ', len(docs))
    print('Clusters:', sum([len(doc) for doc in docs]))
    print('Mentions:', sum([len(cluster) for doc in docs for cluster in doc]))
    mentions = set(mention for doc in docs for cluster in doc for mention in cluster)
    print('Unique Mentions:', len(mentions))
    return docs


def evalDoc(path, gold_path=Gold_Path):
    if len(sys.argv) > 2: gold_path = Test_Path
    print('Evaluating {} using {}'.format(path, gold_path))
    golds = path2docs(gold_path)
    autos = path2docs(path)

    fout = open(path, 'a')

    p, r, f = MentionEvaluator().evaluate_documents(golds, autos)
    print('Mention Detection - %.4f/%.4f/%.4f' % (p, r, f))
    fout.write('# p,r,f = %.4f/%.4f/%.4f\n' % (p, r, f))

    p, r, f = BCubeEvaluator().evaluate_documents(golds, autos)
    print('Bcube - %.4f/%.4f/%.4f' % (p, r, f))
    fout.write('# p,r,f = %.4f/%.4f/%.4f\n' % (p, r, f))

    p, r, f = CeafeEvaluator().evaluate_documents(golds, autos)
    print('Ceafe - %.4f/%.4f/%.4f' % (p, r, f))
    fout.write('# p,r,f = %.4f/%.4f/%.4f\n' % (p, r, f))

    p, r, f = BlancEvaluator().evaluate_documents(golds, autos)
    print('Blanc - %.4f/%.4f/%.4f' % (p, r, f))
    fout.write('# p,r,f = %.4f/%.4f/%.4f\n' % (p, r, f))

    fout.close()


#path2docs('output_61.txt')


class AbstractEvaluator(object):
    @abstractmethod
    def evaluate_documents(self, gold_documents, auto_documents):
        return

    @abstractmethod
    def evaluate_clusters(self, gold_clusters, auto_clusters):
        return

    @staticmethod
    def f1_score(precision, recall):
        return 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    @staticmethod
    def create_mention2cluster_map(clusters):
        m2cs = dict()
        for c in clusters:
            for m in c:
                if m in m2cs:
                    m2cs[m].append(c)
                else:
                    m2cs[m] = [c]
        return m2cs
        # return dict((m, c) for c in clusters for m in c)

class MentionEvaluator(AbstractEvaluator):
    def __init__(self):
        self.name = 'Mention'

    def evaluate_documents(self, gold_documents, auto_documents):
        return self.evaluate_clusters(set(mention for doc in gold_documents for cluster in doc for mention in cluster),
                                      set(mention for doc in auto_documents for cluster in doc for mention in cluster))

    def evaluate_clusters(self, gold_mentions, auto_mentions):
        correct = gold_mentions.intersection(auto_mentions)
        p = len(correct) / len(auto_mentions)
        r = len(correct) / len(gold_mentions)
        return p, r, self.f1_score(p, r)

class BCubeEvaluator(AbstractEvaluator):
    def __init__(self):
        self.name = 'BCube'
    def evaluate_documents(self, gold_documents, auto_documents):
        return self.evaluate_clusters(sum(gold_documents, []), sum(auto_documents, []))

    def evaluate_clusters(self, gold_clusters, auto_clusters):
        gold_m2c_map = self.create_mention2cluster_map(gold_clusters)
        auto_m2c_map = self.create_mention2cluster_map(auto_clusters)
        mentions = auto_m2c_map.keys()

        pc = rc = 0
        for mention in mentions:
            gcs = gold_m2c_map.get(mention)
            acs = auto_m2c_map.get(mention)
            #print('mention', mention)
            #print('gcs', gcs)
            #print('acs', acs)
            agg_gold_cluster = set(flatten_deep(gcs)) if gcs else set()
            agg_auto_cluster = set(flatten_deep(acs))

            correct = len(agg_gold_cluster.intersection(agg_auto_cluster))
            pc += float(correct) / len(agg_auto_cluster) if agg_auto_cluster else 0.0
            rc += float(correct) / len(agg_gold_cluster) if agg_gold_cluster else 0.0

        p = pc / len(mentions)
        r = rc / len(mentions)

        return p, r, self.f1_score(p, r)


class BlancEvaluator(AbstractEvaluator):
    def __init__(self):
        self.name = 'Blanc'
    def evaluate_documents(self, gold_documents, auto_documents):
        # coreferent / non-coreferent indices
        c, n = 0, 1
        confusion = np.zeros((2, 2), dtype="int32")

        for gdoc, adoc in zip(gold_documents, auto_documents):
            confusion += self.evaluate_clusters(gdoc, adoc)

        #print(confusion)

        pc = float(confusion[c, c]) / (confusion[c, c] + confusion[n, c]) \
            if confusion[c, c] + confusion[n, c] > 0 \
            else 0.0
        pn = float(confusion[n, n]) / (confusion[c, n] + confusion[n, n]) \
            if confusion[c, n] + confusion[n, n] > 0 \
            else 0.0
        p = float(pc + pn) / 2

        rc = float(confusion[c, c]) / (confusion[c, c] + confusion[c, n]) \
            if confusion[c, c] + confusion[c, n] > 0 \
            else 0.0
        rn = float(confusion[n, n]) / (confusion[n, c] + confusion[n, n]) \
            if confusion[n, c] + confusion[n, n] > 0 \
            else 0.0
        r = float(rc + rn) / 2

        fc = AbstractEvaluator.f1_score(pc, rc)
        fn = AbstractEvaluator.f1_score(pn, rn)
        f = float(fc + fn) / 2

        return p, r, f

    def total_num_links(self, gold_clusters, auto_clusters):
        gold_ms = {m for gc in gold_clusters for m in gc}
        auto_ms = {m for ac in auto_clusters for m in ac}
        num_ms = len(gold_ms.union(auto_ms))
        # print(num_ms)
        num_links = (num_ms * (num_ms - 1)) / 2

        return num_links

    def evaluate_clusters(self, gold_clusters, auto_clusters):
        def get_links(cluster):
            if len(cluster) > 1:
                links = {(m1, m2) if m1 < m2 else (m2, m1) for i, m1 in enumerate(cluster) for m2 in cluster[i + 1:]}
                # print(links)
                return links
                # return set(itertools.combinations(cluster, 2))
            else:
                return set()

        #print('*map(get_links, gold_clusters)', *map(get_links, gold_clusters))
        gold_links = set.union(*map(get_links, gold_clusters)) if gold_clusters else set()
        #print('gold_links',gold_links)

        #print('auto_clusters',auto_clusters)
        #print('*map(get_links, auto_clusters)',*map(get_links, auto_clusters))
        auto_links = set.union(*map(get_links, auto_clusters)) if auto_clusters else set()

        # print(len(gold_links))
        # print(len(auto_links))

        num_links = self.total_num_links(gold_clusters, auto_clusters)

        # coreferent / non-coreferent indices
        c, n = 0, 1
        confusion = np.zeros((2, 2), dtype="int32")

        confusion[c, c] = len(auto_links & gold_links)  # intersection of links
        confusion[n, c] = len(auto_links.difference(gold_links))    # (auto union gold) \ gold
        confusion[c, n] = len(gold_links.difference(auto_links))    # (auto union gold) \ auto
        confusion[n, n] = num_links - (confusion[c, c] + confusion[n, c] + confusion[c, n])

        # print(confusion)

        return confusion


class CeafeEvaluator(AbstractEvaluator):
    def __init__(self):
        self.name = 'Ceafe'

    def evaluate_documents(self, gold_documents, auto_documents):
        return self.evaluate_clusters(sum(gold_documents, []), sum(auto_documents, []))

    def evaluate_clusters(self, gold_clusters, auto_clusters):
        def phi4(c1, c2):
            return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))

        # enable ceaf to deal with singletons
        # auto_clusters = [c for c in auto_clusters if len(c) != 1]
        scores = np.zeros((len(gold_clusters), len(auto_clusters)))
        for i in range(len(gold_clusters)):
            for j in range(len(auto_clusters)):
                scores[i, j] = phi4(gold_clusters[i], auto_clusters[j])
        matching = linear_assignment(-scores)
        similarity = float(sum(scores[matching[:, 0], matching[:, 1]]))

        p = similarity / len(auto_clusters) if similarity else 0.0
        r = similarity / len(gold_clusters) if similarity else 0.0

        return p, r, self.f1_score(p, r)


class LinkingMicroF1Evaluator(object):
    def __init__(self, labels):
        self.labels = labels

    def evaluate_states(self, states):
        gold_links = {l: [] for l in self.labels}
        auto_links = {l: [] for l in self.labels}

        for m in sum(states, []):
            for ref in m.gold_refs:
                gold_links[ref].append(m)

            for ref in m.auto_refs:
                auto_links[ref].append(m)

        scores = {}
        for l in self.labels:
            g, a = gold_links[l], auto_links[l]
            c = float(len(set(g).intersection(set(a))))

            p = c / len(a) if a else 0.0
            r = c / len(g) if g else 0.0
            f = AbstractEvaluator.f1_score(p, r)

            scores[l] = (p, r, f)

        return scores


class LinkingMacroF1Evaluator(object):
    def __init__(self):
        pass

    def evaluate_states(self, states):
        c, g_count, a_count = 0.0, 0, 0
        m_all = sum(states, [])

        for m in m_all:
            g, a = m.gold_refs, m.auto_refs
            c += float(len(set(g).intersection(set(a))))
            g_count += len(g)
            a_count += len(a)

        p = c / a_count if a_count > 0 else 0.0
        r = c / g_count if g_count > 0 else 0.0

        f = AbstractEvaluator.f1_score(p, r)

        return p, r, f


if __name__ == "__main__":
    evalDoc('output_{}.txt'.format(sys.argv[1]))