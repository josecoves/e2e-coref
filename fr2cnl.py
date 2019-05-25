import json
import os
import pickle
from collections import defaultdict, Counter

in_dir = '/home/jcoves/data/enhanced-jsons/'
#out_dir = '/home/jcoves/data/fr2conl/'
out_dir = '/home/jcoves/e2e-coref/'

#file_name = ['friends_season_0','.json']
fnames = ['friends_season_0{}.json'.format(i+1) for i in range(4)]
#fnames = [file_name[0] + str(i+1) + file_name[1] for i in range(4)]
onames = [x + '.friends.v4_gold_conll' for x in ['train', 'dev', 'test']]

Names = []
Name2id = {}
Length = Counter()
Nspeakers = Counter()
Singletons = 0
Popular = {}
Groups = Counter()

def getNameId(name):
    if name not in Name2id:
      Name2id[name] = len(Names)
      Names.append(name)
    return str(Name2id[name])

def flatten(l):
    return [x for sublist in l for x in sublist]

def get(x, mentions):
    #m = {'begin' : defaultdict(list), 'end' : defaultdict(list), 'unique' : defaultdict(list)}
    m = defaultdict(list)
    for c in x:
        Groups[len(c)-2] += 1
        for name in c[2:]:
            mentions[name] += 1
            b, e = c[:2]
            if b + 1 == e:
                m[b].append('(' + getNameId(name) + ')')
            else:
                m[b].append('(' + getNameId(name))
                m[e - 1].append(getNameId(name) + ')')
            Length[e-b] += 1
    return m #m['begin'] + m['unique'] + m['end']

def getSing(x, scene_id, order):
    m = defaultdict(list)
    for c in x:
        pop = sorted([(Popular[scene_id][name], name) for name in c[2:]]) #sort by popularity
        b, e = c[:2]
        if order == 'top':
            for name in pop[-6:]:
                if b + 1 == e:
                    m[b].append('(' + getNameId(name) + ')')
                else:
                    m[b].append('(' + getNameId(name))
                    m[e - 1].append(getNameId(name) + ')')
            continue
        if order == 'none' and len(pop) > 1: continue  # ignore plural clusters
        name = pop[0 if order == 'min' else -1][1]
        if b + 1 == e:
            m[b].append('(' + getNameId(name) + ')')
        else:
            m[b].append('(' + getNameId(name))
            m[e - 1].append(getNameId(name) + ')')
    if x and False:
        print('x', x)
        print('sid', scene_id)
        print(order)
        print('m', m)
        #exit(1)
    return m

fouts = [open(out_dir + oname,'w+') for oname in onames]
spks = set()
#for fout in fouts
for fname in fnames:
    with open(in_dir + fname, 'r+') as fin:
      data = json.load(fin)
      for episode in data['episodes']:
          ep = int(episode['episode_id'][-2:])
          fout = fouts[0 if ep < 20 else 1 if ep < 22 else 2]
          for scene in episode['scenes']:
              temp = open('temp.txt', 'w+')
              good = False
              temp.write('#begin document ({}); part {}'.format(scene['scene_id'], '000') + '\n')
              mentions = Counter()
              for ut in scene['utterances']:
                  #fout.write('\n')
                  if not ut['transcript']: continue
                  speaker = ut['speakers']
                  Nspeakers[len(speaker)] += 1
                  if len(speaker) > 1:
                      speaker = '|'.join(speaker)
                      #print(speaker)
                      #exit(1)
                  elif len(speaker) == 0: speaker = 'Narrator'
                  else: speaker = speaker[0]
                  spks.update(speaker.split('|')) # spks
                  doc_key = ut['utterance_id']
                  doc = '0'
                  tokens = ut['tokens']
                  pos = ut['part_of_speech_tags']
                  ner = ut['named_entity_tags']
                  clusters = ut['character_entities']
                  clu = [get(x, mentions) for x in clusters]
                  j = 0
                  for s in range(len(tokens)):  # sentence
                      for i in range(len(tokens[s])):
                          p = '*'  # pos
                          const = '(' + pos[s][i] + '*)'
                          n = '*'  # ner
                          c = '-' if speaker == 'Narrator' else '|'.join(clu[s][i])
                          if not c: c = '-'
                          line = [doc_key, doc, str(j), tokens[s][i], p, const, '-', '-', '-', speaker, n, c]
                          if not line: continue
                          #  print(line)
                          #  print('\t'.join(line))
                          #  exit()
                          temp.write('\t'.join(line) + '\n')
                          j += 1
                  temp.write('\n')
                  good = True
              temp.write('#end document' + '\n')
              if good:
                temp.seek(0)
                fout.write(temp.read())
                Popular[scene['scene_id']] = mentions
                if any([x == 1 for x in mentions.values()]):
                    #print('Singleton: ' + scene['scene_id'], mentions)
                    Singletons += 1
              temp.close()
              os.remove('temp.txt')
for f in fouts: f.close()
with open("speaker_file.txt", "wb") as spk_file:
    pickle.dump(list(spks), spk_file)
with open("id2names.txt", "wb") as fp:
    pickle.dump(Names, fp)
print('Length of mentions =', Length)
print('Number of speakers =', Nspeakers)
print('Names', len(Names))
print('Singletons', Singletons)
print('Groups', Groups)

print('Singular Training')
for order in ['top', 'min', 'max', 'none']:
    with open(out_dir + 'sing_{}_'.format(order) + onames[0], 'w+') as fout:
        for fname in fnames:
            with open(in_dir + fname, 'r+') as fin:
                data = json.load(fin)
                for episode in data['episodes']:
                    ep = int(episode['episode_id'][-2:])
                    if ep>=20: continue
                    for scene in episode['scenes']:
                        temp = open('temp.txt', 'w+')
                        good = False
                        temp.write('#begin document ({}); part {}'.format(scene['scene_id'], '000') + '\n')
                        for ut in scene['utterances']:
                            if not ut['transcript']: continue
                            speaker = ut['speakers']
                            if len(speaker) > 1:
                                speaker = '|'.join(speaker)
                            elif len(speaker) == 0:
                                speaker = 'Narrator'
                            else:
                                speaker = speaker[0]
                            doc_key = ut['utterance_id']
                            doc = '0'
                            tokens = ut['tokens']
                            pos = ut['part_of_speech_tags']
                            ner = ut['named_entity_tags']
                            clusters = ut['character_entities']
                            clu = [getSing(x, scene['scene_id'], order) for x in clusters]
                            j = 0
                            for s in range(len(tokens)):  # sentence
                                for i in range(len(tokens[s])):
                                    p = '*'  # pos
                                    const = '(' + pos[s][i] + '*)'
                                    n = '*'  # ner
                                    c = '-' if speaker == 'Narrator' else '|'.join(clu[s][i])
                                    if not c: c = '-'
                                    line = [doc_key, doc, str(j), tokens[s][i], p, const, '-', '-', '-', speaker, n, c]
                                    temp.write('\t'.join(line) + '\n')
                                    j += 1
                            temp.write('\n')
                            good = True
                        temp.write('#end document' + '\n')
                        if good:
                            temp.seek(0)
                            fout.write(temp.read())
                        temp.close()
                        os.remove('temp.txt')






