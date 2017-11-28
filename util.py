from collections import defaultdict, namedtuple
import numpy as np
import math, ast, os, codecs, random
import cPickle as pickle
import json, sys, io, subprocess

SHIFT = 0
SKIP = 1

flatten = lambda l:[item for sublist in l for item in sublist]

recursive_flatten = lambda l:flatten([recursive_flatten(item) if isinstance(item, list) else [item] for item in l])

def masked_compress(sent, mask):
    return [sent[i] for i in range(len(mask)) if mask[i]]

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def normalize(x):
    denom = sum(x)
    return [i/denom for i in x]

def softmax(x, axis=0):
    e_x = np.exp(x - np.max(x, axis=axis))
    out = e_x / e_x.sum(axis=axis)
    return out

def weightedChoice(weights, objects, apply_softmax=False, alpha=None):
    """Return a random item from objects, with the weighting defined by weights
    (which must sum to 1)."""
    if apply_softmax: weights = softmax(weights)
    if alpha: weights = normalize([w**alpha for w in weights])
    cs = np.cumsum(weights) #An array of the weights, cumulatively summed.
    idx = sum(cs < np.random.rand()) #Find the index of the first weight over a random value.
    idx = min(idx, len(objects)-1)
    return objects[idx]

def multi_weighted_choice(p):
    c = p.cumsum(axis=0)
    u = np.random.rand(1) if len(p.shape) == 1 else np.random.rand(1, len(c[0]))
    choices = (u < c).argmax(axis=0)
    if len(choices.shape) == 0: choices = [choices]
    return choices

def itersubclasses(cls, _seen=None):
    if not isinstance(cls, type):
        raise TypeError('itersubclasses must be called with '
                        'new-style classes, not %.100r' % cls)
    if _seen is None: _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError: # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in itersubclasses(sub, _seen):
                yield sub

def get_batches(data, max_size, solo_len=100000):
    data.sort(key=lambda x:-len(x))
    ans = []
    current_start = 0
    current_count = 0
    for i in range(len(data)):
        if current_count >= max_size or len(data[current_start]) != len(data[i])\
                or (len(data[current_start]) > solo_len and current_start != i):
            ans.append(data[current_start: i])
            current_count = 0
            current_start = i
        current_count += 1
    ans.append(data[current_start:])
    return ans

def shuffle_preserve_first(l,n=1):
    first = l[:n]
    rest = l[n:]
    random.shuffle(rest)
    return first + rest

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = defaultdict(lambda :None, self)
    def __getattr__(self, key):
        return self.__dict__[key]
    def __str__(self):
        return str(self.__dict__)

class key_dependent_dict(defaultdict):
    def __init__(self,f_of_x):
        super(key_dependent_dict, self).__init__(None) # base class doesn't get a factory
        self.f_of_x = f_of_x # save f(x)
    def __missing__(self, key): # called when a default needed
        ret = self.f_of_x(key) # calculate default value
        self[key] = ret # and install it in the dict
        return ret

def load_data(data_loc, data_view, start_tok=u'^', end_tok=u'|'):
    if os.path.isdir(data_loc): data_locs = [data_loc+"/"+dl for dl in sorted(os.listdir(data_loc))]
    else: data_locs = [data_loc]
    ans = []
    for data_loc in data_locs:
        print data_loc
        with open(data_loc) as f:
            corpus = f.read().decode('utf-8-sig')
            lines = corpus.split("\n")
            if data_view == "char": ans += [[start_tok] + list(line.strip()) + [end_tok] for line in lines if line]
            else: ans += [[start_tok] + line.strip().split(" ") + [end_tok] for line in lines if line]
    return ans

def bleu_score(references, predicted):
    with open("/tmp/reference", "w") as f: f.write("\n".join(references).encode('utf-8'))
    with open("/tmp/predicted", "w") as f: f.write("\n".join(predicted).encode('utf-8'))
    out = subprocess.check_output('perl multi-bleu.perl /tmp/reference < /tmp/predicted', shell=True)
    out = out.split(",")[0][7:]
    return out
