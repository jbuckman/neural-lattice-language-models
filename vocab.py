from collections import namedtuple
import cPickle as pickle
import os

Token = namedtuple("Token", ["i", "s"])

class Vocab(object):
    def __init__(self, start_string=u"<s>", end_string=u"<e>", delimiter=u' ', unk=None):
        self.start_token = Token(0, start_string)
        self.end_token = Token(1, end_string)

        self.tokens = {self.start_token, self.end_token}
        self.strings = {self.start_token.s, self.end_token.s}
        self.s2t = {self.start_token.s: self.start_token, self.end_token.s: self.end_token}
        self.i2t = {self.start_token.i: self.start_token, self.end_token.i: self.end_token}
        self.counts = {self.start_token: 1, self.end_token: 1}
        self.delimiter = delimiter
        self.unk = unk if unk is None else self.add(unk)

    @property
    def size(self):
        return len(self.tokens)

    def add(self, thing):
        if isinstance(thing, Token): return self.add_token(thing)
        else: return self.add_string(thing)

    def add_string(self, string):
        string = string
        if string in self.strings:
            self.counts[self.s2t[string]] += 1
            return self.s2t[string]
        i = self.size
        s = string
        t = Token(i, s)
        self.i2t[i] = t
        self.s2t[s] = t
        self.tokens.add(t)
        self.strings.add(s)
        self.counts[t] = 1
        return t

    def add_token(self, tok):
        if tok in self.tokens: self.counts[tok] += 1
        else: self.counts[tok] = 1
        self.i2t[tok.i] = tok
        self.s2t[tok.s] = tok
        self.tokens.add(tok)
        self.strings.add(tok.s)
        return tok

    def __getitem__(self, key):
        if isinstance(key, int) and key < self.size: return self.i2t[key]
        elif isinstance(key, Token) and key in self.tokens: return key
        elif isinstance(key, basestring) and key in self.strings: return self.s2t[key]
        elif self.unk is None: raise Exception("tried to access oov token, and no unk found")
        else: return self.unk

    def pp(self, seq, delimiter=None):
        if delimiter is None: delimiter = self.delimiter
        return delimiter.join([self[item].s for item in seq])

    def masked_pp(self, seq, mask, delimiter=None):
        masked_seq = [t for t,m in zip(seq, mask) if m]
        return self.pp(masked_seq, delimiter)

    def count(self, thing):
        try: return self.counts[self[thing]]
        except: return 0

    def restrict_vocab(self, minimum_count=1, maximum_vocab=None):
        if maximum_vocab is None: non_unk_tokens = self.tokens
        else: non_unk_tokens = set(sorted(self.tokens, key=lambda tok: -self.count(tok))[:maximum_vocab])
        unk_tokens = {tok for tok in self.tokens if self.count(tok) < minimum_count}
        return non_unk_tokens - unk_tokens

    @classmethod
    def load_from_data(cls, data, save_cached=False, load_cached=False, cache_loc=None):
        if load_cached and os.path.isfile(cache_loc):
            assert cache_loc is not None
            with open(cache_loc) as f:
                return pickle.load(f)
        v = Vocab()
        for example in data:
            for token in example:
                v.add(token)
        if save_cached:
            assert cache_loc is not None
            with open(cache_loc, "w") as f:
                pickle.dump(v, f)
        return v

    def batchify(self, batch, use_string=False):
        sents = []
        masks = []
        maxSentLength = max([len(sent) for sent in batch])
        for sent in batch:
            if use_string: sents.append([word for word in sent] + [self.end_token.s for _ in range(maxSentLength-len(sent))])
            else:          sents.append([self[word].i for word in sent] + [self.end_token.i for _ in range(maxSentLength-len(sent))])
            masks.append([1            for _    in sent] + [0                for _ in range(maxSentLength-len(sent))])
        sents = map(list, zip(*sents))
        masks = map(list, zip(*masks))
        return sents, masks

    def unbatchify(self, batch):
        sents = map(list, zip(*batch))
        ans = []
        for sent in sents:
            try: ans.append(sent[:sent.index(self.end_token.i)+1])
            except: ans.append(sent)
        return ans
