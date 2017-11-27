import sys, os, argparse
from collections import defaultdict
from itertools import izip

parser = argparse.ArgumentParser()
parser.add_argument("--data_loc", type=str, help="folder of training data")
parser.add_argument("--out_loc", type=str, help="location to write files")
parser.add_argument("--n", type=int, default=10000, help="top n chunks get saved")
parser.add_argument("--char", action="store_true", help="char or nah")
args = parser.parse_args()

data_loc = sys.argv[1]
out_loc = sys.argv[2]

bi_counts = defaultdict(int)
tri_counts = defaultdict(int)

for filename in sorted(os.listdir(args.data_loc)):
  print filename
  with open(args.data_loc + "/" + filename) as f:
    for line in f.read().decode('utf-8-sig').split(u"\n"):
      if args.char:
        loop_over = line
      else:
        loop_over = line.split(u" ")

      if len(loop_over) < 2: continue

      bi_counts[(loop_over[0], loop_over[1])] += 1
      tri_counts[(loop_over[0], loop_over[1])] += 1
      for (a,b,c) in izip(loop_over, loop_over[1:], loop_over[2:]):
        bi_counts[(b, c,)] += 1
        tri_counts[(b, c,)] += 1
        tri_counts[(a, b, c)] += 1

bi_tops = sorted(bi_counts.items(), key=lambda x:-x[1])[:args.n]
tri_tops = sorted(tri_counts.items(), key=lambda x:-x[1])[:args.n]

delim = u"" if args.char else u" "

with open(args.out_loc + "/2.toks", "w") as f:
  f.write(u"\n".join([delim.join(ngram[0]) for ngram in bi_tops]).encode('utf-8-sig'))

with open(args.out_loc + "/2,3.toks", "w") as f:
  f.write(u"\n".join([delim.join(ngram[0]) for ngram in tri_tops]).encode('utf-8-sig'))
