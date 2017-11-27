import dynet
import argparse, random, time, sys, math, numpy, os, pdb
import util, models
from accumulator import Accumulator, accs, disps
from vocab import Vocab

# random.seed(78789) # I like setting a seed for consistent behavior when debugging

parser = argparse.ArgumentParser()

## need to have this dummy argument for dynet
parser.add_argument("--dynet-mem", help="set size of dynet memory allocation, in MB")
parser.add_argument("--dynet-gpu", help="use GPU acceleration")
parser.add_argument("--dynet-devices", default="", help="use multi-GPU acceleration")
parser.add_argument("--dynet-viz", help="visualize")

parser.add_argument("--actual-devices", default=None, help="explicitly set usage of devices")

parser.add_argument("--mode", default="baseline", choices={"baseline", "lattice", "memb"}, help="choose what thing to train right now")
parser.add_argument('--rebuild_vocab', action='store_true', help="rebuild the vocabulary rather than using the cached vocabulary")
parser.add_argument("--size", choices={"small", "medium", "large"}, help="convenience flag for setting the size of the RNN")
parser.add_argument("--dim", default=None, type=int, help="hidden dim of rnns")
parser.add_argument("--emb_dim", default=None, type=int, help="word embedding dim")
parser.add_argument("--dropout", default=None, type=float, help="set dropout probability")
parser.add_argument("--trainer", default="adam", choices={"sgd", "adam", "adagrad"}, help="choose training algorithm")
parser.add_argument("--learning_rate", type=float, default=None, help="set learning rate of trainer")
parser.add_argument("--epochs", default=None, type=int, help="maximum number of epochs to run experiment")
parser.add_argument("--minibatch_size", default=None, type=int, help="size of minibatches")

parser.add_argument("--ptb", action="store_true", help="run on ptb data")
parser.add_argument("--zh", action="store_true", help="chinese gmw data")

parser.add_argument("--log_train_every_n", default=100, type=int, help="how often to log training loss")
parser.add_argument("--log_valid_every_n", default=2000, type=int, help="how often to evaluate on validation set, log the loss, and potentially save off the model")

parser.add_argument("--load", help="location to load model from")
parser.add_argument("--save", help="location to save model to")
parser.add_argument("--output", help="location to output log to")
parser.add_argument("--name", help="use default locations to save and log")

parser.add_argument("--experiment", action="store_true", help="convenience flag for training a model with default settings")
parser.add_argument("--evaluate", action="store_true", help="convenience flag for runnning just test validation (no train step)")
parser.add_argument("--debug", action="store_true", help="convenience flag for debugging")

parser.add_argument("--lattice_size", default=1, type=int, help="size of lattice")
parser.add_argument("--multi_size", default=1, type=int, help="number of multiple embeddings")
parser.add_argument("--no_fixed_embs", action="store_true", help="turn off dynamic embs")
parser.add_argument("--no_dynamic_embs", action="store_true", help="turn off dynamic embs")
parser.add_argument("--no_fixed_preds", action="store_true", help="turn off fixed predictions")
parser.add_argument("--no_dynamic_preds", action="store_true", help="turn off dynamic predictions")
parser.add_argument("--use_cache", action="store_true", help="turn on embedding cache")
parser.add_argument("--train_with_random", action="store_true", help="turn on random selection of path during training")
parser.add_argument("--gumbel_sample", action="store_true", help="turn on random selection of path during training")
parser.add_argument("--max_chunk_vocab_size", default=10000, type=int, help="size of chunk vocab")
parser.add_argument("--test_samples", default=1, type=int, help="number of samples to take")
parser.add_argument("--concat_context_vector", action="store_true", help="concat context vector instead of initializing")

args = util.AttrDict(vars(parser.parse_args()))
print "Args:", args

if args.ptb:
    DATA_LOC = '../data/ptb'
    DATA_VIEW = 'word'
elif args.zh:
    DATA_LOC = '../data/zh'
    DATA_VIEW = 'char'
else:
    DATA_LOC = '../data/en/bpe'
    DATA_VIEW = 'word'

args.train_data = DATA_LOC + '/train'
args.valid_data = DATA_LOC + '/valid'
args.test_data = DATA_LOC + '/test'

args.start_token = u'<s>'
args.end_token = u'<e>'

if args.size == "small":
    if args.layers is None:     args.layers = 2
    if args.dim is None:        args.dim = 200
    if args.emb_dim is None:    args.emb_dim = 60
    if args.dropout is None:
        if args.zh:             args.dropout = 0.3
        else:                   args.dropout = 0.0
elif args.size == "medium":
    if args.layers is None:     args.layers = 2
    if args.dim is None:        args.dim = 650
    if args.emb_dim is None:    args.emb_dim = 200
    if args.dropout is None:
        if args.zh:                 args.dropout = 0.45
        else:                       args.dropout = 0.1
elif args.size == "large":
    if args.layers is None:     args.layers = 3
    if args.dim is None:        args.dim = 1500
    if args.emb_dim is None:    args.emb_dim = 300
    if args.dropout is None:
        if args.zh:             args.dropout = 0.6
        else:                   args.dropout = 0.1
else:
    if args.layers is None:     args.layers = 1
    if args.dim is None:        args.dim = 50
    if args.emb_dim is None:    args.emb_dim = 50
    if args.dropout is None:    args.dropout = 0

args.devices = args.dynet_devices.split(",")
if args.actual_devices is None:
    args.use_gpu = len(args.devices) > 1
    args.use_multigpu = len(args.devices) > 2
    args.param_device = args.devices[1] if args.use_gpu else "CPU"
    if args.use_multigpu: assert args.layers <= len(args.devices) - 1
    args.layer_devices = args.devices[2:] if args.use_multigpu else (
                         [args.devices[1]] * args.layers if args.use_gpu else
                         ["CPU"] * args.layers)
else:
    args.param_device = args.actual_devices.split(",")[0]
    args.layer_devices = args.actual_devices.split(",")[1:]

print "parameter device: %s, layer devices: %s" % (args.param_device, str(args.layer_devices))

if args.trainer == "sgd":
    trainer = dynet.SimpleSGDTrainer
    if args.learning_rate is None: args.learning_rate = .1
elif args.trainer == "adam":
    trainer = dynet.AdamTrainer
    if args.learning_rate is None: args.learning_rate = .001
elif args.trainer == "adagrad":
    trainer = dynet.AdagradTrainer
    if args.learning_rate is None: args.learning_rate = .01
else:
    raise Exception("unknown trainer: "+str(args.trainer))

if args.name:
    if args.output is None:    args.output = "logs/"+args.name+".log"
    if args.save is None:      args.save = "models/"+args.name

if args.experiment:
    if args.minibatch_size is None:    args.minibatch_size = 40
    if args.epochs is None:
        if args.zh:                    args.epochs = 10
        else:                          args.epochs = 3

if args.evaluate:
    if args.minibatch_size is None:    args.minibatch_size = 40
    if args.epochs is None:            args.epochs = 0

if args.debug:
    if args.minibatch_size is None:    args.minibatch_size = 1
    if args.epochs is None:            args.epochs = 0

if args.minibatch_size is None:    args.minibatch_size = 1
if args.epochs is None:            args.epochs = 10

if args.output:
    outfile = open(args.output, 'w')
    outfile.write("#\t"+str(args)+"\n\n")
    outfile.close()

model = dynet.Model()

trainer = trainer(model, args.learning_rate)

if not args.evaluate and not args.debug:
    train_data = util.load_data(args.train_data, DATA_VIEW, args.start_token, args.end_token)
    valid_data = util.load_data(args.valid_data, DATA_VIEW, args.start_token, args.end_token)
else:
    train_data = None

vocab = Vocab.load_from_data(train_data, save_cached=True,
                             load_cached=not args.rebuild_vocab,
                             cache_loc=args.train_data+".vocab."+DATA_VIEW)
vocab.start_token = vocab.add(args.start_token)
vocab.end_token = vocab.add(args.end_token)
vocab.delimiter = u' ' if DATA_VIEW == "word" else u''

args.temperature = 5

if args.mode == "lattice":
    chunk_vocab = Vocab(delimiter=u' ' if DATA_VIEW == "word" else u'')
    for token in vocab.strings: chunk_vocab.add(token)
    if not args.no_fixed_embs and args.lattice_size > 1:
        chunks_added = 0
        fname = ','.join([str(x) for x in range(2,args.lattice_size+1)])
        with open(DATA_LOC+"/vocabularies/"+fname+".toks") as f:
            for tok in f.read().decode('utf-8-sig').split("\n"):
                chunks_added += 1
                chunks_added += 1
                if chunks_added > args.max_chunk_vocab_size: break
                chunk_vocab.add_string(tok)

    chunk_vocab.add(u"<chunk_unk>")
    print "chunk count:", chunk_vocab.size

if args.mode == "baseline": lm = models.BaselineLanguageModel(model, args, vocab)
elif args.mode == "lattice": lm = models.LatticeLanguageModel(model, args, vocab, chunk_vocab)
elif args.mode == "memb": lm = models.MultiEmbLanguageModel(model, args, vocab)
else: raise Exception("unrecognized mode")

if args.load: model.load("models/"+args.load)

if not args.evaluate and not args.debug:
    train_batches = util.get_batches(train_data, args.minibatch_size)
    valid_batches = util.get_batches(valid_data, args.minibatch_size)

best_score = None
args.update_num = 0
train_accumulator = Accumulator(accs, disps)
_start = time.time()
for epoch_i in range(args.epochs):
    args.completed_epochs = epoch_i
    print "Epoch %d. Shuffling..." % epoch_i,
    if epoch_i == 0: train_batches = util.shuffle_preserve_first(train_batches)
    else: random.shuffle(train_batches)
    print "done."

    for i, batch in enumerate(train_batches):
        print i, len(batch), len(batch[0])
        args.update_num += 1
        dynet.renew_cg()
        result = lm.process_batch(batch, training=True)
        nancheck = result["loss"].value()
        while (not isinstance(numpy.isnan(nancheck), numpy.bool_) and True in numpy.isnan(nancheck)) or \
              (isinstance(numpy.isnan(nancheck), numpy.bool_) and numpy.isnan(nancheck) == True):
            print "nan encountered...redoing"
            sys.stdout.flush()
            result = lm.process_batch(batch, training=True)
        training_loss = dynet.sum_batches(result["loss"]) * dynet.scalarInput(1./len(batch), args.param_device)
        training_loss.backward()
        trainer.update()
        train_accumulator.update(result)

        if args.update_num % args.log_train_every_n == 0:
            print "(%d) Update %6d |" % (args.completed_epochs, args.update_num), train_accumulator.pp(), "| Time: %4f" % (time.time() - _start), " ",
            trainer.status()
            print
            train_accumulator = Accumulator(accs, disps)
            args.temperature *= .99
            _start = time.time()

        if args.update_num % args.log_valid_every_n == 0:
            valid_accumulator = Accumulator(accs, disps)
            _start = time.time()
            for v_i, v_batch in enumerate(valid_batches[:4]):
                dynet.renew_cg()
                result = lm.process_batch(v_batch, training=False)
                valid_accumulator.update(result)
            print "[[ Validation %6d | " % args.update_num, valid_accumulator.pp(), "| Time: %4f ]]" % (time.time() - _start),
            if args.output:
                print "(logging to %s)" % args.output,
                with open(args.output, "a") as outfile:
                    outfile.write("\n%d," % args.update_num + valid_accumulator.lp())
            if args.save:
                if best_score is None or best_score > valid_accumulator.values["loss"]:
                    print "new best...saving to", args.save,
                    best_score = valid_accumulator.values["loss"]
                    model.save(args.save)
            trainer.status()
            if "samples" in result: print "\n" + vocab.pp(result["samples"][0])
            print "\n"
            _start = time.time()

    # trainer.update_epoch()


# Test evaluation
test_data = util.load_data(args.test_data, DATA_VIEW, args.start_token, args.end_token)
test_batches = util.get_batches(test_data, args.minibatch_size)
if not args.debug:
    test_accumulator = Accumulator(accs, disps)
    _start = time.time()
    for t_i, t_batch in enumerate(test_batches):
        dynet.renew_cg()
        result = lm.process_batch(t_batch, training=False)
        test_accumulator.update(result)
    print "Test %6d | " % args.update_num, test_accumulator.pp(), "| Time: %4f" % (time.time() - _start)
    if args.output:
        print "(logging to %s)" % args.output
        with open(args.output, "a") as outfile:
            outfile.write("\nTEST," + test_accumulator.lp())

else:

#    dynet.renew_cg()
#    lm.DROPOUT = None
#    lm.instantiate_parameters()

#    embeddings = lm.vocab_R.npvalue()
#    dists = []
#    for word_i in range(lm.vocab.size):
#        dist = numpy.linalg.norm(embeddings[word_i,:]-embeddings[word_i+lm.vocab.size,:])
#        dists.append((dist, vocab[word_i]))

#    dists.sort(reverse=True)
#    for item in dists[:100]:
#        print vocab[item[1]].s.encode('utf-8'), item

    INSPECTION_WORD = "profile"
    ans1 = set([])
    ans2 = set([])
    for t_i, t_sent in enumerate(test_data):
        if INSPECTION_WORD not in t_sent: continue
        t_batch = [t_sent]
        dynet.renew_cg()
        paths = lm.process_batch(t_batch, training=False, debug=True)
        for prefix in paths:
            if vocab[prefix[0][1][0]].s != INSPECTION_WORD: continue
            prefs = sorted(prefix, key=lambda x:x[1][0])
            puts = [(thing[1][0], math.exp(thing[2].scalar_value())) for thing in prefs]
            denom = sum([thing[1] for thing in puts])
            s = u' '.join(t_sent).encode('utf-8')
            if puts[0][1] > puts[1][1]: ans1.add(s)
            else: ans2.add(s)
            # print vocab[puts[0][0]].s
            # for n_t, prob in puts: n_t, prob/denom, "|",
            # print u' '.join(t_sent).encode('utf-8')
            # raw_input()
        if len(ans1) >= 3 and len(ans2) >= 3: break
    print
    for item in list(ans1)[:3]: print item
    print
    for item in list(ans2)[:3]: print item

    print len(ans1), len(ans2)

    # bl = [-50003, -70002, -120000]
    # for b_i in bl:
    #     print
    #     b = test_batches[b_i]
    #     dynet.renew_cg()
    #     result = lm.process_batch(b, debug=True)
    #     for i, prefix in enumerate(result):
    #         dist = [pl[2].value() for pl in sorted(prefix, key=lambda x:len(x[1]))]
    #         probs = util.softmax(dist)
    #         print b[0][i],"\t",
    #         print "\t".join([("%2f" % prob) for prob in probs]) #, [pl[1] for pl in sorted(prefix, key=lambda x:len(x[1]))]
