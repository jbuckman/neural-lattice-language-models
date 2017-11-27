import code
import dynet
import numpy, random, math
from customlstm import TiedWeightsLSTMBuilder
from treelstm import WeightedTreeLSTMBuilder
from vocab import Vocab
import util

class BaselineLanguageModel(object):
    def __init__(self, model, args, vocab):
        self.model = model
        self.args = args
        self.vocab = vocab

        in_dim = self.args.emb_dim
        self.rnn = TiedWeightsLSTMBuilder(self.model, self.args.dim, self.args.layers, self.args.layer_devices, input_dim=in_dim)
        self.parameters = {
            "R": self.model.add_parameters((in_dim, self.args.dim), device=args.param_device),
            "bias": self.model.add_parameters((in_dim,), device=args.param_device),
            "vocab_R": self.model.add_parameters((self.vocab.size, in_dim), device=args.param_device),
            "vocab_bias": self.model.add_parameters((self.vocab.size,), device=args.param_device)
        }

    def instantiate_parameters(self):
        for param_name, param in self.parameters.items(): self.__dict__[param_name] = dynet.parameter(param)
        # self.vocab_lookup = dynet.transpose(self.vocab_R)

        if self.DROPOUT:
            y_t_mask = (1. / (1. - self.args.dropout)) * (
                numpy.random.uniform(size=[self.args.dim, self.BATCH_SIZE]) > self.args.dropout)
            self.dropout_mask_y_t = dynet.inputTensor(y_t_mask, batched=True, device=self.args.param_device)


    def process_batch(self, batch, training=False):
        self.TRAINING_ITER = training
        self.DROPOUT = self.args.dropout if (self.TRAINING_ITER and self.args.dropout > 0) else None
        self.BATCH_SIZE = len(batch)

        sents, masks = self.vocab.batchify(batch)

        self.instantiate_parameters()
        init_state = self.rnn.initial_state(mb_size=self.BATCH_SIZE, dropout=self.DROPOUT)

        # embeddings = [dynet.reshape(dynet.select_cols(self.vocab_lookup, toks), (self.args.dim,), self.BATCH_SIZE)
        # embeddings = [dynet.reshape(dynet.transpose(dynet.select_rows(self.vocab_R, toks)), (self.args.dim*2,), self.BATCH_SIZE)
        embeddings = [dynet.pick_batch(self.vocab_R, toks)
                      for toks in sents]
        outputs = init_state.transduce(embeddings)
        outputs = [dynet.to_device(out, self.args.param_device) for out in outputs]
        if self.DROPOUT: y_ts = [dynet.cmult(y_t, self.dropout_mask_y_t) for y_t in outputs]
        else: y_ts = outputs

        r_ts = [dynet.affine_transform([self.vocab_bias, self.vocab_R, dynet.tanh(dynet.affine_transform([self.bias, self.R, y_t]))]) for y_t in y_ts]
        errs = [dynet.pickneglogsoftmax_batch(r_t, toks) for r_t, toks in zip(r_ts, sents[1:])]

        for tok_i, (err, mask) in enumerate(zip(errs, masks[1:])):
            if min(mask) == 0: errs[tok_i] = err * dynet.inputTensor(mask, batched=True, device=self.args.param_device)

        err = dynet.esum(errs)
        char_count = [1+len(self.vocab.pp(sent[1:-1])) for sent in batch]
        word_count = [len(sent[1:]) for sent in batch]
        # word_count = [2+self.vocab.pp(sent[1:-1]).count(' ') for sent in batch]
        return {"loss": err,
                "charcount": char_count,
                "wordcount": word_count}

class LatticeLanguageModel(object):
    def __init__(self, model, args, vocab, chunk_vocab):
        assert args.dim % 2 == 0

        self.model = model
        self.args = args
        self.lattice_vocab = vocab
        self.chunk_vocab = chunk_vocab
        for word in self.lattice_vocab.strings: self.chunk_vocab.add(word)
        self.lattice_vocab.chunk_start = self.lattice_vocab.add("<chunk_start>")
        self.lattice_vocab.chunk_end = self.lattice_vocab.add("<chunk_end>")

        treernn_input_dim = self.args.emb_dim if (self.args.no_dynamic_preds or self.args.no_fixed_preds) else self.args.emb_dim*2
        self.rnn = WeightedTreeLSTMBuilder(self.model, self.args.dim, self.args.layers, self.args.layer_devices, input_dim=treernn_input_dim)

        # self.lattice_rnn = TiedWeightsLSTMBuilder(self.model, self.args.dim, self.args.layers, self.args.layers*[self.args.layer_devices[0]], input_dim=self.args.emb_dim)
        # self.lattice_fwd_comp_rnn = TiedWeightsLSTMBuilder(self.model, self.args.emb_dim / 2, self.args.layers, self.args.layers*[self.args.layer_devices[1]], input_dim=self.args.emb_dim)
        # self.lattice_bwd_comp_rnn = TiedWeightsLSTMBuilder(self.model, self.args.emb_dim / 2, self.args.layers, self.args.layers*[self.args.layer_devices[1]], input_dim=self.args.emb_dim)
        self.lattice_rnn = TiedWeightsLSTMBuilder(self.model, self.args.emb_dim, self.args.layers, self.args.layer_devices, input_dim=self.args.emb_dim)
        self.lattice_fwd_comp_rnn = TiedWeightsLSTMBuilder(self.model, self.args.emb_dim / 2, self.args.layers, self.args.layer_devices, input_dim=self.args.emb_dim)
        self.lattice_bwd_comp_rnn = TiedWeightsLSTMBuilder(self.model, self.args.emb_dim / 2, self.args.layers, self.args.layer_devices, input_dim=self.args.emb_dim)
        self.parameters = {
            "project_main_to_lattice_init_R": self.model.add_parameters((self.args.emb_dim, self.args.dim), device=args.param_device),
            "lattice_R": self.model.add_parameters((self.args.emb_dim, self.args.emb_dim + self.args.dim if self.args.concat_context_vector else self.args.emb_dim), device=args.param_device),
            "lattice_bias": self.model.add_parameters((self.args.emb_dim,), device=args.param_device),
            "vocab_R": self.model.add_parameters((self.lattice_vocab.size, self.args.emb_dim), device=args.param_device),
            "vocab_bias": self.model.add_parameters((self.lattice_vocab.size,), device=args.param_device),
            "chunk_R": self.model.add_parameters((self.args.emb_dim, self.args.dim), device=args.param_device),
            "chunk_bias": self.model.add_parameters((self.args.emb_dim,), device=args.param_device),
            "chunk_vocab_R": self.model.add_parameters((self.chunk_vocab.size, self.args.emb_dim), device=args.param_device),
            "chunk_vocab_bias": self.model.add_parameters((self.chunk_vocab.size,), device=args.param_device)
            }

        self.first_time_memory_test = True # this is to prevent surprise out-of-memory crashes

    def instantiate_parameters(self):
        for param_name, param in self.parameters.items(): self.__dict__[param_name] = dynet.parameter(param)
        # self.vocab_lookup = dynet.transpose(self.vocab_R)
        # self.chunk_vocab_lookup = dynet.transpose(self.chunk_vocab_R)

        if self.DROPOUT:
            y_t_mask = (1. / (1. - self.args.dropout)) * (
                numpy.random.uniform(size=[self.args.dim, self.BATCH_SIZE]) > self.args.dropout)
            self.dropout_mask_y_t = dynet.inputTensor(y_t_mask, batched=True, device=self.args.param_device)
            y_t_lat_mask = (1. / (1. - self.args.dropout)) * (
                numpy.random.uniform(size=[self.args.emb_dim, self.BATCH_SIZE]) > self.args.dropout)
            self.dropout_mask_lattice_y_t = dynet.inputTensor(y_t_lat_mask, batched=True, device=self.args.param_device)
            self.rnn.initialize_dropout(self.DROPOUT, self.BATCH_SIZE)

        else:
            self.rnn.disable_dropout()

    def process_batch(self, batch, training=False, debug=False):
        if self.args.test_samples > 1 and not training:
            results = []
            for _ in range(self.args.test_samples):
                dynet.renew_cg()
                results.append(self.process_batch_internal(batch, training=training, debug=debug))
                results[-1]["loss"] = results[-1]["loss"].npvalue()
            dynet.renew_cg()
            for r in results: r["loss"] = dynet.inputTensor(r["loss"], batched=True)
            result = results[0]
            result["loss"] = -(dynet.logsumexp([-r["loss"] for r in results]) - math.log(self.args.test_samples))
            return result
        else:
            return self.process_batch_internal(batch, training=training, debug=debug)

    def process_batch_internal(self, batch, training=False, debug=False):
        self.TRAINING_ITER = training
        self.DROPOUT = self.args.dropout if (self.TRAINING_ITER and self.args.dropout > 0) else None
        self.BATCH_SIZE = len(batch)
        self.instantiate_parameters()

        if self.args.use_cache: self.initialize_cache(batch)

        sents, masks = self.lattice_vocab.batchify(batch)

        # paths represent the different connections within the lattice. paths[i] contains all the state/chunk pairs that
        #  end at index i
        paths = [[] for _ in range(len(sents))]
        paths[0] = [(self.rnn.fresh_state(init_to_zero=True), [sents[0]], dynet.scalarInput(0.0))]
        for tok_i in range(len(sents)-1):
            # calculate the total probability of reaching this state
            _, _, lps = zip(*paths[tok_i])
            if len(lps) == 1: cum_lp = lps[0]
            else:             cum_lp = dynet.logsumexp(list(lps))

            # add all previous state/chunk pairs to the tree_lstm
            new_state = self.rnn.fresh_state()
            if self.TRAINING_ITER and self.args.train_with_random and not self.first_time_memory_test:
                state, c_t, lp = random.choice(paths[tok_i])
                if self.args.use_cache: x_t = self.cached_embedding_lookup(c_t)
                else: x_t = self.get_chunk_embedding(c_t)
                h_t_stack, c_t_stack = state.add_input(x_t)
                new_state.add_history(h_t_stack, c_t_stack, lp)
            else:
                self.first_time_memory_test = False
                for state, c_t, lp in paths[tok_i]:
                    if self.args.use_cache: x_t = self.cached_embedding_lookup(c_t)
                    else: x_t = self.get_chunk_embedding(c_t)
                    h_t_stack, c_t_stack = state.add_input(x_t)
                    new_state.add_history(h_t_stack, c_t_stack, lp)

            # treeLSTM state merging
            new_state.concat_weights()
            if self.args.gumbel_sample:
                new_state.apply_gumbel_noise_to_weights(temperature=max(.25, self.args.temperature))
                if not self.TRAINING_ITER: new_state.weights_to_argmax()
                # new_state.weights_to_argmax()

            # output of tree_lstm
            y_t = new_state.output()
            y_t = dynet.to_device(y_t, self.args.param_device)
            if self.DROPOUT: y_t = dynet.cmult(y_t, self.dropout_mask_y_t)

            # based on lattice_size, decide what set of chunks to consider from here
            if self.args.lattice_size < 1: end_tok_i = len(sents)
            else:                          end_tok_i = min(tok_i + 1 + self.args.lattice_size, len(sents))
            next_chunks = sents[tok_i + 1: end_tok_i]

            # for each chunk, calculate the probability of that chunk, and then add a pointer to the state/chunk into
            #  the place in the sentence where the chunk will end
            assert not (self.args.no_fixed_preds and self.args.no_dynamic_preds)
            if not self.args.no_fixed_preds: fixed_chunk_lps, use_dynamic_lp = self.predict_chunks(y_t, next_chunks)
            if not self.args.no_dynamic_preds: dynamic_chunk_lps = self.predict_chunks_by_tokens(y_t, next_chunks)
            for chunk_i, tok_loc in enumerate(range(tok_i + 1, end_tok_i)):
                if self.args.no_fixed_preds:
                    lp = dynamic_chunk_lps[chunk_i]
                elif self.args.no_dynamic_preds:
                    lp = fixed_chunk_lps[chunk_i]
                else: # we are using both fixed & dynamic predictions
                    lp = dynet.logsumexp([fixed_chunk_lps[chunk_i], use_dynamic_lp + dynamic_chunk_lps[chunk_i]])
                paths[tok_loc].append((new_state, sents[tok_i + 1:tok_loc + 1], cum_lp + lp))

        ending_masks = [[0.0]*self.BATCH_SIZE for _ in range(len(masks))]
        for sent_i in range(len(batch)): ending_masks[batch[sent_i].index(self.lattice_vocab.end_token.s)][sent_i] = 1.0

        # put together all of the final path states to get the final error
        cum_lp = dynet.scalarInput(0.0)
        for path, mask in zip(paths, ending_masks):
            if max(mask) == 1:
                assert len(path) != 0
                _, _, lps = zip(*path)
                if len(lps) == 1: local_cum_lp = lps[0]
                else:             local_cum_lp = dynet.logsumexp(list(lps))
                cum_lp += local_cum_lp * dynet.inputTensor(mask, batched=True)

        if debug: return paths

        err = -cum_lp
        char_count = [1+len(self.lattice_vocab.pp(sent[1:-1])) for sent in batch]
        word_count = [len(sent[1:]) for sent in batch]
        # word_count = [2+self.lattice_vocab.pp(sent[1:-1]).count(' ') for sent in batch]
        return {"loss": err,
                "charcount": char_count,
                "wordcount": word_count}

    def get_chunk_embedding(self, chunks, masks=None):
        if masks is None: merged_chunks = [self.lattice_vocab.pp(chunk) for chunk in map(list, zip(*chunks))]
        else: merged_chunks = [self.lattice_vocab.masked_pp(chunk, mask) for chunk, mask in zip(map(list, zip(*chunks)), map(list, zip(*masks)))]
        chunk_emb_is = [self.chunk_vocab[chunk].i if chunk in self.chunk_vocab.strings else self.chunk_vocab['<chunk_unk>'].i
                        for chunk in merged_chunks]
        # fixed_embs = dynet.reshape(dynet.transpose(dynet.select_rows(self.chunk_vocab_R, chunk_emb_is)), (self.args.dim,), len(chunk_emb_is))
        # fixed_embs = dynet.reshape(dynet.select_cols(self.chunk_vocab_lookup, chunk_emb_is), (self.args.dim,), len(chunk_emb_is))
        fixed_embs = dynet.pick_batch(self.chunk_vocab_R, chunk_emb_is)
        if self.args.no_dynamic_embs:
            return fixed_embs
        else:
            dynamic_embs = self.compress_chunk(chunks, masks)
            full_embs = dynet.concatenate([fixed_embs, dynamic_embs])
            return full_embs

    def compress_chunk(self, chunks, masks=None):
        compression_batch_size = len(chunks[0])
        # token_embeddings = [dynet.reshape(dynet.select_cols(self.vocab_lookup, tokens), (self.args.dim,), compression_batch_size)
        # token_embeddings = [dynet.reshape(dynet.transpose(dynet.select_rows(self.vocab_R, tokens)), (self.args.dim,), compression_batch_size)
        token_embeddings = [dynet.pick_batch(self.vocab_R, tokens)
                            for tokens in chunks]
        fwd_state = self.lattice_fwd_comp_rnn.initial_state(mb_size=compression_batch_size, dropout=self.DROPOUT)
        bwd_state = self.lattice_bwd_comp_rnn.initial_state(mb_size=compression_batch_size, dropout=self.DROPOUT)
        if masks is None:
            fwd_emb = fwd_state.transduce(token_embeddings)[-1]
            bwd_emb = bwd_state.transduce(list(reversed(token_embeddings)))[-1]
        else:
            masks = [dynet.inputTensor(mask, batched=True, device=self.args.param_device) if min(mask) == 0 else None for mask in masks]
            fwd_emb = fwd_state.transduce(token_embeddings, masks)[-1]
            bwd_emb = bwd_state.transduce(reversed(token_embeddings), reversed(masks))[-1]
        emb = dynet.concatenate([fwd_emb, bwd_emb])
        emb = dynet.to_device(emb, self.args.param_device)
        return emb

    def predict_chunks(self, w_t, chunk_batch):
        r_t = dynet.affine_transform([self.chunk_vocab_bias, self.chunk_vocab_R,
                                      dynet.tanh(dynet.affine_transform([self.chunk_bias, self.chunk_R, w_t]))])

        dyn_idx = [self.chunk_vocab['<chunk_unk>'].i] * self.BATCH_SIZE
        dyn_lp = -dynet.pickneglogsoftmax_batch(r_t, dyn_idx)

        partial_chunk_lps = []
        partial_chunks = [[] for _ in range(self.BATCH_SIZE)]
        for toks in chunk_batch:
            for tok, pc in zip(toks, partial_chunks):
                pc.append(tok)
            merged_chunks = [self.lattice_vocab.pp(pc) for pc in partial_chunks]
            chunk_emb_is = [self.chunk_vocab[chunk].i if chunk in self.chunk_vocab.strings else self.chunk_vocab['<chunk_unk>'].i
                            for chunk in merged_chunks]
            chunk_masks = [1 if chunk in self.chunk_vocab.strings else 0 for chunk in merged_chunks]
            chunk_mask_tensor = dynet.inputTensor(chunk_masks, batched=True, device=self.args.param_device)
            chunk_lps = -dynet.pickneglogsoftmax_batch(r_t, chunk_emb_is)
            chunk_lps = chunk_lps * chunk_mask_tensor + ((1.0 - chunk_mask_tensor) * -99999999.)
            partial_chunk_lps.append(chunk_lps)
        return partial_chunk_lps, dyn_lp

    def predict_chunks_by_tokens(self, w_t, chunk_batch):
        ender = [self.lattice_vocab.chunk_end.i] * self.BATCH_SIZE
        lps = []
        state = self.lattice_rnn.initial_state(dropout=self.DROPOUT)
        cs = [[self.lattice_vocab.chunk_start.i] * self.BATCH_SIZE] + chunk_batch
        cum_lp = dynet.scalarInput(0.0)
        for i, (cc, nc) in enumerate(zip(cs, cs[1:])):
            if self.args.concat_context_vector:
                x_t = dynet.pick_batch(self.vocab_R, cc)
                state.add_input(x_t)
            else:
                if i == 0:
                    state.add_input(self.project_main_to_lattice_init_R * w_t)
                else:
                    x_t = dynet.pick_batch(self.vocab_R, cc)
                    state.add_input(x_t)
            y_t = state.output()
            y_t = dynet.to_device(y_t, self.args.param_device)
            if self.DROPOUT: y_t = dynet.cmult(y_t, self.dropout_mask_lattice_y_t)
            if self.args.concat_context_vector: y_t = dynet.concatenate([y_t, w_t])
            r_t = dynet.affine_transform([self.vocab_bias, self.vocab_R, dynet.tanh(dynet.affine_transform([self.lattice_bias, self.lattice_R, y_t]))])
            if i > 0: lps.append(cum_lp + -dynet.pickneglogsoftmax_batch(r_t, ender))
            cum_lp = cum_lp + -dynet.pickneglogsoftmax_batch(r_t, nc)
        lps.append(cum_lp)
        return lps

    def initialize_cache(self, batch):
        chunk_set = set([])
        for sentence in batch:
            for start_i in range(len(sentence)):
                for end_i in range(start_i+1, min(len(sentence), start_i + 1 + self.args.lattice_size)):
                    chunk_set.add(tuple([self.lattice_vocab[word].i for word in sentence[start_i:end_i]]))

        chunk_list = list(chunk_set)
        chunks, masks = self.lattice_vocab.batchify(chunk_list)
        self.cache_locs = {chunk: i for i, chunk in enumerate(chunk_list)}
        self.cached_embeddings = self.get_chunk_embedding(chunks, masks)

    def cached_embedding_lookup(self, toks):
        chunks = map(tuple, zip(*toks))
        cache_is = [self.cache_locs[chunk] for chunk in chunks]
        return dynet.pick_batch_elems(self.cached_embeddings, cache_is)

class MultiEmbLanguageModel(object):
    def __init__(self, model, args, vocab):
        self.model = model
        self.args = args
        self.vocab = vocab

        input_dim = self.args.emb_dim / self.args.multi_size
        self.rnn = WeightedTreeLSTMBuilder(self.model, self.args.dim, self.args.layers, self.args.layer_devices, input_dim=input_dim)

        self.parameters = {
            "R": self.model.add_parameters((input_dim, self.args.dim), device=self.args.param_device),
            "bias": self.model.add_parameters((input_dim,), device=self.args.param_device),
            "vocab_R": self.model.add_parameters((self.vocab.size * self.args.multi_size, input_dim), device=self.args.param_device),
            "vocab_bias": self.model.add_parameters((self.vocab.size * self.args.multi_size,), device=self.args.param_device),
        }

        self.first_time_memory_test = True  # this is to prevent surprise out-of-memory crashes

    def instantiate_parameters(self):
        for param_name, param in self.parameters.items(): self.__dict__[param_name] = dynet.parameter(param)
        # self.vocab_lookup = dynet.transpose(self.vocab_R)
        # self.chunk_vocab_lookup = dynet.transpose(self.chunk_vocab_R)

        if self.DROPOUT:
            y_t_mask = (1. / (1. - self.args.dropout)) * (
                numpy.random.uniform(size=[self.args.dim, self.BATCH_SIZE]) > self.args.dropout)
            self.dropout_mask_y_t = dynet.inputTensor(y_t_mask, batched=True, device=self.args.param_device)
            self.rnn.initialize_dropout(self.DROPOUT, self.BATCH_SIZE)

        else:
            self.rnn.disable_dropout()

    def process_batch(self, batch, training=False, debug=False):
        if self.args.test_samples > 1 and not training:
            results = []
            for _ in range(self.args.test_samples):
                dynet.renew_cg()
                results.append(self.process_batch_internal(batch, training=training, debug=debug))
                results[-1]["loss"] = results[-1]["loss"].npvalue()
            dynet.renew_cg()
            for r in results: r["loss"] = dynet.inputTensor(r["loss"], batched=True, device=self.args.param_device)
            result = results[0]
            result["loss"] = -(dynet.logsumexp([-r["loss"] for r in results]) - dynet.scalarInput(math.log(self.args.test_samples), device=self.args.param_device))
            return result
        else:
            return self.process_batch_internal(batch, training=training, debug=debug)

    def process_batch_internal(self, batch, training=False, debug=False):
        self.TRAINING_ITER = training
        self.DROPOUT = self.args.dropout if (self.TRAINING_ITER and self.args.dropout > 0) else None
        self.BATCH_SIZE = len(batch)
        self.instantiate_parameters()

        if self.args.use_cache: self.initialize_cache(batch)

        sents, masks = self.vocab.batchify(batch)

        # paths represent the different connections within the lattice. paths[i] contains all the state/chunk pairs that
        #  end at index i
        paths = [[] for _ in range(len(sents))]
        paths[0] = [(self.rnn.fresh_state(init_to_zero=True), sents[0], dynet.scalarInput(0.0, device=self.args.param_device))]
        for tok_i in range(len(sents) - 1):
            # calculate the total probability of reaching this state
            _, _, lps = zip(*paths[tok_i])
            if len(lps) == 1:
                cum_lp = lps[0]
            else:
                cum_lp = dynet.logsumexp(list(lps))

            # add all previous state/chunk pairs to the tree_lstm
            new_state = self.rnn.fresh_state()
            if self.TRAINING_ITER and self.args.train_with_random and not self.first_time_memory_test:
                raise Exception("bruh")
            else:
                self.first_time_memory_test = False
                for state, c_t, lp in paths[tok_i]:
                    x_t = dynet.pick_batch(self.vocab_R, c_t)
                    h_t_stack, c_t_stack = state.add_input(x_t)
                    new_state.add_history(h_t_stack, c_t_stack, lp)

            # treeLSTM state merging
            new_state.concat_weights()
            if self.args.gumbel_sample:
                new_state.apply_gumbel_noise_to_weights(temperature=max(.25, self.args.temperature))
                if not self.TRAINING_ITER or self.args.sample_train: new_state.weights_to_argmax()
                # new_state.weights_to_argmax()

            # output of tree_lstm
            y_t = dynet.to_device(new_state.output(), self.args.param_device)
            if self.DROPOUT: y_t = dynet.cmult(y_t, self.dropout_mask_y_t)

            # get the list of next tokens to consider
            base_is = sents[tok_i+1]
            n_ts = [[nt + (i*self.vocab.size) for nt in base_is] for i in range(self.args.multi_size)]

            r_t = dynet.affine_transform([self.vocab_bias, self.vocab_R,
                                          dynet.tanh(dynet.affine_transform([self.bias, self.R, y_t]))])
            for n_t in n_ts:
                lp = -dynet.pickneglogsoftmax_batch(r_t, n_t)
                paths[tok_i+1].append((new_state, n_t, cum_lp + lp))

        ending_masks = [[0.0] * self.BATCH_SIZE for _ in range(len(masks))]
        for sent_i in range(len(batch)): ending_masks[batch[sent_i].index(self.vocab.end_token.s)][sent_i] = 1.0

        # put together all of the final path states to get the final error
        cum_lp = dynet.scalarInput(0.0, device=self.args.param_device)
        for path, mask in zip(paths, ending_masks):
            if max(mask) == 1:
                assert len(path) != 0
                _, _, lps = zip(*path)
                if len(lps) == 1:
                    local_cum_lp = lps[0]
                else:
                    local_cum_lp = dynet.logsumexp(list(lps))
                cum_lp += local_cum_lp * dynet.inputTensor(mask, batched=True, device=self.args.param_device)

        if debug: return paths

        err = -cum_lp
        char_count = [1 + len(self.vocab.pp(sent[1:-1])) for sent in batch]
        word_count = [len(sent[1:]) for sent in batch]
        # word_count = [2+self.lattice_vocab.pp(sent[1:-1]).count(' ') for sent in batch]
        return {"loss": err,
                "charcount": char_count,
                "wordcount": word_count}