from collections import namedtuple
import math
import util

AccumulatorItem = namedtuple("Accumulator", ["name", "init", "update"])
Display = namedtuple("Display", ["name", "func"])

class Accumulator(object):
    def __init__(self, items, displays):
        self.items = {item.name: item for item in items}
        self.values = {item.name: None for item in items}
        self.displays = displays

    def update(self, update_dict):
        for item_name in self.items:
            try:
                if self.values[item_name] is None:
                    self.values[item_name] = self.items[item_name].update(self.items[item_name].init(), update_dict)
                else:
                    self.values[item_name] = self.items[item_name].update(self.values[item_name], update_dict)
            except: pass

    def pp(self, delimiter=u' | '):
        outs = []
        for display in self.displays:
            try: outs.append(display.name + ": " + display.func(self.values))
            except: pass
        return delimiter.join(outs)

    def lp(self, delimiter=u','):
        outs = []
        for display in self.displays:
            try: outs.append(display.func(self.values))
            except: pass
        return delimiter.join(outs)


accs = [
    AccumulatorItem("loss", lambda :0, lambda v,d: v + sum(d["loss"].vec_value())),
    AccumulatorItem("klloss", lambda :0, lambda v,d: v + sum(d["klloss"].vec_value())),
    AccumulatorItem("klanneal", lambda :0, lambda v,d: d["klanneal"]),
    AccumulatorItem("discloss", lambda :0, lambda v,d: v + sum(d["discloss"].vec_value())),
    AccumulatorItem("genloss", lambda :0, lambda v,d: v + sum(d["genloss"].vec_value())),
    AccumulatorItem("reconloss", lambda :0, lambda v,d: v + sum(d["reconloss"].vec_value())),
    AccumulatorItem("convergence", lambda :0, lambda v,d: v + sum(d["convergence"].vec_value())),
    AccumulatorItem("charcount", lambda :0, lambda v,d: v + sum(d["charcount"])),
    AccumulatorItem("wordcount", lambda :0, lambda v,d: v + sum(d["wordcount"])),
    AccumulatorItem("sentcount", lambda :0, lambda v,d: v + len(d["wordcount"])),
]

disps = [
    Display("Loss", lambda d:"%4f" % (d["loss"] / d["wordcount"])),
    Display("KLL", lambda d:"%4f" % (d["klloss"] / d["wordcount"])),
    Display("KLA", lambda d:"%4f" % (d["klanneal"])),
    Display("Gen", lambda d:"%4f" % (d["genloss"] / d["wordcount"])),
    Display("Disc", lambda d:"%4f" % (d["discloss"] / d["wordcount"])),
    Display("Recon", lambda d:"%4f" % (d["reconloss"] / d["wordcount"])),
    Display("Conv", lambda d:"%4f" % (d["convergence"] / d["wordcount"])),
    Display("Perp", lambda d:"%4f" % math.exp(d["loss"] / d["wordcount"])),
    Display("BPC", lambda d:"%4f" % (d["loss"]/math.log(2) / d["charcount"])),
]