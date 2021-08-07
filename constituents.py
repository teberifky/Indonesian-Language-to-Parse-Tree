from __future__ import print_function, division, unicode_literals

import nltk
from nltk import Tree

import re
import itertools
import warnings

from nltk.util import OrderedDict
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.parse.api import ParserI

class EdgeRelation(object):
    def __init__(self):
        if self.__class__ == EdgeRelation:
            raise TypeError('Edge is an abstract interface')

    def span(self):
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError()

    def end(self):
        raise NotImplementedError()

    def length(self):
        raise NotImplementedError()

    def lhs(self):
        raise NotImplementedError()

    def rhs(self):
        raise NotImplementedError()

    def dot(self):
        raise NotImplementedError()

    def nextsym(self):
        raise NotImplementedError()

    def is_complete(self):
        raise NotImplementedError()

    def is_incomplete(self):
        raise NotImplementedError()

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self._comparison_key == other._comparison_key)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, EdgeRelation):
            raise_unorderable_types("<", self, other)
        if self.__class__ is other.__class__:
            return self._comparison_key < other._comparison_key
        else:
            return self.__class__.__name__ < other.__class__.__name__

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(self._comparison_key)
            return self._hash

class TreeEdge(EdgeRelation):
    def __init__(self, span, lhs, rhs, dot=0):
        self._span = span
        self._lhs = lhs
        rhs = tuple(rhs)
        self._rhs = rhs
        self._dot = dot
        self._comparison_key = (span, lhs, rhs, dot)

    @staticmethod
    def from_production(production, index):
        return TreeEdge(span=(index, index), lhs=production.lhs(),
                        rhs=production.rhs(), dot=0)

    def move_dot_forward(self, new_end):
        return TreeEdge(span=(self._span[0], new_end),
                        lhs=self._lhs, rhs=self._rhs,
                        dot=self._dot+1)

    def lhs(self): return self._lhs
    def span(self): return self._span
    def start(self): return self._span[0]
    def end(self): return self._span[1]
    def length(self): return self._span[1] - self._span[0]
    def rhs(self): return self._rhs
    def dot(self): return self._dot
    def is_complete(self): return self._dot == len(self._rhs)
    def is_incomplete(self): return self._dot != len(self._rhs)
    def nextsym(self):
        if self._dot >= len(self._rhs): return None
        else: return self._rhs[self._dot]

    def __str__(self):
        str = '[%s:%s] ' % (self._span[0], self._span[1])
        str += '%-2r ->' % (self._lhs,)

        for i in range(len(self._rhs)):
            if i == self._dot: str += ' *'
            str += ' %s' % unicode_repr(self._rhs[i])
        if len(self._rhs) == self._dot: str += ' *'
        return str

    def __repr__(self):
        return '[Edge: %s]' % self

class LeafEdge(EdgeRelation):
    def __init__(self, leaf, index):
        self._leaf = leaf
        self._index = index
        self._comparison_key = (leaf, index)

    def lhs(self): return self._leaf
    def span(self): return (self._index, self._index+1)
    def start(self): return self._index
    def end(self): return self._index+1
    def length(self): return 1
    def rhs(self): return ()
    def dot(self): return 0
    def is_complete(self): return True
    def is_incomplete(self): return False
    def nextsym(self): return None
    
    def __str__(self):
        return '[%s:%s] %s' % (self._index, self._index+1, unicode_repr(self._leaf))
    def __repr__(self):
        return '[Edge: %s]' % (self)

class Chart(object):
    def __init__(self, tokens):
        self._tokens = tuple(tokens)
        self._num_leaves = len(self._tokens)

        self.initialize()

    def initialize(self):
        self._edges = []

        self._edge_to_cpls = {}

        self._indexes = {}

    def num_leaves(self):
        return self._num_leaves

    def leaf(self, index):
        return self._tokens[index]

    def leaves(self):
        return self._tokens

    def edges(self):
        return self._edges[:]

    def iteredges(self):
        return iter(self._edges)

    __iter__ = iteredges

    def num_edges(self):
        return len(self._edge_to_cpls)

    def select(self, **restrictions):
        if restrictions=={}: return iter(self._edges)

        restr_keys = sorted(restrictions.keys())
        restr_keys = tuple(restr_keys)

        if restr_keys not in self._indexes:
            self._add_index(restr_keys)

        vals = tuple(restrictions[key] for key in restr_keys)
        return iter(self._indexes[restr_keys].get(vals, []))

    def _add_index(self, restr_keys):
        for key in restr_keys:
            if not hasattr(EdgeRelation, key):
                raise ValueError('Bad restriction: %s' % key)

        index = self._indexes[restr_keys] = {}

        for edge in self._edges:
            vals = tuple(getattr(edge, key)() for key in restr_keys)
            index.setdefault(vals, []).append(edge)

    def _register_with_indexes(self, edge):
        for (restr_keys, index) in self._indexes.items():
            vals = tuple(getattr(edge, key)() for key in restr_keys)
            index.setdefault(vals, []).append(edge)

    def insert_with_backpointer(self, new_edge, previous_edge, child_edge):
        cpls = self.child_pointer_lists(previous_edge)
        new_cpls = [cpl+(child_edge,) for cpl in cpls]
        return self.insert(new_edge, *new_cpls)

    def insert(self, edge, *child_pointer_lists):
        if edge not in self._edge_to_cpls:
            self._append_edge(edge)
            self._register_with_indexes(edge)

        cpls = self._edge_to_cpls.setdefault(edge, OrderedDict())
        chart_was_modified = False
        for child_pointer_list in child_pointer_lists:
            child_pointer_list = tuple(child_pointer_list)
            if child_pointer_list not in cpls:
                cpls[child_pointer_list] = True
                chart_was_modified = True
        return chart_was_modified

    def _append_edge(self, edge):
        self._edges.append(edge)

    def parses(self, root, tree_class=Tree):
        for edge in self.select(start=0, end=self._num_leaves, lhs=root):
            for tree in self.trees(edge, tree_class=tree_class, complete=True):
                yield tree

    def trees(self, edge, tree_class=Tree, complete=False):
        return iter(self._trees(edge, complete, memo={}, tree_class=tree_class))

    def _trees(self, edge, complete, memo, tree_class):
        if edge in memo:
            return memo[edge]

        if complete and edge.is_incomplete():
            return []

        if isinstance(edge, LeafEdge):
            leaf = self._tokens[edge.start()]
            memo[edge] = [leaf]
            return [leaf]

        memo[edge] = []
        trees = []
        lhs = edge.lhs().symbol()

        for cpl in self.child_pointer_lists(edge):
            child_choices = [self._trees(cp, complete, memo, tree_class)
                             for cp in cpl]

            for children in itertools.product(*child_choices):
                trees.append(tree_class(lhs, children))

        if edge.is_incomplete():
            unexpanded = [tree_class(elt,[])
                          for elt in edge.rhs()[edge.dot():]]
            for tree in trees:
                tree.extend(unexpanded)

        memo[edge] = trees

        return trees

    def child_pointer_lists(self, edge):
        return self._edge_to_cpls.get(edge, {}).keys()

    def pretty_format_edge(self, edge, width=None):
        if width is None: width = 50 // (self.num_leaves()+1)
        (start, end) = (edge.start(), edge.end())

        str = '|' + ('.'+' '*(width-1))*start

        if start == end:
            if edge.is_complete(): str += '#'
            else: str += '>'

        elif edge.is_complete() and edge.span() == (0,self._num_leaves):
            str += '['+('='*width)*(end-start-1) + '='*(width-1)+']'
        elif edge.is_complete():
            str += '['+('-'*width)*(end-start-1) + '-'*(width-1)+']'
        else:
            str += '['+('-'*width)*(end-start-1) + '-'*(width-1)+'>'

        str += (' '*(width-1)+'.')*(self._num_leaves-end)
        return str + '| %s' % edge

    def pretty_format_leaves(self, width=None):
        if width is None: width = 50 // (self.num_leaves()+1)

        if self._tokens is not None and width>1:
            header = '|.'
            for tok in self._tokens:
                header += tok[:width-1].center(width-1)+'.'
            header += '|'
        else:
            header = ''

        return header

    def pretty_format(self, width=None):
        if width is None: width = 50 // (self.num_leaves()+1)
        edges = sorted([(e.length(), e.start(), e) for e in self])
        edges = [e for (_,_,e) in edges]

        return (self.pretty_format_leaves(width) + '\n' +
                '\n'.join(self.pretty_format_edge(edge, width) for edge in edges))

    def dot_digraph(self):
        s = 'digraph nltk_chart {\n'
        #s += '  size="5,5";\n'
        s += '  rankdir=LR;\n'
        s += '  node [height=0.1,width=0.1];\n'
        s += '  node [style=filled, color="lightgray"];\n'

        # Set up the nodes
        for y in range(self.num_edges(), -1, -1):
            if y == 0:
                s += '  node [style=filled, color="black"];\n'
            for x in range(self.num_leaves()+1):
                if y == 0 or (x <= self._edges[y-1].start() or
                              x >= self._edges[y-1].end()):
                    s += '  %04d.%04d [label=""];\n' % (x,y)

        # Add a spacer
        s += '  x [style=invis]; x->0000.0000 [style=invis];\n'

        # Declare ranks.
        for x in range(self.num_leaves()+1):
            s += '  {rank=same;'
            for y in range(self.num_edges()+1):
                if y == 0 or (x <= self._edges[y-1].start() or
                              x >= self._edges[y-1].end()):
                    s += ' %04d.%04d' % (x,y)
            s += '}\n'

        # Add the leaves
        s += '  edge [style=invis, weight=100];\n'
        s += '  node [shape=plaintext]\n'
        s += '  0000.0000'
        for x in range(self.num_leaves()):
            s += '->%s->%04d.0000' % (self.leaf(x), x+1)
        s += ';\n\n'

        # Add the edges
        s += '  edge [style=solid, weight=1];\n'
        for y, edge in enumerate(self):
            for x in range(edge.start()):
                s += ('  %04d.%04d -> %04d.%04d [style="invis"];\n' %
                      (x, y+1, x+1, y+1))
            s += ('  %04d.%04d -> %04d.%04d [label="%s"];\n' %
                  (edge.start(), y+1, edge.end(), y+1, edge))
            for x in range(edge.end(), self.num_leaves()):
                s += ('  %04d.%04d -> %04d.%04d [style="invis"];\n' %
                      (x, y+1, x+1, y+1))
        s += '}\n'
        return s

class ChartRule_(object):
    def apply(self, chart, grammar, *edges):
        raise NotImplementedError()

    def apply_everywhere(self, chart, grammar):
        raise NotImplementedError()

class AbstractChartRule(ChartRule_):
    def apply(self, chart, grammar, *edges):
        raise NotImplementedError()

    def apply_everywhere(self, chart, grammar):
        if self.NUM_EDGES == 0:
            for new_edge in self.apply(chart, grammar):
                yield new_edge

        elif self.NUM_EDGES == 1:
            for e1 in chart:
                for new_edge in self.apply(chart, grammar, e1):
                    yield new_edge

        elif self.NUM_EDGES == 2:
            for e1 in chart:
                for e2 in chart:
                    for new_edge in self.apply(chart, grammar, e1, e2):
                        yield new_edge

        elif self.NUM_EDGES == 3:
            for e1 in chart:
                for e2 in chart:
                    for e3 in chart:
                        for new_edge in self.apply(chart,grammar,e1,e2,e3):
                            yield new_edge

        else:
            raise AssertionError('NUM_EDGES>3 is not currently supported')

    def __str__(self):
        return re.sub('([a-z])([A-Z])', r'\1 \2', self.__class__.__name__)

class LeafInitRule(AbstractChartRule):
    NUM_EDGES=0
    def apply(self, chart, grammar):
        for index in range(chart.num_leaves()):
            new_edge = LeafEdge(chart.leaf(index), index)
            if chart.insert(new_edge, ()):
                yield new_edge

class EmptyPredictRule(AbstractChartRule):
    NUM_EDGES = 0
    def apply(self, chart, grammar):
        for prod in grammar.productions(empty=True):
            for index in compat.xrange(chart.num_leaves() + 1):
                new_edge = TreeEdge.from_production(prod, index)
                if chart.insert(new_edge, ()):
                    yield new_edge

class BottomUpPredictRule(AbstractChartRule):
    NUM_EDGES = 1
    def apply(self, chart, grammar, edge):
        if edge.is_incomplete(): return
        for prod in grammar.productions(rhs=edge.lhs()):
            new_edge = TreeEdge.from_production(prod, edge.start())
            if chart.insert(new_edge, ()):
                yield new_edge

class FundamentalRule(AbstractChartRule):
    NUM_EDGES = 2
    def apply(self, chart, grammar, left_edge, right_edge):
        # Make sure the rule is applicable.
        if not (left_edge.is_incomplete() and
                right_edge.is_complete() and
                left_edge.end() == right_edge.start() and
                left_edge.nextsym() == right_edge.lhs()):
            return

        new_edge = left_edge.move_dot_forward(right_edge.end())

        if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
            yield new_edge

class SingleEdgeFundamentalRule(FundamentalRule):
    NUM_EDGES = 1

    def apply(self, chart, grammar, edge):
        if edge.is_incomplete():
            for new_edge in self._apply_incomplete(chart, grammar, edge):
                yield new_edge
        else:
            for new_edge in self._apply_complete(chart, grammar, edge):
                yield new_edge

    def _apply_complete(self, chart, grammar, right_edge):
        for left_edge in chart.select(end=right_edge.start(),
                                      is_complete=False,
                                      nextsym=right_edge.lhs()):
            new_edge = left_edge.move_dot_forward(right_edge.end())
            if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
                yield new_edge

    def _apply_incomplete(self, chart, grammar, left_edge):
        for right_edge in chart.select(start=left_edge.end(),
                                       is_complete=True,
                                       lhs=left_edge.nextsym()):
            new_edge = left_edge.move_dot_forward(right_edge.end())
            if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
                yield new_edge