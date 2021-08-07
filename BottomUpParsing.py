from constituents import *

BU_STRATEGY = [LeafInitRule(),
               EmptyPredictRule(),
               BottomUpPredictRule(),
               SingleEdgeFundamentalRule()]
               
class ChartParser(ParserI):
    def __init__(self, grammar, strategy=BU_STRATEGY, trace=0,
                 trace_chart_width=50, use_agenda=True, chart_class=Chart):
        self._grammar = grammar
        self._strategy = strategy
        self._trace = trace
        self._trace_chart_width = trace_chart_width
        self._use_agenda = use_agenda
        self._chart_class = chart_class

        self._axioms = []
        self._inference_rules = []
        for rule in strategy:
            if rule.NUM_EDGES == 0:
                self._axioms.append(rule)
            elif rule.NUM_EDGES == 1:
                self._inference_rules.append(rule)
            else:
                self._use_agenda = False

    def grammar(self):
        return self._grammar

    def _trace_new_edges(self, chart, rule, new_edges, trace, edge_width):
        if not trace: return
        print_rule_header = trace > 1
        for edge in new_edges:
            if print_rule_header:
                print('%s:' % rule)
                print_rule_header = False
            print(chart.pretty_format_edge(edge, edge_width))

    def chart_parse(self, tokens, trace=None):
        if trace is None: trace = self._trace
        trace_new_edges = self._trace_new_edges

        tokens = list(tokens)
        self._grammar.check_coverage(tokens)
        chart = self._chart_class(tokens)
        grammar = self._grammar

        trace_edge_width = self._trace_chart_width // (chart.num_leaves() + 1)
        if trace: print(chart.pretty_format_leaves(trace_edge_width))
        
        if self._use_agenda:
            for axiom in self._axioms:
                new_edges = list(axiom.apply(chart, grammar))
                trace_new_edges(chart, axiom, new_edges, trace, trace_edge_width)

            inference_rules = self._inference_rules
            agenda = chart.edges()
            agenda.reverse()
            while agenda:
                edge = agenda.pop()
                for rule in inference_rules:
                    new_edges = list(rule.apply(chart, grammar, edge))
                    if trace:
                        trace_new_edges(chart, rule, new_edges, trace, trace_edge_width)
                    agenda += new_edges
                    
        else:
            edges_added = True
            while edges_added:
                edges_added = False
                for rule in self._strategy:
                    new_edges = list(rule.apply_everywhere(chart, grammar))
                    edges_added = len(new_edges)
                    trace_new_edges(chart, rule, new_edges, trace, trace_edge_width)
        return chart

    def parse(self, tokens, tree_class=Tree):
        chart = self.chart_parse(tokens)
        return iter(chart.parses(self._grammar.start(), tree_class=tree_class))

# main BottomUpParsing method

class BottomUpParsing(ChartParser):
    def __init__(self, grammar, **parser_args):
        if isinstance(grammar, PCFG):
            print("BottomUpParser only works for CFG!", category=DeprecationWarning)
        ChartParser.__init__(self, grammar, BU_STRATEGY, **parser_args)