from nltk.corpus import treebank
from nltk.grammar import CFG, Nonterminal, Production

from BottomUpParsing import *

def bikinStandarTreeBank(dataset):
    rules = []
    abandoned_treebank = []
    for tree in dataset:
        try:
            rules.append(Tree.fromstring(tree))
        except:
            abandoned_treebank.append(tree)
            
    return rules

# case folding lower case

def traverse(tree):
    for index, subtree in enumerate(tree):
        if type(subtree) == nltk.tree.Tree:
            traverse(subtree)
        elif type(subtree) == str:
            tree[index] = subtree.lower()


def bikinProductionRules(rules):
    distinct_rules = set(())
    for rule in rules:
        for prod in rule.productions():
            distinct_rules.add(str(prod))

    starting_point = []
    general_point = []
    for rule in distinct_rules:
        prod = rule.split(' ')
        if prod[0] == 'S':
            starting_point.append(rule)
        else:
            general_point.append(rule)

    productions = starting_point + general_point

    cfg_rules = '\n'.join(productions)

    grammars = nltk.CFG.fromstring(cfg_rules)
    
    return grammars

# import dataset
dataset = open('db/dataset (cleaning).bracket', 'r', encoding = "utf8").readlines()

#rules
rules = bikinStandarTreeBank(dataset)

#ke lowercase in
traverse(rules)

#production rules
grammars = bikinProductionRules(rules)

# convert sentences to parse tree (parsing)
parser = BottomUpParsing(grammars)

print(bikinProductionRules(rules))

def parsing(sentence):
    sentence = sentence.lower()
    try:
        parsed = list(parser.parse(sentence.split(' ')))
    except ValueError:
        print("Salah satu atau seluruh kata dalam kalimat tidak terdapat pada grammars.")
        return None
    else:
        print(parsed[0])
        print("\nGrammars: ")
        for x in parsed[0].productions():
            print(x)

        return str(parsed[0])