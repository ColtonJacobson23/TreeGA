# Tree.py 
# Python file for GA in Lisp w/ Trees
# We'll get representation, fitness function, and I/O pairs
# For lisp running, we'll use subprocesses
# Could do lisp on python or something else personal purposes
# # clojure -e "(code)" for clojure 
# # subprocess these via python

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import random
from typing import TypedDict
import math
import json
import string

operators = "+-*/"

class Node:
    def __init__(self, data) -> None:
        self.left = None
        self.right = None
        self.data = data

class InternalNode(Node):
    def __init__(self, data) -> None:
        super().__init__(data)

class LeafNode(Node):
    def __init__(self, data) -> None:
        super().__init__(data)

# assume genome ~ root node==
class Individual(TypedDict):
    genome: InternalNode
    fitness: int

Population = list[Individual]

def parse_rec(code: list[str]) -> Node:
    # Assuming prefix notation for Lisp code
    if code[0] in operators:
        temp = InternalNode(data = code.pop(0))
        temp.left = parse_rec(code)
        temp.right = parse_rec(code)
        return temp
    else:
        return LeafNode(code.pop(0))


def parse_expression(code: str) -> InternalNode:
    code = code.replace("(", "")
    code = code.replace(")", "")
    codeArr = code.split()
    return parse_rec(codeArr)

def print_tree(root: InternalNode, indent: str) -> None: #indiv: Individual
    # root = indiv["genome"]
    if isinstance(root, LeafNode):
        print(indent + root.data)
        # print(f"\nAbove code has fitness of: {indiv['fitness']}")
        return
    print_tree(root=root.right, indent=indent+"    ")
    print(indent, root.data)
    print_tree(root=root.left, indent = indent+"    ")

def gen_rand_tree(size: int, root: InternalNode) -> InternalNode:
    # Leaf node gen, either a number or a variable (operands)
    if size <= 1:
        root.left = LeafNode(
            str(float(random.randrange(0,1000,1)) + random.random())
            )
        root.right = LeafNode(
            str(float(random.randrange(0,1000,1)) + random.random())
            )
        return root
    
    # Internal node gen
    if size > 1:
        root.left = InternalNode(
            random.choice(operators)
            )
        gen_rand_tree(size - 1, root.left)
        root.right = InternalNode(
            random.choice(operators)
            )
        gen_rand_tree(size - 1, root.right)



def initialize_individual(genome: InternalNode, fitness: int) -> Individual:
    """
    Purpose:        Create one individual
    Parameters:     genome as tree, fitness as integer (higher better)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a dict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    return {"genome": genome, "fitness": fitness}


# no example genome for now, add min/max depth to tree
def initialize_pop(min_depth: int, max_depth: int, pop_size: int) -> Population:
    """
    Purpose:        Create population to evolve
    Parameters:     Min depth, max depth, population size as int, min_depth <= max_depth
    User Input:     no
    Prints:         no
    Returns:        a population, as a list of Individuals
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    
    population: Population = []
    for x in range (pop_size):
        nodes = random.randint(min_depth, max_depth)
        

        new_gen = ""
        individual = initialize_individual(genome=new_gen, fitness=0)
        population.append(individual)
    return population


def recombine_pair(parent1: Individual, parent2: Individual) -> Population:
    """
    Purpose:        Recombine two parents to produce two children
    Parameters:     Two parents as Individuals
    User Input:     no
    Prints:         no
    Returns:        A population of size 2, the children
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    
    if (True):
        index = random.choice(range(len(parent1["genome"])))
        if index > len(parent1["genome"]) / 2:
            new_gen1 = parent1["genome"][:index]
            new_gen2 = parent2["genome"][:index]
        else:
            new_gen1 = parent1["genome"][index:]
            new_gen2 = parent2["genome"][index:]

        for x in range(len(parent1["genome"])):
            if parent1["genome"][x] not in new_gen2:
                new_gen2 += parent1["genome"][x]
            if parent2["genome"][x] not in new_gen1:
                new_gen1 += parent2["genome"][x]
    else:
        rec_rate = 0.5
        l1 = list(parent1)
        l2 = list(parent2)
        test1 = []
        test2 = []
        gen1 = ""
        gen2 = ""

        # dealing with lists of len() < len(parent genome), need to configure loops to properly track offset
        for y in range(2):
            for x in range(len(parent1["genome"])):
                if random.random() < rec_rate:
                    if y == 0 and len(l1) > 0:
                        test1.append((l1.pop(x-len(test1)),x))
                    elif len(l1) > 0:
                        test2.append((l2.pop(x-len(test2)),x))

        for x in range(len(parent1["genome"])):
            if x == test1[x][1]:
                gen1 += test1[x-len(test1)][0]
            else:
                gen1 += l1.pop(random.randint(0,len(l1)-1))

            if x == test2[x][1]:
                gen2 += test2[x-len(test1)][0]
            else:
                gen2 += l2.pop(random.randint(0,len(l2)-1))

        

        new_gen1 = "".join(gen1)
        new_gen2 = "".join(gen2)
    
    child1 = initialize_individual(genome=new_gen1, fitness=0)
    child2 = initialize_individual(genome=new_gen2, fitness=0)
    return [child1, child2]


def recombine_group(parents: Population, recombine_rate: float) -> Population:
    """
    Purpose:        Recombines a whole group, returns the new population
                    Pair parents 1-2, 2-3, 3-4, etc..
                    Recombine at rate, else clone the parents.
    Parameters:     parents and recombine rate
    User Input:     no
    Prints:         no
    Returns:        New population of children
    Modifies:       Nothing
    Calls:          ?
    """
    combination: Population = []
    for ipair in range(0, len(parents) - 1, 2):
        if random.random() < recombine_rate:
            child1, child2 = recombine_pair(
                parent1=parents[ipair], parent2=parents[ipair + 1]
            )
        else:
            child1, child2 = parents[ipair], parents[ipair + 1]
        combination.extend([child1, child2])
    return combination 

def mutate_individual(parent: Individual, mutate_rate: float) -> Individual:
    """
    Purpose:        Mutate one individual
    Parameters:     One parents as Individual, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    new_genome = parent["genome"].split()
    for ch in new_genome:
        if random.random() < mutate_rate:
            temp = random.choice(new_genome)
            tempInd = new_genome.index(ch)
            tempInd2 = new_genome.index(temp)
            new_genome[tempInd] = temp
            new_genome[tempInd2] = ch

    gen_str = ""
    for ch in parent["genome"]:
        gen_str = gen_str + ch
    mutant = initialize_individual(genome=gen_str, fitness=0)
    return mutant


def mutate_group(children: Population, mutate_rate: float) -> Population:
    """
    Purpose:        Mutates a whole Population, returns the mutated group
    Parameters:     Population, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    new_group: Population = []
    for child in children:
        new_group.append(mutate_individual(parent=child, mutate_rate=mutate_rate))
    return new_group


# DO NOT MODIFY >>>>
def evaluate_individual(individual: Individual) -> None:
    """
    Purpose:        Computes and modifies the fitness for one individual
                    Assumes and relies on the logc of ./corpus/2_counts.py
    Parameters:     One Individual
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The individual (mutable object)
    Calls:          Basic python only
    Example doctest:
    """
    layout = individual["genome"]

    # Basic return to home row, with no extra cost for repeats.
    fitness = 0
    for key in layout:
        fitness += count_dict[key] * int(DISTANCE[layout.find(key)])

    # Vowels on the left, Consosants on the right
    for pos, key in enumerate(layout):
        if key in "AEIOUY" and pos > 14:
            fitness += 1

    # Top-down guess at ideal ergonomics
    for key in layout:
        fitness += count_dict[key] * int(ERGONOMICS[layout.find(key)])

    # [] {} () <> should be adjacent.
    # () ar fixed by design choice (number line).
    # [] and {} are on same keys.
    # Perhaps ideally, <> and () should be on same keys too...
    right_edges = [4, 9, 14, 16, 23, 29, 34]
    for pos, key in enumerate(layout):
        # order of (x or y) protects index on far right:
        if key == "[" and (pos in right_edges or "]" != layout[pos + 1]):
            fitness += 1
        if key == "," and (pos in right_edges or "." != layout[pos + 1]):
            fitness += 1

    # Symbols should be toward edges.
    for pos, key in enumerate(layout):
        if key in "-[],.';/=":
            fitness += int(PREFER_EDGES[pos])

    # Keybr.com querty-dvorak average data as estimate of real hand
    for pos, key in enumerate(layout):
        fitness += count_dict[key] / data_combine[pos]

    # Shortcut characters (skip this one).
    # On right hand for keyboarders (left ctrl is usually used)
    # On left hand for mousers (for one-handed shortcuts).
    pass

    individual["fitness"] = fitness


# <<<< DO NOT MODIFY


def evaluate_group(individuals: Population) -> None:
    """
    Purpose:        Computes and modifies the fitness for population
    Parameters:     Objective string, Population
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The Individuals, all mutable objects
    Calls:          ?
    Example doctest:
    """
    for i in range(len(individuals)):
        evaluate_individual(individual=individuals[i])


def rank_group(individuals: Population) -> None:
    """
    Purpose:        Create one individual
    Parameters:     Population of Individuals
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The population's order (a mutable object)
    Calls:          ?
    Example doctest:
    """
    individuals.sort(key=lambda ind: ind["fitness"], reverse=False)


def parent_select(individuals: Population, number: int) -> Population:
    """
    Purpose:        Choose parents in direct probability to their fitness
    Parameters:     Population, the number of individuals to pick.
    User Input:     no
    Prints:         no
    Returns:        Sub-population
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    parents: Population = []
    fitnesses = [i["fitness"] for i in individuals]
    parents = random.choices(individuals, fitnesses, k=number)
    return parents


def survivor_select(individuals: Population, pop_size: int) -> Population:
    """
    Purpose:        Picks who gets to live!
    Parameters:     Population, and population size to return.
    User Input:     no
    Prints:         no
    Returns:        Population, of pop_size
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    new_pop = initialize_pop(DVORAK, pop_size)
    evaluate_group(new_pop)
    return individuals[:int(pop_size/2)] + new_pop
    #return individuals[:pop_size]
    


def evolve(example_genome: str) -> Population:
    """
    Purpose:        A whole EC run, main driver
    Parameters:     The evolved population of solutions
    User Input:     No
    Prints:         Updates every time fitness switches.
    Returns:        Population
    Modifies:       Various data structures
    Calls:          Basic python, all your functions
    """
    # To debug doctest test in pudb
    # Highlight the line of code below below
    # Type 't' to jump 'to' it
    # Type 's' to 'step' deeper
    # Type 'n' to 'next' over
    # Type 'f' or 'r' to finish/return a function call and go back to caller
    pop_size: int = 15000
    population = initialize_pop(example_genome, pop_size) 
    evaluate_group(population) 
    rank_group(population) 
    best_fitness = population[0]['fitness'] 
    perfect_fitness = len(example_genome) 
    counter = 0
    while best_fitness > 20:
        counter += 1 
        parents = parent_select(individuals=population, number=pop_size) 
        children = recombine_group(parents=parents, recombine_rate=0.9) 
        mutate_rate = (1 - best_fitness / perfect_fitness) / 5 
        mutants = mutate_group(children=children, mutate_rate=0.7) 
        evaluate_group(individuals=mutants) 
        everyone = population + mutants 
        rank_group(individuals=everyone) 
        population = survivor_select(individuals=everyone, pop_size=pop_size)
        #if best_fitness != population[0]['fitness']:
        best_fitness = population[0]['fitness'] 
        print('Iteration number', counter, 'with best individual', population[0])


    return population


# Seed for base grade.
# For the exploratory competition points (last 10),
# comment this one line out if you want, but put it back please.
seed = True

# DO NOT MODIFY >>>>
if __name__ == "__main__":
    # divider = "===================================================="
    # Execute doctests to protect main:
    # import doctest

    # doctest.testmod()
    # doctest.testmod(verbose=True)

    # if seed:
    #     random.seed(42)

    # population = evolve(example_genome="")

    # grade = 0

    # with open(file="results.txt", mode="w") as f:
    #     f.write(str(grade))

    # with open(file="best_ever.txt", mode="r") as f:
    #     past_record = f.readlines()[1]
    # if population[0]["fitness"] < float(past_record):
    #     with open(file="best_ever.txt", mode="w") as f:
    #         f.write(population[0]["genome"] + "\n")
    #         f.write(str(population[0]["fitness"]))

    # Celcius to Farenheit
    # prefix notation
    # F = ( + ( * C ( / 9 5 ) ) 32)
    print("generating tree")
    root = InternalNode("+")
    gen_rand_tree(2, root)
    print_tree(root, "")

