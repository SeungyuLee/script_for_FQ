#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deap import algorithms, base, creator, tools, gp
import multiprocessing, random, numpy, datetime

import sg_transfer 

num_fqlayers = 7
min_bitwidth = 2
max_bitwidth = 5

creator.create("FitnessMax", base.Fitness, weights=(1,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, min_bitwidth, max_bitwidth)
toolbox.register("individual", tools.initRepeat, creator.Individual, \
		toolbox.attr_bool, n=num_fqlayers)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def cxTwoPointCopy(ind1, ind2):
	size = len(ind1)
	cxpoint1 = random.randint(1, size)
	cxpoint2 = random.randint(1, size-1)
	if cxpoint2 >= cxpoint1:
		cxpoint2 += 1
	else:
		cxpoint1, cxpoint2 = cxpoint2, cxpoint1

	ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
		= ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

	return ind1, ind2

toolbox.register("evaluate", sg_transfer.evalAccMax)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutUniformInt, low=min_bitwidth, up=max_bitwidth, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
"""
pool = multiprocessing.Pool()
toolbox.register("map", pool.map)
"""

def main():
	random.seed(64)
	pop = toolbox.population(n=20)
	hof = tools.HallOfFame(10, similar=numpy.array_equal)
	
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", numpy.mean)
	stats.register("std", numpy.std)
	stats.register("min", numpy.min)
	stats.register("max", numpy.max)

	start_time = datetime.datetime.now()
	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, \
			ngen=40, stats=stats, halloffame=hof, verbose=True)
	end_time = datetime.datetime.now()


	# result documentation
	f = open('result.txt', 'w')
	f.write('start time: %s\n' % (str(start_time)))
	f.write('end time: %s\n' % (str(end_time)))
	f.write('log: \n%s\n' % (str(log)))
	f.write('Hall of Fame 10: \n')
	for ind in range(len(hof)):
		f.write('%s\n' %(hof[ind]))
	f.close()

	print (hof)

	return pop, log, hof

if __name__ == "__main__":
	main()
