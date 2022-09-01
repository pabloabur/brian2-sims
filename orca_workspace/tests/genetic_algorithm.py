#!/bin/python

import os
import nevergrad as ng
trial_range = [trial for trial in range(0,9) ]
gaps = [g for g in range(0,40) ]


def objective_function(opt_param):
    gaps,trial = opt_param[0],opt_param[1]  
    gaps = gaps*9.0
    trial = trial*40.0
    os.system("python ../SLIF_extrapolation.py --gap "+str(gaps)+"--trial "+str(trial)+str("--path ")+os.getcwd())

    return error
  
optimizer = ng.optimizers.CMA(parametrization=2, budget=1500)
recommendation = optimizer.minimize(function_name_y,verbosity=0)  
results =9*recommendation.value[0],40.0*recommendation.value[1]
