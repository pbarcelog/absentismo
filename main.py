import sys

from numpy.core.defchararray import lower

import workers
import matrix
import multinomial
import NaiveBayes
import RF

if(__name__== "__main__"):

    # valores por defecto (si no hay argumentos de entrada) es realizar las tres fases - carga y curación, generación
    # de matriz, aplicación de regresión multinomial - con la hoja excel pequeña, no la de todos los empleados
    stage = 0
    stages = 3
    model = 1
    db = 'small'
    excels = '_small.xlsx'

    if (len(sys.argv) == 0):

        print("\nRead & curate workers' data:   1 "
              "\nBuild day2day matrix       :   2 "
              "\nPrediction                 :   3 "
              )

        while True:
            answer = input('Single stage or multiple stages (s/-): ')
            answer = lower(answer)
            if answer == 's':
                stage = input('Enter stage to be executed: ')
            else:
                stages = input('Up to which stage to execute: ')

            if stage == 3 or stages == 3:
                print("\nMultinomial Regression:        1 "
                      "\nNaive Bayes:                   2 "
                      "\nRandom Forest:                 3 "
                      "\nNeural Network:                4 "
                      "\nMulti Layer Perceptron:        5\n")
                model = input('Enter type of model to use: ')

            largeorsmall = input("large or small matrix (l/-)?: ")
            if largeorsmall == 'l': db = 'test'

            if 1 <= stage <= 3 or 1 <= stages <= 5: break
            else: print('\nWrong values, please try again, valid values should be between 1-3 for stages and between 1-5 for predictions.\n')

    else:
        stage = int(sys.argv[1])
        stages = int(sys.argv[2])
        model = int(sys.argv[3])
        db = sys.argv[4]
        excels = sys.argv[5]

    if stage == 1 or stages>0:

        output_data = db+'.workers'
        num_cpus = '4'  # número cpus sesión spark
        if workers.workers(output_data, excels, num_cpus): print('\nSuccess\n')
        else: print('\nError\n')

    if stage == 2 or stages>1:

        input_data = db+'.workers'
        output_data = db+'.diaria'
        sample = False
        if sample: output_data = 'sample.diaria'
        num_cpus = '4'
        if matrix.matrix(input_data, output_data, num_cpus, sample): print('\nSuccess\n')
        else: print('\nError\n')

    if stage == 3 or stages>2:

        if model == 1:
            input_data = db+'.diaria'
            num_cpus = '3'
            #if multinomial.multinomial(input_data, num_cpus): print('\nSuccess\n')
            # else: print('\nError\n')
            if multinomial.multinomial(input_data, num_cpus): print('\nSuccess\n')
            else: print('\nError\n')

        elif model == 2:
            input_data = db+'.diaria'
            num_cpus = '3'
            if NaiveBayes.naive(input_data, num_cpus): print('\nSuccess\n')
            else: print('\nError\n')

        elif model == 3:
            input_data = db+'.diaria'
            num_cpus = '3'
            if RF.RF(input_data, num_cpus): print('\nSuccess\n')
            else: print('\nError\n')

        else: print('\nNon-valid parameters\n')

