import os
import sys

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from datetime import datetime

import time

import numpy as np
from pyspark.ml.feature import StandardScaler, StringIndexer, OneHotEncoder, VectorAssembler

"""HADOOP_HOME = "C:\\Java\\winutils"

os.environ["HADOOP_HOME"] = HADOOP_HOME
sys.path.append(HADOOP_HOME + "\\bin")
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
"""
def RF(input_data, num_cpus):

    now = datetime.now()
    now_inicial = now
    # cargaremos de mongo porque carga el schema bien, no como el csv
    spark = SparkSession.builder.master("local[" + num_cpus + "]").appName("ml") \
        .config("spark.mongodb.input.uri", "mongodb://localhost:27017/" + input_data) \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .config("spark.executor.memory", "3g")\
        .config("spark.driver.memory", "15g")\
        .getOrCreate()

    print('Ha tardado en sesión Spark: ', datetime.now() - now)

    now = datetime.now()
    # sparkDF = spark.createDataFrame(pd.read_csv('mycsv.csv'))
    # sparkDF = spark.read.csv("mycsv.csv")

    label = 'absentismo'
    complement = 'clase_absentismo'
    adicional = ['categoria_profesional', 'numero_de_orden', 'edad', 'antiguedad', 'nacionalidad']
    barrier = 2018
    prediction = 2018

    key = ['_id', complement, 'codpostal', 'ausencia_abs', 'ausencia_rel', 'fecha_nac', 'inicial_real', 'final_real',
           'dia_matriz', 'feantig', 'desde_real', 'hasta_real', 'year'] + adicional

    sparkDF = spark.read.format("mongo").load().drop(*key) #.sample(0.5)
    print('Ha tardado en cargar DF: ', datetime.now() - now)

    no_pers = True
    if no_pers:
        sparkDF = sparkDF.drop('no_pers')

    originales = sparkDF.columns

    for col in [label, 'semana_numero', 'semana_santa', 'year_real']:
        sparkDF = sparkDF.withColumn(col, sparkDF[col].cast(StringType()))

    numericas = ['ausencia_total_abs', 'ausencia_total_rel', 'idobj', 'proyectos', 'dias_rango', 'h_sem', 'no_pers',
                 'porcentaje_de_discapacidad']
    if no_pers: numericas.remove('no_pers')

    for col in numericas:
        sparkDF = sparkDF.withColumn(col, sparkDF[col].cast(FloatType()))

    # sparkDF = sparkDF.withColumnRenamed(label, 'label')

    sparkDF.printSchema()

    #########################################################
    ###       SCALING DE LAS VARIABLES CATEGÓRICAS        ###
    ###   USANDO SOLO LOS DATOS DE TRAIN PARA EL MODELO   ###
    #########################################################

    now = datetime.now()
    categories = []
    types = sparkDF.dtypes
    for element in types:
        if element[1] == 'string': categories.append(element[0])
    categories.remove('year_real')

    print('Categorias: ', categories)

    categories_numeric = []
    for element in types:
        if element[1] == 'float' or element[1] == 'double': categories_numeric.append(element[0])

    print('Numéricas: ', categories_numeric)

    categories_index = []
    for element in categories:
        categories_index.append(element + '_index')
    categories_index.remove(label + '_index')

    # print('Index: ', categories_index)

    categories_vector = []
    for element in categories:
        categories_vector.append(element + '_vector')
    categories_vector.remove(label + '_vector')

    # print('Vectors: ', categories_vector)

    columns = categories_numeric

    assembler = VectorAssembler().setInputCols(columns).setOutputCol('numeric_vector')
    newDF = assembler.transform(sparkDF)
    scaler = StandardScaler(inputCol='numeric_vector', outputCol='numeric_scaled')
    scalerModel = scaler.fit(newDF.filter(newDF.year_real < barrier))
    newDF = scalerModel.transform(newDF)

    print('Scaling: ', datetime.now()-now)

    ######################################################
    ###            INDEXACIÓN Y VECTORIZACIÓN          ###
    ###                                                ###
    ######################################################

    now = datetime.now()
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

    for element in categories:
        newDF = StringIndexer(inputCol=element, outputCol=element+'_index').setHandleInvalid("skip").fit(newDF).transform(newDF)
    label = label+'_index'
    features = categories_index+['numeric_scaled']+list(set(numericas)-set(columns))
    print(features)
    newDF = VectorAssembler(inputCols = features, outputCol="features").transform(newDF).withColumnRenamed(label, 'label')
    print('Indexación: ', datetime.now()-now)

    #########################################################
    ###   EQUILIBRAR LA MUESTRA (O NO) CON UN CRITERIO    ###
    ###   NO LINEAL DEBIDO A LOS ENORMES DESEQUILIBRIOS   ###
    #########################################################

    now = datetime.now()

    df2=newDF.filter(newDF.year_real < barrier).groupby('label').count().cache()
    df2.show()
    total = df2.agg(F.sum("count")).collect()[0][0]
    number = df2.count()
    print(total)
    print(number)

    weights = False
    equilibrium = False
    manual = True

    if weights:
        weight = [total/(number*df2.collect()[i][1]) for i in range(number)]
        print(weight)
        expression = 'CASE '
        for i in range(number):
            print(df2.collect()[i][0], weight[i])
            expression = expression + 'WHEN label = ' + str(df2.collect()[i][0]) + ' THEN ' + str(weight[i]) + ' '
        expression += 'ELSE "None" END'
        print(expression)
        #a = input('check')
        newDF = newDF.withColumn("weight", F.expr(expression)).withColumn('weight', F.col('weight').cast(FloatType()))
        df_train = newDF.filter(newDF.year_real<barrier)

    elif equilibrium:
        fractions = [df2.collect()[i][1]/total for i in range(number)]
        balance = [np.sqrt(max(fractions)/x) for x in fractions]
        print(fractions, '\n', balance)
        df_train = newDF.filter(newDF.label == df2.collect()[0][0]).sample(True, balance[0])
        for i in range(1,number):
            df_train = df_train.union(newDF.filter(newDF.label == df2.collect()[i][0]).sample(True, balance[i]))
        df_train.groupby('label').count().show()

    elif manual:
        df_train = newDF.filter(newDF.label == 0).sample(0.8)
        df_train = df_train.union(newDF.filter(newDF.label == 1).sample(True, 3.2))
        df_train.groupby('label').count().show()

    else: df_train = newDF.filter(newDF.year_real < barrier)

    print('Equilibrar: ', datetime.now()-now)

    ########################################
    ###     ENTRENAR RANDOM FOREST       ###
    ###   VALIDACIÓN CON DATOS DE 2019   ###
    ########################################

    (trainingData, testData) = (df_train, newDF.filter(newDF.year_real == prediction))

    # Train a RandomForest model.

    if weights:
        rf = RandomForestClassifier(numTrees = 200, featuresCol='features', labelCol='label', weightCol='weight', seed=0, maxBins=90)
        rf = rf.fit(trainingData)
        res_rf = rf.transform(testData)
        res_rf.groupBy('label', 'prediction').count().show()

    else:
        rf = RandomForestClassifier(numTrees=200, featuresCol='features', labelCol='label', seed=0,
                                    maxBins=90)  # .setThresholds([0.97])
        rf = rf.fit(trainingData)
        res_rf = rf.transform(testData)
        print('Validación 2019 con 0.8-3 de sampling:')
        res_rf.groupBy('label', 'prediction').count().show()
        print('Validación 2018')
        res_rf = rf.transform(newDF.filter(newDF.year_real == 2018))
        res_rf.groupBy('label', 'prediction').count().show()
        print('Validación 2020')
        res_rf = rf.transform(newDF.filter(newDF.year_real == 2020))
        res_rf.groupBy('label', 'prediction').count().show()
    #predictions = model.transform(testData)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    if label == 'absentismo_index': evaluator = BinaryClassificationEvaluator()

    print(label)
    #accuracy = evaluator.evaluate(predictions)

    #print("Accuracy (accuracy or ROC): ", accuracy)

    spark.stop()

    return True