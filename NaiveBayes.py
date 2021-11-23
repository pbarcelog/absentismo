import os
import sys

#from pyspark.mllib.classification import NaiveBayes
from pyspark.ml.classification import NaiveBayes
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from datetime import datetime

import time

import numpy as np
from pyspark.ml.feature import MinMaxScaler, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

"""HADOOP_HOME = "C:\\Java\\winutils"

os.environ["HADOOP_HOME"] = HADOOP_HOME
sys.path.append(HADOOP_HOME + "\\bin")
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
"""
input_data = 'test.diaria'
num_cpus = '3'

now = datetime.now()
now_inicial = now
# cargaremos de mongo porque carga el schema bien, no como el csv
spark = SparkSession.builder.master("local[" + num_cpus + "]").appName("ml") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/" + input_data) \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.executor.memory", "3g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print('Ha tardado en sesión Spark: ', datetime.now() - now)

now = datetime.now()
# sparkDF = spark.createDataFrame(pd.read_csv('mycsv.csv'))
# sparkDF = spark.read.csv("mycsv.csv")

label = 'absentismo'
complement = 'clase_absentismo'

key = ['_id', complement, 'absentismo_rel', 'inicio', 'fin', 'hasta_max', 'desde', 'hasta', 'ausencia_abs',
       'ausencia_rel_parcial', 'fecha_nac', 'inicial_real', 'final_real', 'dia_matriz', 'feantig', 'desde_real',
       'hasta_real', 'year']

sparkDF = spark.read.format("mongo").load().drop(*key) #.sample(0.5)
print('Ha tardado en cargar DF: ', datetime.now() - now)

no_pers = True
if no_pers: sparkDF = sparkDF.drop('no_pers')

originales = sparkDF.columns

for col in [label, 'codpostal', 'idobj', 'proyectos', 'semana_numero', 'semana_santa', 'year_real']:
    sparkDF = sparkDF.withColumn(col, sparkDF[col].cast(StringType()))

numericas = ['ausencia_total_abs', 'ausencia_total_rel', 'dias_rango', 'h_sem', 'no_pers',
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
# print('Categorias: ', categories)

categories_numeric = []
for element in types:
    if element[1] == 'float' or element[1] == 'double': categories_numeric.append(element[0])

# print('Numéricas: ', categories_numeric)

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

# columns = ['dias_rango', 'h_sem', 'no_pers', 'semana_numero', 'porcentaje_de_discapacidad']
columns = categories_numeric

assembler = VectorAssembler().setInputCols(columns).setOutputCol('numeric_vector')
newDF = assembler.transform(sparkDF)
scaler = StandardScaler(inputCol='numeric_vector', outputCol='numeric_scaled')
scalerModel = scaler.fit(newDF.filter(newDF.year_real < 2018))
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
###      EQUILIBRAR LA MUESTRA CON UN CRITERIO        ###
###   NO LINEAL DEBIDO A LOS ENORMES DESEQUILIBRIOS   ###
#########################################################

now = datetime.now()

df2=newDF.groupby('label').count().cache()
# df2.show()
total = df2.agg(F.sum("count")).collect()[0][0]
number = df2.count()
print(total)
print(number)

(trainingData, testData) = (newDF.filter(newDF.year_real < 2018), newDF.filter(newDF.year_real == 2019))
nb = NaiveBayes()
model = nb.fit(trainingData)
result = model.transform(testData)
#result.show()
result.groupBy('label', 'prediction').count().show()

weights = True

if weights:
    weight = [total/(number*df2.collect()[i][1]) for i in range(number)]
    expression = 'CASE '
    for i in range(number):
        print(df2.collect()[i][0], weight[i])
        expression = expression + 'WHEN label = ' + str(df2.collect()[i][0]) + ' THEN ' + str(weight[i]) + ' '
    expression += 'ELSE "None" END'
    print(expression)
    #a = input('check')
    newDF = newDF.withColumn("weight", F.expr(expression)).withColumn('weight', F.col('weight').cast(FloatType()))
    df_train = newDF.filter(newDF.year_real<2019)

#newDF.coalesce(1).write.json("C:\\tmp\\weight.json", mode='overwrite')

else:
    fractions = [df2.collect()[i][1]/total for i in range(number)]
    balance = [np.sqrt(max(fractions)/x) for x in fractions]
    print(fractions, '\n', balance)
    df_train = newDF.filter(newDF.label == df2.collect()[0][0]).sample(True, balance[0])
    for i in range(1,number):
        df_train = df_train.union(newDF.filter(newDF.label == df2.collect()[i][0]).sample(True, balance[i]))
    df_train.groupby('label').count().show()


print('Equilibrar: ', datetime.now()-now)

########################################
###      ENTRENAR NAIVE BAYES        ###
###   VALIDACIÓN CON DATOS DE 2019   ###
########################################

(trainingData, testData) = (df_train, newDF.filter(newDF.year_real == 2019))
#testData.groupBy('label', 'prediction').count().show()

if weights:
    nb = NaiveBayes(weightCol='weight')
    model = nb.fit(trainingData)
    result = model.transform(testData)
    #result.show()
    result.groupBy('label', 'prediction').count().show()

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",  metricName="accuracy")
    accuracy = evaluator.evaluate(result)
    print("Test set accuracy = " + str(accuracy))


##############################################################
###    FALTA TODA LA PARTE DE EVALUACIÓN DE RESULTADOS     ###
###                                                        ###
##############################################################



spark.stop()
