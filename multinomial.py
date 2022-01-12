import os
import sys
import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MultilabelClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from datetime import datetime, timedelta
import numpy as np
from pyspark.ml.feature import MinMaxScaler, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import training

"""HADOOP_HOME = "C:\\Java\\winutils"

os.environ["HADOOP_HOME"] = HADOOP_HOME
sys.path.append(HADOOP_HOME + "\\bin")
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
"""

import findspark ######### PROBAR SI SE ESTRELLA ##########
findspark.init() #########      CON TAREAS       ##########

def multinomial(input_data, num_cpus):

    now = datetime.now()
    now_inicial = now
    # cargaremos de mongo porque carga el schema bien, no como el csv
    spark = SparkSession.builder.master("local["+num_cpus+"]").appName("ml") \
            .config("spark.mongodb.input.uri", "mongodb://localhost:27017/" + input_data) \
            .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
            .config("spark.executor.memory", "3g")\
            .config("spark.driver.memory", "8g")\
            .getOrCreate()

    print('Ha tardado en sesión Spark: ', datetime.now() - now)

    now = datetime.now()
    #sparkDF = spark.createDataFrame(pd.read_csv('mycsv.csv'))
    #sparkDF = spark.read.csv("mycsv.csv")

    label = 'absentismo'
    complement = 'clase_absentismo'

    key = ['_id', complement, 'absentismo_rel', 'inicio', 'fin', 'hasta_max', 'desde', 'hasta', 'ausencia_abs',
           'ausencia_rel_parcial', 'fecha_nac', 'inicial_real', 'final_real', 'dia_matriz', 'feantig', 'estado_civil',
           'desde_real', 'hasta_real', 'year']

    usar = True
    no_pers = True
    if usar:
        key.remove('dia_matriz')

    sparkDF = spark.read.format("mongo").load().drop(*key) #sample(0.1)
    print('Ha tardado en cargar DF: ', datetime.now()-now)
    print(sparkDF.columns)

    original = sparkDF.columns

    if usar:
        sparkDF = sparkDF.withColumn('dia_matriz', F.col('dia_matriz').cast(DateType()))
        original.remove('dia_matriz')

    for col in [label, 'codpostal', 'semana_santa', 'year_real']:
        sparkDF = sparkDF.withColumn(col, sparkDF[col].cast(StringType()))

    numericas = ['ausencia_total_abs', 'ausencia_total_rel', 'idobj', 'proyectos', 'dias_rango', 'h_sem', 'no_pers',
                 'semana_numero', 'porcentaje_de_discapacidad', 'edad', 'antiguedad']
    if no_pers:
        numericas.remove('no_pers')
        sparkDF = sparkDF.drop('no_pers')

    for col in numericas:
        sparkDF = sparkDF.withColumn(col, sparkDF[col].cast(FloatType()))

    #sparkDF.printSchema()

    #########################################################
    ###       SCALING DE LAS VARIABLES CATEGÓRICAS        ###
    ###   USANDO SOLO LOS DATOS DE TRAIN PARA EL MODELO   ###
    #########################################################

    now = datetime.now()

    """categories = []
    types = sparkDF.dtypes
    for element in types:
        if element[1] == 'string': categories.append(element[0])
    categories.remove('year_real')
    print('Categorias: ', categories)
    
    categories =  ['absentismo', 'categoria_profesional', 'centro_de_coste', 'clase_de_contrato',
                  'clase_de_instituto_de_ensenanz', 'clase_registro_parentesco', 'clave_de_sexo', 'codpostal',
                  'descripcion', 'dia_semana', 'division_de_personal', 'estado_civil', 'identificacion_de_contratos_se',
                   'motivo_de_la_medida', 'nacionalidad', 'numero_de_orden', 'semana_santa']
    """
    categories = ['absentismo', 'categoria_profesional', 'centro_de_coste', 'clase_de_contrato',
                  'clase_de_instituto_de_ensenanz', 'clase_registro_parentesco', 'clave_de_sexo', 'codpostal',
                  'descripcion', 'dia_semana', 'division_de_personal', 'identificacion_de_contratos_se',
                  'motivo_de_la_medida', 'nacionalidad', 'numero_de_orden', 'semana_santa']

    categories_numeric = numericas

    """categories_numeric = []
    for element in types:
        if element[1] == 'float' or element[1] == 'double': categories_numeric.append(element[0])
    """
    print('Numéricas: ', categories_numeric)

    categories_index = []
    for element in categories:
        categories_index.append(element + '_index')
    categories_index.remove(label + '_index')
    """
    for element in categories:
        thisDF = sparkDF.toPandas()
        thisDF[element] = pd.Categorical(thisDF[element])
        print (thisDF[element].cat.categories)
    a = input('Continuar?')
    # print('Index: ', categories_index)
    """
    categories_vector = []
    for element in categories:
        categories_vector.append(element + '_vector')
    categories_vector.remove(label + '_vector')

    # print('Vectors: ', categories_vector)

    """columns = ['dias_rango', 'h_sem', 'no_pers', 'semana_numero']
    if no_pers: columns.remove('no_pers')
    """
    columns = categories_numeric
    assembler = VectorAssembler().setInputCols(columns).setOutputCol('numeric_vector').setHandleInvalid("skip")
    newDF = assembler.transform(sparkDF)
    scaler = StandardScaler(inputCol='numeric_vector', outputCol='numeric_scaled')
    scalerModel = scaler.fit(newDF.filter(newDF.year_real < 2018))
    newDF = scalerModel.transform(newDF)

    print('Final Scaling')
    newDF.printSchema()
    newDF.show()


    print('Scaling: ', datetime.now()-now)

    ######################################################
    ###            PIPELINE DE INDEXACIÓN Y            ###
    ###   VECTORIZACIÓN DE LAS VARIABLES CATEGÓRICAS   ###
    ######################################################

    now = datetime.now()
    stages = [StringIndexer(inputCol=element, outputCol=element+'_index').setHandleInvalid("skip") for element in categories]
    label = label+'_index'
    encoder = OneHotEncoder(inputCols=categories_index, outputCols=categories_vector)
    stages += [encoder]
    features = categories_vector + ['numeric_scaled', 'ausencia_total_rel', 'porcentaje_de_discapacidad']
    print('Features (mirar que no está year): ', features)
    assembler = VectorAssembler(inputCols = features, outputCol="features")
    stages += [assembler]
    prepPipeline = Pipeline().setStages(stages)
    pipelineModel = prepPipeline.fit(newDF)
    original.remove('year_real')
    key = original + categories_index + categories_vector + ['numeric_vector', 'numeric_scaled']
    newDF = pipelineModel.transform(newDF).drop(*key).withColumnRenamed(label, 'label')
    newDF.first()

    print('Indexación: ', datetime.now()-now)

    #########################################################
    ###      EQUILIBRAR LA MUESTRA CON UN CRITERIO        ###
    ###   NO LINEAL DEBIDO A LOS ENORMES DESEQUILIBRIOS   ###
    #########################################################

    now = datetime.now()
    """
    trainDF = newDF.filter(newDF.year_real < 2019)
    df2=trainDF.groupby('label').count().cache()
    df2.show()
    total = df2.agg(F.sum("count")).collect()[0][0]
    number = df2.count()
    print(total)
    print(number)
    fractions = [df2.collect()[i][1]/total for i in range(number)]
    balance = [np.sqrt(max(fractions)/x) for x in fractions]
    print(fractions, '\n', balance)
    df_train = trainDF.filter(trainDF.label == df2.collect()[0][0]).sample(True, balance[0])
    for i in range(1,number):
        df_train = df_train.union(trainDF.filter(trainDF.label == df2.collect()[i][0]).sample(True, balance[i]))
    df_train.groupby('label').count().show()
    
    df_train.coalesce(1).write.json("C:\\tmp\\train.json", mode='overwrite')
    """
    now = datetime.now()

    df2=newDF.groupby('label').count().cache()
    print('df2 inicio equilibrio')
    df2.show()
    total = df2.agg(F.sum("count")).collect()[0][0]
    number = df2.count()
    print(total)
    print(number)

    weights = True

    if weights:
        weight = [total/(number*df2.collect()[i][1]) for i in range(number)]
        expression = 'CASE '
        for i in range(number):
            print(df2.collect()[i][0], weight[i])
            expression = expression + 'WHEN label = ' + str(df2.collect()[i][0]) + ' THEN ' + str(weight[i]) + ' '
        expression += 'ELSE 0 END'
        print(expression)
        #a = input('check')
        newDF = newDF.withColumn("weight", F.expr(expression)).withColumn('weight', F.col('weight').cast(FloatType()))
        df_train = newDF.filter(newDF.year_real < 2018)

    else:
        size = [df2.collect()[i][1] for i in range(number)]
        fractions = [df2.collect()[i][1] / total for i in range(number)]
        minimum = min(size)
        balance = [np.sqrt(max(fractions) / x) for x in fractions]
        print(fractions, '\n', balance)
        print(minimum)
        #minimum = float(input('Minimum que sea fracción < 1: '))
        df_train = newDF.filter(newDF.label == df2.collect()[0][0]).sample(True, balance[0])
        # df_train = newDF.filter(newDF.label == df2.collect()[0][0]).orderBy(F.rand()).limit(minimum)
        for i in range(1, number):
            df_train = df_train.union(newDF.filter(newDF.label == df2.collect()[i][0]).sample(True, balance[i]))
            #df_train = df_train.union(newDF.filter(newDF.label == df2.collect()[i][0]).orderBy(F.rand()).limit(minimum))
        print('Grupo equilibrado y muestra del df_train groupby label:')
        df_train.groupby('label').count().show()

    #newDF.coalesce(1).write.json("C:\\tmp\\weight.json", mode='overwrite')

    df_test = newDF.filter(newDF.year_real == 2019)
    print('df_test al final del equilibrado: ')
    df_test.printSchema()
    df_test.groupby('label').count().show()
    (df_train, df_test) = (df_train, df_test)

    print('Equilibrar: ', datetime.now()-now)

    #########################################################
    ###     ENTRENAR REGRESIÓN LOGÍSTICA MULTINOMIAL      ###
    ###          VALIDACIÓN CON DATOS DE 2019             ###
    #########################################################

    mode = 'multinomial'
    if label == 'absentismo_index': mode = 'binomial'
    print('Correspondencia :', label, mode)

    now = datetime.now()

    evaluator = BinaryClassificationEvaluator()
    if mode == 'multinomial': evaluator = MultilabelClassificationEvaluator()

    cross = False

    if cross:

        #lr = LogisticRegression(maxIter=10, featuresCol='features', labelCol='label')
        lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)

        paramGrid = ParamGridBuilder() \
            .addGrid(lr.elasticNetParam, [0.4, 0.6, 0.8]) \
            .addGrid(lr.regParam, [0.3, 0.1, 0.01]) \
            .build()

        crossval = CrossValidator(estimator=lr,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=4,
                                  parallelism=4)  # use 3+ folds in practice
        lr = crossval.fit(df_train)

        if weights:
            lrw = LogisticRegression(featuresCol='features', labelCol='label', weightCol='weight')
            crossvalw = CrossValidator(estimator=lrw,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=4,
                                  parallelism=4)  # use 3+ folds in practice
            lrw = crossvalw.fit(df_train)

    else:
        if weights:
            lrw = LogisticRegression(featuresCol='features', labelCol='label', weightCol='weight', regParam=0.3,
                                     elasticNetParam=0.8)
            lrw = lrw.fit(df_train)
        else:
            lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, featuresCol='features', labelCol='label')
            lr_model = lr.fit(df_train)
            lr = LogisticRegression(featuresCol='features', labelCol='label', elasticNetParam=0.8, regParam=0.3, maxIter=10)
            lr = lr.fit(df_train)

    # print('Resultados globales sin y con equilibrio de datos: ')
    # res_lr = lr.transform(df_test)
    # print('Recall de ausencias: ')

    if weights:
        #print('ElasticNetParam: ', lrw.getParam('elasticNetParam'))
        #print('ElasticNetParam: ', lrw.getParam('regParam'))
        lrw.write().overwrite().save('C:\\tmp\\logistic')
        res_lrw = lrw.transform(df_test)

    print('Entrenar y predecir: ', datetime.now()-now)

    # lr_model.save('C:\\tmp2\\lr_model')

    now = datetime.now()

    #testDf = newDF.filter(newDF.year_real == 2019).sample(0.2).select(['features', 'label'])
    #predictionsDf =lr_model.transform(testDf)

    # print('Predecir: ', datetime.now()-now)

    #predictionsDf.write.json(mode='overwrite', path='C:\\tmp2\\lr_prediction\\prediction.json')


    ################################################
    ###        EVALUACIÓN DE RESULTADOS:         ###
    ###    DATOS AGREGADOS Y POR EL DÍA A DÍA    ###
    ################################################

    print('Predicciones sin pesos y con pesos (si se ha seleccionado):')

    #prediccion_ausencias = res_lr.groupBy('label', 'prediction').count()
    #prediccion_ausencias.show()
    if weights:
        prediccion_ausencias = res_lrw.groupBy('label', 'prediction').count()
        prediccion_ausencias.show()

    """
    # se puede mejorar haciendo suma de columna en lugar de bucle
    total = 0
    prediction = 0
    real = 0
    for element in prediccion_ausencias.collect():
        if element['prediction'] == 1: prediction += element['count']
        else: total += element['count']
        if element['label'] == 1: real += element['count']
    total += prediction
    """

    prediction = prediccion_ausencias.where(F.col('prediction') == 1).count()
    real = prediccion_ausencias.where(F.col('label') == 1).count()
    total = prediccion_ausencias.count()

    print('Absentismo predicho: ', prediction/total)
    print('Absentismo real: ', real/total)
    print('Error sobre absentismo: ', abs((prediction - real)/total))

    predictions = res_lrw.groupby('dia_matriz', 'prediction').count().where(F.col('prediction') == 1)\
                         .withColumnRenamed('dia_matriz', 'dia_prediction').withColumnRenamed('count', 'count_prediction')
    labels = res_lrw.groupby('dia_matriz', 'label').count().orderBy('dia_matriz').where(F.col('label') == 1)\
                    .withColumnRenamed('dia_matriz', 'dia_label').withColumnRenamed('count', 'count_label')
    total = res_lrw.groupby('dia_matriz').count()
    total.printSchema()
    final = labels.join(predictions, labels.dia_label == predictions.dia_prediction)
    final = final.join(total, final.dia_label == total.dia_matriz)\
                .drop(*['dia_label', 'dia_prediction', 'label', 'prediction']).orderBy('dia_matriz')\
                .withColumn('absentismo', F.col('count_label')/F.col('count'))\
                .withColumn('prediction', F.col('count_prediction')/F.col('count'))\
                .withColumn('error_over_total', F.abs((F.col('count_prediction') - F.col('count_label')))/F.col('count'))\
                .withColumn('error_over_absences', F.abs((F.col('count_prediction') - F.col('count_label')))/F.col('count_label'))

    #df_stats = final.select(F.mean(col('error')).alias('mean'),F.stddev(col('error')).alias('std')).collect()
    print('Media del error sobre el total:')
    final.agg({'error_over_total': 'mean'}).show()
    print('Desviación del error sobre el total:')
    final.agg({'error_over_total': 'stddev'}).show()

    print('Media del error sobre absentismos:')
    final.agg({'error_over_absences': 'mean'}).show()
    print('Desviación del error obre absentismos:')
    final.agg({'error_over_absences': 'stddev'}).show()

    #mean = df_stats[0]['mean']
    #std = df_stats[0]['std']
    #print('Error:\n Media: ', mean, '\nDesviacion: ', std)

    final.orderBy('dia_matriz').withColumn('absentismo', F.round(F.col('absentismo'), 2))\
                 .withColumn('prediction', F.round(F.col('prediction'), 2))\
                 .withColumn('error_over_total', F.round(F.col('error_over_total'), 2))\
                 .withColumn('error_over_absences', F.round(F.col('error_over_absences'),2)).show()

    print('Media de la predicción:')
    final.agg({'count_prediction': 'mean'}).show()
    print('Desviación de la predicción:')
    final.agg({'count_prediction': 'stddev'}).show()

    spark.stop()

    return True
