import os
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import when
from datetime import datetime

#jars_directory = 'jars/*'

#HADOOP_HOME = "C:\\hadoop"

#os.environ["HADOOP_HOME"] = HADOOP_HOME
#sys.path.append(HADOOP_HOME + "\\bin")
#os.environ['PYSPARK_PYTHON'] = sys.executable
#os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
from pyspark.sql.types import StringType

input_data = 'test.workers'
output_data = 'test.diaria'
sample = True
if sample: output_data = 'sample.diaria'
num_cpus = '6'

now = datetime.now()
now_inicial = now
# cargaremos de mongo porque carga el schema bien, no como el csv
spark = SparkSession.builder.master("local[" + num_cpus + "]").appName("ml") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/" + input_data) \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/" + output_data) \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.executor.memory", "3g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

#        .config("spark.jars.packages", 'mongodb-driver-reactivestreams-1.13.1-javadoc.jar
# .config("spark.mongodb.input.uri", "mongodb://localhost:27017/" + input_data) \
# .config("spark.mongodb.output.uri", "mongodb://localhost:27017/" + output_data) \
# .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
# .config('spark.driver.extraClassPath', 'jars/*') \
# sparkDF = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("uri", "mongodb://127.0.0.1/" + input_data).load()  #.drop(*key)

print('Ha tardado en sesión Spark: ', datetime.now() - now)

now = datetime.now()

# eliminar el id de mongo y otras columnas que no son de interés
key = ['_id', 'inicio', 'fin', 'hasta_max', 'desde', 'hasta']

sparkDF = spark.read.format('com.mongodb.spark.sql.DefaultSource').load().drop(*key)
if sample: sparkDF = sparkDF.sample(0.01)
sparkDF.printSchema()
print(sparkDF.count())
#a = input('cerrar')
# cambiar el tipo de variable de las columnas con fechas
for col in ['feantig', 'fecha_nac', 'desde_real', 'hasta_real', 'final_real', 'inicial_real']:
    sparkDF = sparkDF.withColumn(col, F.to_date(sparkDF[col]))
print('Ha tardado en cargar DF: ', datetime.now()-now)

now = datetime.now()

# En la siguiente sentencia se realizan los siguiente procesos:
# - construir una columna cuyas celdas tienen una lista con tantas comas como días tenía el contrato
# - posexplode que genera tantas filas como longitud de las listas y otra columna pos que viene a ser como
#   un contador de la poisición que ocupaba la coma en cada lista
# - crear columna con todas las fechas de vigencia de cada contrato
# - eliminar las columnas que se han usado para los cálculos anteriores pero que no tienen valor
# - crear columna con el día de la semana de cada fecha por nombre
# - crear columna con el día de la semana de cada fecha

print(sparkDF.columns)

KEY = ['ausencia_total_abs', 'ausencia_total_rel',
       'categoria_profesional', 'centro_de_coste', 'clase_de_contrato',
       'clase_de_instituto_de_ensenanz', 'clase_registro_parentesco', 'clave_de_sexo', 'codpostal', 'descripcion',
       'desde_real', 'dia_matriz', 'dias_rango', 'division_de_personal', 'feantig',
       'fecha_nac', 'h_sem', 'hasta_real', 'identificacion_de_contratos_se', 'idobj',
       'motivo_de_la_medida', 'nacionalidad', 'no_pers', 'numero_de_orden', 'porcentaje_de_discapacidad', 'proyectos'] #'semana_santa',

sparkDF = sparkDF.withColumn("repeat", F.expr("split(repeat(',', dias_rango-1), ',')"))\
                .select(F.col("*"),F.posexplode(F.col("repeat")))\
                .withColumn("dia_matriz", F.expr("date_add(desde_real, pos)"))\
                .drop("repeat","col", "pos") \

#sparkDF.printSchema()
#a = input('Seguir: ')

################################################
###   CÁLCULO DE LOS PARÁMETROS ASOCIADOS    ###
###   AL ABSENTISMO Y A LA CARACTERIZACIÓN   ###
###           DEL DÍA ESPECÍFICO             ###
################################################

dropKEY = KEY + ['absentismo']
print(sparkDF.columns)
sparkDF = sparkDF.withColumn('absentismo', F.when(F.col('dia_matriz').between(F.col('inicial_real'), F.col('final_real')),
                                   F.lit(1)).otherwise(F.lit(0))) \
                .dropDuplicates(dropKEY) \
                .withColumn('clase_absentismo', F.when(F.col('absentismo') == 1, F.col('clase_absentismo')).otherwise(F.lit('')))\
                .groupBy(KEY).agg(F.sum('absentismo').alias('absentismo'),
                                  F.collect_set('clase_absentismo').alias('set_clase_absentismo')) \
                .withColumn('absentismo_rel', F.col('absentismo') / F.col('proyectos')) \
                .withColumn('clase_absentismo', F.array_join(F.col('set_clase_absentismo'), ',')).drop('set_clase_absentismo')\
                .withColumn('clase_absentismo', F.when(F.col('clase_absentismo') == '', F.lit('no_absentismo'))
                            .otherwise(F.col('clase_absentismo')))\
                .withColumn('edad', F.datediff(F.col('dia_matriz'), F.col('fecha_nac')))\
                .withColumn('edad', when(F.col('edad') < 5840, F.lit(5840)).otherwise(F.col('edad')))\
                .withColumn('antiguedad', F.datediff(F.col('dia_matriz'), F.col('feantig')))\
                .withColumn('semana_santa', when(F.col('dia_matriz').between(F.to_date(F.lit('2014-04-13')),
                                                                             F.to_date(F.lit('2014-04-20'))), F.lit(1))
                                            .when(F.col('dia_matriz').between(F.to_date(F.lit('2015-03-29')),
                                                                              F.to_date(F.lit('2015-04-05'))), F.lit(1))
                                            .when(F.col('dia_matriz').between(F.to_date(F.lit('2016-03-20')),
                                                                              F.to_date(F.lit('2016-03-27'))), F.lit(1))
                                            .when(F.col('dia_matriz').between(F.to_date(F.lit('2017-04-09')),
                                                                              F.to_date(F.lit('2017-04-16'))), F.lit(1))
                                            .when(F.col('dia_matriz').between(F.to_date(F.lit('2018-03-29')),
                                                                              F.to_date(F.lit('2018-04-02'))), F.lit(1))
                                            .when(F.col('dia_matriz').between(F.to_date(F.lit('2019-04-14')),
                                                                              F.to_date(F.lit('2019-04-21'))), F.lit(1))
                                            .when(F.col('dia_matriz').between(F.to_date(F.lit('2020-04-05')),
                                                                              F.to_date(F.lit('2020-04-12'))), F.lit(1))
                                            .otherwise(F.lit(0)))\
                .withColumn("semana_numero", F.weekofyear(F.col("dia_matriz"))) \
                .withColumn("dia_semana", F.date_format(F.col("dia_matriz"), "E")) \
                .withColumn('year_real', F.year(F.col('dia_matriz')))

###############################
###    CATEGORIZACIÓN DE    ###
###   EDADES Y ANTIGUEDAD   ###
###############################

min_age = sparkDF.agg({'edad': 'min'}).first()[0]
max_age = sparkDF.agg({'edad': 'max'}).first()[0]
age_dif = max_age - min_age

segmentacion_edad = 6
franja = age_dif / segmentacion_edad
senectud = max_age - 2 * franja
gravitas = senectud - 2 * franja
juventud = gravitas - franja

print('Tramos de edades: edad mínima: ', min_age, ' juventud: ', juventud, ' gravitas: ', gravitas, ' senectud: ', senectud, ' edad máxima: ', max_age)

sparkDF = sparkDF.withColumn('franja_edad', when(F.col('edad') < juventud, 'adolescencia')
                             .when((F.col('edad') >= juventud) & (F.col('edad') < gravitas), 'juventud')
                             .when((F.col('edad') >= gravitas) & (F.col('edad') < senectud), 'gravitas').otherwise('senectud'))

min_antig = sparkDF.agg({'antiguedad': 'min'}).first()[0]
max_antig = sparkDF.agg({'antiguedad': 'max'}).first()[0]
antig_dif = max_antig - min_antig

rangos = []
segmentacion_antig = 40
rango = antig_dif / segmentacion_antig
rangos.append(min_antig + rango)
for i in range(segmentacion_antig - 1):
    rangos.append(rangos[i] + rango)
print('Tramos de antiguedades: antig min: ', min_antig, ' segmento: ', rango, ' Rangos: ', rangos)
sparkDF = sparkDF.withColumn('rango_antig', when(F.col('antiguedad') < rangos[0], 'rango1')
                             .when((F.col('antiguedad') >= rangos[0]) & (F.col('antiguedad') < rangos[1]), 'rango2')
                             .when((F.col('antiguedad') >= rangos[1]) & (F.col('antiguedad') < rangos[3]), 'rango3')
                             .when((F.col('antiguedad') >= rangos[3]) & (F.col('antiguedad') < rangos[5]), 'rango4')
                             .when((F.col('antiguedad') >= rangos[5]) & (F.col('antiguedad') < rangos[8]), 'rango5')
                             .otherwise('rango6'))


sparkDF.write.format("mongo").mode("overwrite").save()
print('Info salvada en mongo: ', datetime.now()-now)
absentismo = sparkDF.select('absentismo').where(sparkDF.absentismo == 1).count()
total = sparkDF.count()
print('Total registros: ', total, ' Absentismo: ', absentismo)
print('Absentismo total sin guardas ni vacaciones: ', absentismo/total)
sparkDF.select('no_pers', 'dia_matriz', 'absentismo', 'clase_absentismo').where(sparkDF.absentismo > 1).select('no_pers').distinct().show()
sparkDF.select('no_pers', 'dia_matriz', 'absentismo', 'clase_absentismo', 'ausencia_total_rel')\
    .where(sparkDF.ausencia_total_rel > 1).select('no_pers').distinct().show()
# a = input('Continuar?')
print('Generación días: ', datetime.now()-now)
sparkDF.printSchema()
now = datetime.now()
# salvar a Mongo


now = datetime.now()
#sparkDF.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save('C:\\tmp2\\diaria_small.csv')
#sparkDF.rdd.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile("c:\\tmp\\file.csv")

"""
loops = 10
initial = 1
import math
size = math.floor(sparkDF.count()/loops)
sparkDF_tmp = sparkDF

for i in range(loops):
    df = sqlContext.createDataFrame(sparkDF_tmp.head(size), sparkDF_tmp.schema)
"""
# sparkDF.write.csv(path='c:\\tmp\\diaria.csv', mode='overwrite', header=True, sep=',') #, emptyValue='')
sparkDF.toPandas().to_csv('c:\\tmp\\diaria.csv')
"""
    sparkDF_tmp = sparkDF_tmp.subtract(df)

print('\nEscritura a csv completada en: ', datetime.now() - now)

sparkDF.printSchema()
"""

spark.stop()

