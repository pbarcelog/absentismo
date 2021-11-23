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
output_data = 'test.aggregated'
num_cpus = '3'

now = datetime.now()
now_inicial = now
# cargaremos de mongo porque carga el schema bien, no como el csv
spark = SparkSession.builder.master("local[" + num_cpus + "]").appName("ml") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/" + input_data) \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/" + output_data) \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.executor.memory", "3g") \
    .config("spark.driver.memory", "6g") \
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

sparkDF.printSchema()

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


KEY = ['ausencia_total_abs', 'ausencia_total_rel', 'categoria_profesional',
       'centro_de_coste', 'clase_de_contrato', 'clase_de_instituto_de_ensenanz', 'clase_registro_parentesco',
       'clave_de_sexo', 'codpostal', 'descripcion', 'desde_real', 'dia_matriz', 'dias_rango',
       'division_de_personal', 'estado_civil', 'feantig', 'fecha_nac', 'h_sem', 'hasta_real',
       'identificacion_de_contratos_se', 'idobj', 'motivo_de_la_medida', 'nacionalidad', 'no_pers', 'numero_de_orden',
       'porcentaje_de_discapacidad', 'proyectos', 'year'] #'semana_santa',

dropKEY = KEY + ['absentismo']
dropKEY.remove('year_real')

sparkDF = sparkDF.withColumn("repeat", F.expr("split(repeat(',', dias_rango-1), ',')"))\
                .select(F.col("*"), F.posexplode(F.col("repeat")))\
                .withColumn("dia_matriz", F.expr("date_add(desde_real, pos)"))\
                .drop("repeat", "col", "pos") \
                .withColumn('absentismo', F.when(F.col('dia_matriz').between(F.col('inicial_real'), F.col('final_real')),
                                                 F.lit(1)).otherwise(F.lit(0)))\
                .dropDuplicates(dropKEY) \
                .withColumn('absentismo_rel', F.col('absentismo') / F.col('proyectos')) \
                .withColumn('clase_absentismo', F.when(F.col('absentismo') == 1, F.col('clase_absentismo')).otherwise(F.lit('')))\
                .groupBy(KEY).agg(F.sum('absentismo').alias('absentismo'),
                                  F.collect_set('clase_absentismo').alias('set_clase_absentismo')) \
                .withColumn('clase_absentismo', F.array_join(F.col('set_clase_absentismo'), ',')).drop('set_clase_absentismo')\
                .withColumn('clase_absentismo', F.when(F.col('clase_absentismo') == '', F.lit('no_absentismo'))
                            .otherwise(F.col('clase_absentismo')))\
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
                                            .otherwise(F.lit(0))) \
                .withColumn('edad', F.datediff(F.col('dia_matriz'), F.col('fecha_nac'))) \
                .withColumn('edad', when(F.col('edad') < 5840, F.lit(5840)).otherwise(F.col('edad'))) \
                .withColumn('antiguedad', F.datediff(F.col('dia_matriz'), F.col('feantig')))

sparkDF_agg = sparkDF.groupBy('dia_matriz', 'semana_santa')\
                     .agg(F.count('absentismo').alias('employees'),
                          F.avg('absentismo').alias('absentismo'),
                          F.avg('dias_rango').alias('dias_rango'),
                          F.avg('edad').alias('edad'),
                          F.avg('antiguedad').alias('antiguedad'),
                          F.avg('h_sem').alias('h_sem'),
                          F.avg('idobj').alias('idobj'),
                          F.avg('porcentaje_de_discapacidad').alias('porcentaje_de_discapacidad'),
                          F.avg('proyectos').alias('proyectos')) \
                    .withColumn('year_real', F.year(F.col('dia_matriz')))\
                    .withColumn("semana_numero", F.weekofyear(F.col("dia_matriz"))) \
                    .withColumn("dia_semana", F.date_format(F.col("dia_matriz"), "E")) \
                    .withColumn("dia_num", F.dayofweek(F.col("dia_matriz")))

    # sparkDF.select('no_pers', 'dia_matriz', 'absentismo', 'clase_absentismo').show()

print('Generación días: ', datetime.now()-now)
sparkDF_agg.printSchema()
sparkDF_agg.sort('dia_matriz').show()
now = datetime.now()
# salvar a Mongo
sparkDF_agg.write.format("mongo").mode("overwrite").save()
print('Info salvada en mongo: ', datetime.now()-now)

now = datetime.now()
#sparkDF.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save('C:\\tmp2\\diaria_small.csv')
#sparkDF.rdd.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile("c:\\tmp\\file.csv")
sparkDF_agg.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save('C:\\tmp2\\diaria_agg.csv')

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
"""
    sparkDF_tmp = sparkDF_tmp.subtract(df)

print('\nEscritura a csv completada en: ', datetime.now() - now)

sparkDF.printSchema()
"""

spark.stop()


"""
# celda para hacer pruebas: seleccionamos unas pocas filas de la agregación
sparkDF_agg.createOrReplaceTempView("datos_trabajador")
row = spark.sql('SELECT * FROM datos_trabajador AS a WHERE a.no_pers = 1284 and \
a.centro_de_coste = \'055 Madrid Comercial\'')
print('Ha tardado hacer sql: ', datetime.now()-now)
row.show()

# o bien cogemos varias filas de los datos sin agregar
sparkDF.createOrReplaceTempView("datos_trabajador")
row = spark.sql('SELECT * FROM datos_trabajador AS a WHERE a.no_pers = 1284 and \
a.centro_de_coste = \'055 Madrid Comercial\'')
row.show()
row.printSchema()

df_exploded = newDF.withColumn("tmp", arrays_zip("dia", "absence"))\
    .withColumn("tmp", explode("tmp"))\
    .select(columns)
df_exploded.show()

# pasar la lista de días a nuevas filas con el explode

columns = sparkDF.columns
columns.append('tmp.dia')
columns.append('tmp.absence')
df_exploded = newDF.withColumn("tmp", F.arrays_zip("dia", "absence"))\
    .withColumn("tmp", F.explode("tmp"))\
    .select(columns)
df_exploded.show()

"""
