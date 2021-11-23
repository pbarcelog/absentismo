import pandas as pd
from datetime import datetime, timedelta

# convertidor para tratar formatos: cambiar a fechas, floats o dejar igual
def myconverter(o):
    if isinstance(o, datetime):
        return "{}-{}-{}".format(o.year, o.month, o.day)
    if isinstance(o, float):
        return round(o, 2)
    return o

def create_row_worker(row, header_original, names_dict, dictionary):
    to_append = []
    for j in range(len(header_original)):  # consultar elemento por elemento de esa fila
        if header_original[j] in names_dict:  # comprobar si es un parámetro que se ha de traducir
            name = names_dict[header_original[j]]  # cambiar el nombre del nombre-header al nombre-diccionario
            conversion = dictionary[name]  # cargar diccionario de ese parámetro
            if row[j] in conversion:  # ver si es un valor a traducir
                to_append.append(conversion[row[j]])  # añadir el valor traducido
            else:
                to_append.append(row[j])  # si no era valor a traducir, añadir el valor original
        else:
            to_append.append(row[j])  # si no había que traducir, añadir el valor original
    to_append = list(map(lambda x: myconverter(x), to_append))  # construir lista que será la nueva row
    # print('\nto_append: ', to_append)
    # print('\nComparar longitudes esta fila con fila headers: ', len(to_append), len(header_original))
    return pd.Series(to_append, index=header_original)  # transformar en serie para añadir al dataframe

def process_row_absence(row, header_absence, names_dict, dictionary):
    to_append = []
    for j in range(len(header_absence)): # consultar elemento por elemento de esa fila
        if header_absence[j] in names_dict: # comprobar si es un parámetro que se ha de traducir
            name = names_dict[header_absence[j]] # cambiar el nombre del nombre-header al nombre-diccionario
            conversion = dictionary[name] # cargar diccionario de ese parámetro
            if row[j] in conversion: # ver si es un valor a traducir
                to_append.append(conversion[row[j]]) # añadir el valor traducido
                #print(row[j])
            else:
                to_append.append(row[j]) # si no era valor a traducir, añadir el valor original
                #print(row[j])
        else:
            to_append.append(row[j]) # si no había que traducir, añadir el valor original
            #print(row[j])
    return to_append

def create_row_absence(row, header_absence, names_dict, dictionary,df_absence):
    to_append = process_row_absence(row, header_absence, names_dict, dictionary) # procesar row
    #print(row, '\n', to_append)
    to_append = [myconverter(x) for x in to_append] # construir lista que será la nueva row
    return pd.Series(to_append, index = header_absence) # transformar en serie para añadir al dataframe
