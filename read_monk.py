#importiamo la libreria pandas per la gestione del dataFrame
import pandas as pd

#Definiamo il percorsoo del file .train
file_path = "data/monks-1.train"

#Definiamo i nomi delle colonne del dataset
columns = ["class", "A1", "A2", "A3", "A4", "A5", "A6", "id"]

#Leggiamo il file in un dataFrame pandas
#delim_whitespace indica i che ii campi sono separati da spazi o tab
#header=None indica che il file non ha intestazioni
#names=columns assegna i nomi alle colonne
df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=columns)

#Stampiamo le prime 5 righe del DataFrame per controllare che la lettura sia corretta
print(df.head())

#Apriamo il file direttamente per controllare il numero di righe presenti
with open("data/monks-1.train") as f:
    lines = f.readlines() #leggiamo tutte le righe in una lista

#Stampiamo il numero totale di righe del file
print(len(lines))
