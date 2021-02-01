import pandas as pd

df = pd.read_csv("houseelf-earlength-dna-data.csv")


for ind in df.index: 
     print(df['id'][ind], df['earlength'][ind], df['dnaseq'][ind]) 

