import pandas as pd

df = pd.read_csv('sidebysidev1.csv')
# print(df.index,'\n\n', df.columns)
for idx in df.index:
    df.at[idx, 'Obolo'] = (df.loc[idx]['Obolo']).strip()
    df.at[idx, 'English'] = (df.loc[idx]['English']).strip()
df.to_csv('v2.csv', index=False)