# you may need to run this to open excel file
# !python -m pip install xlrd

import pandas as pd

# read the excel file
df = pd.read_excel("縣市人口按單齡-112年5月.xls", skiprows=3, skipfooter=1)

# rename column names by referring to the excel file
df.rename(
    columns={
        "Unnamed: 0": "Region",
        "Unnamed: 1": "Sex",
        "Unnamed: 2": "Total",
        "Unnamed: 123": 100,
    },
    inplace=True,
)

# fill in city names
cities = []
for city in df["Region"][df["Region"].notna()]:
    cities.extend([city] * 3)
df["Region"] = cities

# drop the "合　計" columns
columns_to_drop = df.columns[df.columns.str.contains("合").fillna(False)]
df.drop(columns=columns_to_drop, inplace=True)

# only look at the total rows, ignoring the male/female rows
df_total = df.query('Sex == "計"').copy(deep=True)

# 2.4 將各縣市0~12歲未進入青春期之小學兒童，與13歲以上人口分開相加，不分男女
df_total["under-12"] = df_total.loc[:, 0:12].sum(axis=1)
df_total["over-13"] = df_total.loc[:, 13:100].sum(axis=1)
