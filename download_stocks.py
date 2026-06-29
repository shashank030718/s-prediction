
import pandas as pd
url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

df = pd.read_csv(url)

print(df.head())

df.to_csv(
    "nse_stocks.csv",
    index=False
)

print(
    "Downloaded successfully!"
)



import pandas as pd

df = pd.read_csv(
    "nse_stocks.csv"
)

print(df.head())
print(df.columns)