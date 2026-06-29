import yfinance as yf

search = yf.Search(
    "rel",
    max_results=10
)

print(search.quotes)