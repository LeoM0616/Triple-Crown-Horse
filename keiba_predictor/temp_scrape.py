import requests
from bs4 import BeautifulSoup
import pandas as pd
import io

url = "https://db.netkeiba.com/race/202409020411/"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
response = requests.get(url, headers=headers)
response.encoding = 'euc-jp'
dfs = pd.read_html(io.StringIO(response.text))
print("Columns for JRA race:")
print(list(dfs[0].columns))

url_regional = "https://db.netkeiba.com/race/202344103111/"
response_regional = requests.get(url_regional, headers=headers)
response_regional.encoding = 'euc-jp'
dfs_regional = pd.read_html(io.StringIO(response_regional.text))
print("\nColumns for Regional race:")
print(list(dfs_regional[0].columns))
