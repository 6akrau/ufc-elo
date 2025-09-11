from bs4 import BeautifulSoup
import requests

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

r =session.get("https://www.roster.watch/fighters/israeladesanya.html")

print(r)