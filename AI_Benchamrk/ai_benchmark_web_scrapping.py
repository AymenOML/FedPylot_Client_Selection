# # Python scraping with pandas
# import pandas as pd

# tables = pd.read_html("https://ai-benchmark.com/ranking_processors.html")
# df = tables[0]
# df.to_csv("ai_benchmark_data.csv", index=False)

import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the page containing the benchmark table
url = "https://ai-benchmark.com/ranking_processors.html"

# Step 1: Get the HTML
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Step 2: Extract all tables (only one of them contains the main SoC benchmark)
tables = pd.read_html(str(soup))

# Step 3: Find the right one — usually the largest one
largest_table = max(tables, key=lambda t: t.shape[0])

# Step 4: Save it to CSV
largest_table.to_csv("ai_benchmark_data.csv", index=False)

print("✅ Benchmark data saved to 'ai_benchmark_data.csv'")
print(largest_table)

