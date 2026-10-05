# Real World Data Sets

## Assignment

Choose a real-world CSV file and make it available in data/raw/.

## Option 1: Choose One of These (Included)

### A) FiveThirtyEight airline safety dataset

Good clean, smaller dataset; good for exploration.

- Source: <https://github.com/fivethirtyeight/data/blob/master/airline-safety/README.md>
- CSV: <https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv>

### B) NYC taxi trips

Good for exploration in this module.

Also enables later regression analysis.

Once familiar with the data, it enables predicting
fare/total/tip from distance, passengers, time, borough.

- Source: <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
- File: <https://forum.starrocks.io/t/loading-and-querying-nyc-yellow-or-green-taxi-data-parquet-format-with-starrocks/188>
-

### C) California housing / housing prices

Good for exploration in this module.

Also enables later regression analysis.

Once familiar with the data, it enables predicting
median house value from income, rooms, population, location features.

- Source: <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html?utm_source=chatgpt.com>
- Download: <https://figshare.com/articles/dataset/cal_housing_tgz/3829992>

### D) Ames housing dataset

A bit richer than California housing data set.
Needs a bit more data cleanup (good practice).

Also enables later regression analysis.

Once familiar with the data, it enables predicting
sale price from a variety of features/columns.

- Source: <https://www.openintro.org/book/statdata/>
- Download: <https://github.com/wblakecannon/ames/blob/master/data/housing.csv>

### E) Our World in Data CO2 / energy dataset  (Partial Provided)

Great real-world context.
Can explore data subset by country/year.
The full CSV file is 14 MB.

Once familiar with the data, it enables predicting
emissions or energy use from economic/population variables.

A subset with United States, China, India, Germany, United Kingdom,
France, Japan, Brazil, Canada, World for years 1990-present
has been **used for the example**.

- GitHub: <https://github.com/owid/co2-data>
- Explorer: <https://ourworldindata.org/explorers/co2>

## Option 2: Choose Another CSV Real-World Dataset

Possible places to look:

- <https://github.com/awesomedata/awesome-public-datasets>
- <https://www.dataquest.io/blog/free-datasets-for-projects/>

## Option 3: Explore your Own CSV Dataset

Use your own data from work or home and explore possible relationships.

Dataset must:

- have some numeric features
- have some categorical features
- not be TOO messy (cleaning is hard)
