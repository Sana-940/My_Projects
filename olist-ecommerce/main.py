import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


data_files={
    "customers": "Data/olist_customers_dataset.csv",
    "geolocation": "Data/olist_geolocation_dataset.csv",
    "order_items": "Data/olist_order_items_dataset.csv",
    "order_payments": "Data/olist_order_payments_dataset.csv",
    "order_reviews": "Data/olist_order_reviews_dataset.csv",
    "orders": "Data/olist_orders_dataset.csv",
    "products": "Data/olist_products_dataset.csv",
    "sellers": "Data/olist_sellers_dataset.csv",
    "product_category_name_translation": "Data/product_category_name_translation.csv"
}

df={}

for name,filename in data_files.items():
    df[name] = pd.read_csv(filename)


df["customers"].head()
df["customers"].info()
df["geolocation"].head()
df["geolocation"].info()
df["order_items"].head()
df["order_items"].info()
df["order_payments"].head()
df["order_payments"].info()
df["order_reviews"].head()
df["order_reviews"].info()
df["orders"].head()
df["orders"].info()
df["products"].head()
df["products"].info()
df["sellers"].head()
df["sellers"].info()