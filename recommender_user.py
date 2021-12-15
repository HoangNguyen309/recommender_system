import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql import SQLContext

spark = SparkSession.builder.appName('rec').getOrCreate()

recommender_user = spark.read.parquet('user_recs.parquet')