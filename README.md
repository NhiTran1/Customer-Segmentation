## Final Project: Customer Segmentation 
# Abstract 
Summary Case

Objective: Segment customers from the Adventure Works Sales dataset by applying RFM and K-means clustering then offering tailored solutions for each customer group to help businesses solve problems such as increasing profit, lengthen Customer Lifetime Value and reduce churn rate of customer by marketing efficiency with the right segment (know how to treat customer with specific characteristic).

Analytic Technique:
- Descriptive Analysis
- Graph analysis
- Segment Analysis

Expected Outcome:
- Get business insight about customer behavior such as who willing to spend on their shopping activities, churn rate and CLV of customer, … 
- Customer segmentation based on specific characteristic.
- Recommendation some marketing plan for this business
## Data Understanding
- Data of Adventure Works Sales from 01 July 2017 to 15 June 2020
- Source Date: Adventure Works Sales (we just use Sales data sheet in this project to apply unsupervised machine learning with K-Means and RFM models)
- Sales Order Line Key: Order ID
- Customer Key: Customer code assigned to each customer
- Product Key: Product ID
- Order Date Key: Order Date ID
- Sales Territory Key: Territory ID
- Order Quantity:  The quantities of each product
- Unit Price: Product price per unit
- Sales Amount: Total Revenue of each customer
## Initial Setup: 
- Python in Google Colab
- Packages libraries:
- Install libraries:
!pip install xgboost
!pip install imbalanced-learn
!pip install category_encoders
+ Import libraries to math and visualization:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import seaborn as sns
from datetime import timedelta
import plotly.express as px
from matplotlib import colors
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import boxcox
import plotly.graph_objs as go 
import plotly.offline as pyoff 
+ Import libraries to clustering and Data Standardization:
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
import category_encoders as ce
from imblearn.over_sampling import SMOTE, SVMSMOTE
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer

## Chapter 1: Theoretical Basis
This chapter will present and explain the theories related to our project, such as Customer Behavior, RFM, Machine Learning, K-Means, CLV, …
## Chapter 2: Data Preparation
We concentrate on preparing input data for Machine Learning with steps like EDA and Pre - processing.
## Chapter 3: Customer segmentation with Machine Learning method
Show the model and how we combine Machine Learning with RFM on the Adventure Works dataset. Besides, this chapter presents clustering preparation steps, such as determining the number of clusters by Elbow and Silhouette, the clustering results, comparing and analyzing clusters’ features, labeling clusters and discussing issues that related to business and management from the customer segmentation result.
## Chapter 4: Data Visualization and Analyzing CLV Prediction 
Presents how we visualize the actual result data from Chapter 3 and analyze CLV prediction. It also indicates our discussion about issues from customer groups and business.
## Chapter 5: Conclusion and Future Works 
Summarize the issues discussed in the previous chapter and draw conclusions and suggest future development directions for the business.


