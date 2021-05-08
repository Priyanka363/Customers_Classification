#importing libraries for wrangling of data and plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

#reading csv file using pandas
orders=pd.read_csv("Orders - Analysis Task.csv")

#vsualizing top 5 rows orders dataset
orders.head()

orders.describe()

orders.info()

#checking number of attributes where net qunatity is less than 0
orders[orders.net_quantity < 0].shape[0]

#number of orders less than equal to 0
orders[orders.ordered_item_quantity <= 0].shape[0]

#removing all the orders where qyantity was less than equal to 0
orders=orders[orders['ordered_item_quantity']>0]

#Finding the customers who ordered multiple products
def chk_len_col(col):
  if col>0:
    return 1
  else:
    return 0

#
def quan_aggre(frame,cols):
  aggre_frm=(frame.groupby(cols).ordered_item_quantity.count().reset_index())
  aggre_frm['prod_ordr']=(aggre_frm.ordered_item_quantity.apply(chk_len_col))
  final_frame=(aggre_frm.groupby(cols[0]).prod_ordr.sum().reset_index())
  return final_frame

customers=quan_aggre(orders,['customer_id','product_type'])

#Calculating avg return rate

cus_order=(orders.groupby(['customer_id','order_id']).ordered_item_quantity.sum().reset_index())

cus_return=(orders.groupby(['customer_id','order_id']).returned_item_quantity.sum().reset_index())

#merging order and return col
sum_order_return=pd.merge(cus_order,cus_return)

sum_order_return['avg_return_rate'] = (-1 * sum_order_return['returned_item_quantity']/sum_order_return['ordered_item_quantity'])

sum_order_return.head()

cus_return_rate=(sum_order_return.groupby('customer_id').avg_return_rate.mean().reset_index())

cus_return_rate

return_rate=pd.DataFrame(cus_return_rate['avg_return_rate'].value_counts().reset_index())

return_rate

return_rate.rename(columns= {'index':'avg_return_rate' , 'avg_return_rate':'count of people with that return rate'}, inplace=True )
return_rate.sort_values(by='avg_return_rate')

customers=pd.merge(customers,cus_return_rate,on='customer_id')

#Spendings

cus_spend=(orders.groupby('customer_id').total_sales.sum().reset_index())

cus_spend.rename(columns={'total_sales':'total_spendings'},inplace=True)

customers=pd.merge(customers,cus_spend,on='customer_id')

customers.shape[0]

customers.head()

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

#Normalizing the features

#function that log normalizes the feature
def log_trans(frame,col):
  dataframe["normalized_" + col] = np.log1p(frame[col])
  return dataframe["normalized_" + col]

#For the products ordered
log_trans(customers,"prod_ordr")

#For average return
log_trans(customers, 'avg_return_rate')

#For total spendings
log_trans(customers,'total_spendings')

customers

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

fig = make_subplots(rows=3, cols=3,
                   subplot_titles=("Products Ordered", 
                                   "Average Return Rate", 
                                   "Total Spending"))

fig.append_trace(go.Histogram(x=customers.prod_ordr),
                 row=1, col=1)

fig.append_trace(go.Histogram(x=customers.avg_return_rate),
                 row=1, col=2)

fig.append_trace(go.Histogram(x=customers.total_spendings),
                 row=1, col=3)

fig.update_layout(height=800, width=800,
                  title_text="Distribution of the Features")

fig.show()

customers.head()

#droping customer id column
customers.drop(['customer_id'],axis=1,inplace=True)

#K-means Model
model=KMeans(init='k-means++',max_iter=500,random_state=42)
model.fit(customers.iloc[:,3:])

#function for finding inertia values for some values of k
def list_K(k_max,frame):
  k_vals=list(range(1,k_max+1))
  sse=[]  #sse is sum of squared errors

  for k in k_vals:
    model=KMeans(n_clusters=k, init='k-means++', max_iter=500 , random_state=42)
    model.fit(frame)
    sse.append(model.inertia_)

  return sse

sse_k=list_K(15,customers.iloc[:,3:])

sse_distances=pd.DataFrame({'No. of clusters':list(range(1,16)), 'SSE':sse_k})

fig=go.Figure(go.Scatter(x=sse_distances['No. of clusters'], y=sse_distances['SSE']))
fig.update_layout(xaxis=dict(tickmode='linear',tick0=1,dtick=1))

#Taking value of number of clusters to be 4
new_model=KMeans(n_clusters=4, init='k-means++', max_iter=500, random_state=42)
new_model.fit_predict(customers.iloc[:,3:])

cluster_centers = new_model.cluster_centers_   #normalized cluster centers
original_centers = np.expm1(cluster_centers)   #original cluster center before normalizing
add_points = np.append(original_centers, cluster_centers, axis=1)
add_points

add_points = np.append(add_points, [[0], [1], [2], [3]], axis=1)
customers["clusters"] = new_model.labels_

add_points

#centers dataframe
centers_df = pd.DataFrame(data=add_points, columns=["products_ordered",
                                                    "average_return_rate",
                                                    "total_spending",
                                                    "normalized_products_ordered",
                                                    "normalized_average_return_rate",
                                                    "normalized_total_spending",
                                                    "clusters"])
centers_df.head()

centers_df["clusters"] = centers_df["clusters"].astype("int")

centers_df.head()

customers["is_center"] = 0  #is not a center
centers_df["is_center"] = 1  #is a center

customers = customers.append(centers_df, ignore_index=True)

customers["cluster_name"] = customers["clusters"].astype(str)

#plotting customer groups centers in 3d scatter plot
fig = px.scatter_3d(customers,
                    x="normalized_products_ordered",
                    y="normalized_average_return_rate",
                    z="normalized_total_spending",
                    color='cluster_name',
                    hover_data=["products_ordered",
                                "average_return_rate",
                                "total_spending"],
                    category_orders = {"cluster_name": 
                                       ["0", "1", "2", "3"]},
                    symbol = "is_center")

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()

cardinality_df = pd.DataFrame(
    customers.cluster_name.value_counts().reset_index())

cardinality_df.rename(columns={"index": "Customer Groups","cluster_name": "Number of customers"},
                              inplace=True)

#plotting a bar graph for customer groups
fig = px.bar(cardinality_df, x="Customer Groups", 
             y="Number of customers",
             color = "Customer Groups",
             category_orders = {"Customer Groups": ["0", "1", "2", "3"]})

fig.update_layout(xaxis = dict(tickmode = 'linear', tick0 = 1, dtick = 1), 
                  yaxis = dict(tickmode = 'linear', tick0 = 1000, dtick = 1000))

fig.show()