import streamlit as st
from streamlit_gsheets import GSheetsConnection
import datetime as dt
import pandas as pd
import time
import plotly.express as px
import pickle
from sklearn.preprocessing import StandardScaler

kmeans_model = pickle.load(open('k_means_model.pkl', 'rb'))

conn = st.connection("gsheets", type=GSheetsConnection)
# functions

# fetch data
def fetch_data(): 
    return conn.read(worksheet="TransactionData", usecols=list(range(8)), ttl=5)

# data preprocessing
def preprocess_data(df):
    # drop duplicates    
    df = df.drop_duplicates()
    
    #  drop missing values
    df = df.dropna()
    
    # get retail order
    retail_order  = df[df['Quantity'] > 0]
    
    # get real order
    retail_order = retail_order[retail_order['StockCode'] != "M"]
    
    # fill 0 with mean
    retail_order['UnitPrice'] = retail_order['UnitPrice'].replace(0, retail_order['UnitPrice'].mean())
    
    # create total
    retail_order['Total'] = retail_order['Quantity'] * retail_order['UnitPrice']
    
    # Change InvoiceDate to Datetime datatpye
    retail_order['InvoiceDate'] = pd.to_datetime(retail_order['InvoiceDate'])
    
    # split InvoiceDate
    retail_order['Date']= retail_order['InvoiceDate'].dt.date
    retail_order['Year']= retail_order['InvoiceDate'].dt.year
    retail_order['Month'] = retail_order['InvoiceDate'].dt.month_name()
    retail_order['Day'] = retail_order['InvoiceDate'].dt.day_name()
    retail_order['Hour'] = retail_order['InvoiceDate'].dt.hour
    
    # drop InvoiceDate
    # retail_order = retail_order.drop("InvoiceDate", axis=1)
    
    return retail_order

# rfm 
def calculate_rfm(retail_order):
    recent= retail_order["InvoiceDate"].max()
    
    # calculate
    rfm = retail_order.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (recent - x.min()).days, # Recency
    'InvoiceNo': 'count',      # Frequency
    'Total': lambda x: x.sum()}).reset_index()
    
    # rename columns
    rfm.rename(columns = {
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'Total': 'MonetaryValue'
    }, inplace = True)
    
    # drop CustomerID
    rfm = rfm.drop("CustomerID", axis=1)
    
    Q1 = rfm.MonetaryValue.quantile(0.05)
    Q3 = rfm.MonetaryValue.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.MonetaryValue >= Q1 - 1.5*IQR) & (rfm.MonetaryValue <= Q3 + 1.5*IQR)]

    Q1 = rfm.Recency.quantile(0.05)
    Q3 = rfm.Recency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]

    Q1 = rfm.Frequency.quantile(0.05)
    Q3 = rfm.Frequency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]
    
    return rfm

def preprocess_for_predict(rfm):
    rfm_df = rfm[["Recency", "Frequency", "MonetaryValue"]]
    scaler = StandardScaler()

    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df)
    rfm_df_scaled.columns = ["Recency", "Frequency", "MonetaryValue"]
    return rfm_df_scaled
    
def predict(data):
    rfm = calculate_rfm(data)
    rfm = preprocess_for_predict(rfm)
    result_df = kmeans_model.predict(rfm)
    # print("\n\n",result_df[0], "\n\n")
    # print("\n\n",rfm, "\n\n")
    label = ""
    if result_df[0] == 0: label = "Customers Needing Attention"
    elif result_df[0] == 1: label = "Recent Customers"
    else: label = "Can't Loose"
    
    data["Cluster"] = label
    return data

# merged with existed dataset
def merge_data(old, new):
    return pd.concat([old, new], ignore_index=True)

# export to excel file 
def to_xlsx(df, filename):
    df.to_excel(filename, index=False)

# get existed dataset
def get_existed_data():
    return pd.read_csv('sample_retail.csv')

# Dashboard
st.title("DASHBOARD")
old = get_existed_data()
data = merge_data(get_existed_data(), fetch_data())
data = data.drop("Unnamed: 0", axis=1)
new = preprocess_data(fetch_data())
# data = merge_data(old, new)

country_list = ['United Kingdom', 'France', 'Australia', 'Netherlands', 'Germany', 'Norway',
 'EIRE', 'Switzerland', 'Spain', 'Poland', 'Portugal', 'Italy', 'Belgium',
 'Lithuania', 'Japan', 'Iceland', 'Channel Islands', 'Denmark', 'Cyprus',
 'Sweden', 'Finland', 'Austria', 'Greece', 'Singapore', 'Lebanon',
 'United Arab Emirates', 'Israel', 'Saudi Arabia', 'Czech Republic', 'Canada',
 'Unspecified', 'Brazil', 'USA', 'European Community', 'Bahrain', 'Malta', 'RSA']

st.sidebar.header("Filter")

country = st.sidebar.multiselect(
    "Select the Country",
    options=country_list,
    default=country_list,
)

#  QUERY
df_data_selection = data.query(
    "Country == @country"
)

df_old_selection = old.query(
    "Country == @country"
)

df_new_selection = new.query(
    "Country == @country"
)

# CHARTS
sales_by_product = ( 
        df_data_selection.groupby(by=['Description']).sum()[["Total"]].head(10).sort_values(by="Total")
    )

sales_by_month = (
    df_data_selection.groupby(by=['Month']).sum()[["Total"]].head(10).sort_values("Total")
)

sales_by_hour = (
    df_data_selection.groupby(by=['Hour'])[["Total"]].sum().head(10)
)

placeholder = st.empty()
while True:
    new = preprocess_data(fetch_data())
    data = merge_data(old, new)
    

    with placeholder.container():
        
        st.markdown("---")
        total_sales, total_invoices, total_customers = st.columns(3) 
        total_sales.metric(label="Total Sales (USD): ", value=round((df_data_selection["Total"].sum())), delta=round((df_data_selection["Total"].sum() - df_old_selection["Total"].sum())))
        total_invoices.metric(label="Total Invoices: ", value=(len(df_data_selection["InvoiceNo"].unique())), delta=(len(df_data_selection["InvoiceNo"].unique()) - len(df_old_selection["InvoiceNo"].unique())))
        total_customers.metric(label="Total Customers: ", value=(len(df_data_selection["CustomerID"].unique())), delta=(len(df_data_selection["CustomerID"].unique()) - len(df_old_selection["CustomerID"].unique())))
        st.markdown("---")
        
        fig_product_sales = px.bar(
            sales_by_product,
            x="Total",
            y=sales_by_product.index,
            orientation='h',
            title="<b>Sales by Product</b>",
            color_discrete_sequence=["#0083B8"] * len(sales_by_product),
            template="plotly_white",
            
        )
        st.plotly_chart(fig_product_sales) 
        st.markdown("---")

        fig_month_sales = px.bar(
            sales_by_month,
            x="Total",
            y=sales_by_month.index,
            orientation='h',
            title="<b>Sales by Month</b>",
            color_discrete_sequence=["#0083B8"] * len(sales_by_month),
            template="plotly_white",
            
        )
        st.plotly_chart(fig_month_sales) 
        st.markdown("---")
        
        fig_hour_sales = px.line(
            sales_by_hour,
            x=sales_by_hour.index,
            y="Total",
            title="<b>Sales by Hour</b>",
            # template="plotly_white",
            
        )
        st.plotly_chart(fig_hour_sales) 
        st.markdown("---")
        
        st.text("Clustering")
        st.dataframe(predict(new)[["InvoiceNo", "InvoiceDate", "CustomerID", "Description", "Total", "Cluster"]])
        
        
        time.sleep(5)
      
        


# new = fetch_data()
# data = merge_data(old, new)



# print(data["Total"].count())


# st.dataframe(fetch_data())