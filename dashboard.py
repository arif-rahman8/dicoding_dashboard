import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
sns.set(style="whitegrid")

# Prepare DataFrame

def create_atemp_df_daily(df):
    df['atemp_group'] = pd.cut(
        x=df['atemp'],
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        labels=['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50']
    )
    atemp_df_daily = df.groupby(by="atemp_group", as_index=False).agg({
        "casual": "mean",
        "registered": "mean",
        "cnt": "mean"
    })
    atemp_df_daily.columns = ["atemp_celcius", "casual", "registered", "total"]

    return atemp_df_daily

def create_season_df_daily(df):
    season_df_daily = df.groupby(by="season", as_index=False).agg({
        "casual": "mean",
        "registered": "mean",
        "cnt": "mean"
    })
    season_df_daily.season.replace(1, "Spring", inplace=True)
    season_df_daily.season.replace(2, "Summer", inplace=True)
    season_df_daily.season.replace(3, "Fall", inplace=True)
    season_df_daily.season.replace(4, "Winter", inplace=True)

    return season_df_daily

def create_weathersit_df_daily(df):
    weathersit_df_daily = df.groupby(by="weathersit", as_index=False).agg({
        "casual": "mean",
        "registered": "mean",
        "cnt": "mean"
    })
    weathersit_df_daily.weathersit.replace(1, "Clear", inplace=True)
    weathersit_df_daily.weathersit.replace(2, "Cloudy", inplace=True)
    weathersit_df_daily.weathersit.replace(3, "Light Rain/Snow", inplace=True)
    weathersit_df_daily.weathersit.replace(4, "Heavy Rain/Snow", inplace=True)

    return weathersit_df_daily

def create_user(df):
    user = df.agg({
        "casual": "sum",
        "registered": "sum"
    })

    return user

def create_user_df_hourly(df):
    user_df_hourly = df.groupby(by="hr").agg({
        "casual": "mean",
        "registered": "mean",
        "cnt": "mean"
    }).reset_index()

    return user_df_hourly

def create_weekly_df_daily(df):
    weekly_df_daily = df.groupby(by="weekday").agg({
        "casual": "mean",
        "registered": "mean",
        "cnt": "mean"
    }).reset_index()
    weekly_df_daily.weekday.replace(0, "Sunday", inplace=True)
    weekly_df_daily.weekday.replace(1, "Monday", inplace=True)
    weekly_df_daily.weekday.replace(2, "Tuesday", inplace=True)
    weekly_df_daily.weekday.replace(3, "Wednesday", inplace=True)
    weekly_df_daily.weekday.replace(4, "Thursday", inplace=True)
    weekly_df_daily.weekday.replace(5, "Friday", inplace=True)
    weekly_df_daily.weekday.replace(6, "Saturday", inplace=True)

    return weekly_df_daily

def create_week_merged_df(df):
    holiday_df = df.groupby(by="holiday").agg({
        "casual": "mean",
        "registered": "mean",
    })
    holiday_df_T = holiday_df.transpose().reset_index()
    holiday_df_T.columns = ["status", "normal", "holiday"]
    workingday_df = df.groupby(by="workingday").agg({
        "casual": "mean",
        "registered": "mean",
    })
    workingday_df_T = workingday_df.transpose().reset_index()
    workingday_df_T.columns = ["status", "weekend", "working_day"]
    week_merged_df = pd.merge(
        left=holiday_df_T,
        right=workingday_df_T,
        how="inner",
        left_on="status",
        right_on="status"
    )

    return week_merged_df

def create_monthly_df_daily(df):
    monthly_df_daily = df.groupby(by=["yr", "mnth"]).agg({
        "casual": "sum",
        "registered": "sum",
        "cnt": "sum"
    }).reset_index()
    monthly_df_daily.mnth.replace(1, "January", inplace=True)
    monthly_df_daily.mnth.replace(2, "February", inplace=True)
    monthly_df_daily.mnth.replace(3, "March", inplace=True)
    monthly_df_daily.mnth.replace(4, "April", inplace=True)
    monthly_df_daily.mnth.replace(5, "May", inplace=True)
    monthly_df_daily.mnth.replace(6, "June", inplace=True)
    monthly_df_daily.mnth.replace(7, "July", inplace=True)
    monthly_df_daily.mnth.replace(8, "August", inplace=True)
    monthly_df_daily.mnth.replace(9, "September", inplace=True)
    monthly_df_daily.mnth.replace(10, "October", inplace=True)
    monthly_df_daily.mnth.replace(11, "November", inplace=True)
    monthly_df_daily.mnth.replace(12, "December", inplace=True)
    monthly_df_daily.yr.replace(2011, "11", inplace=True)
    monthly_df_daily.yr.replace(2012, "12", inplace=True)
    monthly_df_daily["period"] = monthly_df_daily[["mnth", "yr"]].astype(str).apply(" '".join, axis=1)

    return monthly_df_daily

def create_monthly_atemp_df(df):
    monthly_atemp_df = df.groupby(by=['yr', 'mnth']).agg({
        "atemp": "mean",
    }).reset_index()
    atemp_unorm = monthly_atemp_df["atemp"].apply(lambda x: x * 50)
    monthly_atemp_df["atemp"] = atemp_unorm
    monthly_atemp_df.mnth.replace(1, "January", inplace=True)
    monthly_atemp_df.mnth.replace(2, "February", inplace=True)
    monthly_atemp_df.mnth.replace(3, "March", inplace=True)
    monthly_atemp_df.mnth.replace(4, "April", inplace=True)
    monthly_atemp_df.mnth.replace(5, "May", inplace=True)
    monthly_atemp_df.mnth.replace(6, "June", inplace=True)
    monthly_atemp_df.mnth.replace(7, "July", inplace=True)
    monthly_atemp_df.mnth.replace(8, "August", inplace=True)
    monthly_atemp_df.mnth.replace(9, "September", inplace=True)
    monthly_atemp_df.mnth.replace(10, "October", inplace=True)
    monthly_atemp_df.mnth.replace(11, "November", inplace=True)
    monthly_atemp_df.mnth.replace(12, "December", inplace=True)
    monthly_atemp_df.yr.replace(2011, "11", inplace=True)
    monthly_atemp_df.yr.replace(2012, "12", inplace=True)
    monthly_atemp_df["period"] = monthly_atemp_df[["mnth", "yr"]].astype(str).apply(" '".join, axis=1)

    return monthly_atemp_df

# Read CSV File
df_hourly = pd.read_csv("df_hourly.csv")

# Filter

datetime_columns = ["dteday", "hr"]
df_hourly.sort_values(by="dteday", inplace=True)
df_hourly.reset_index(inplace=True)

for column in datetime_columns:
    df_hourly[column] = pd.to_datetime(df_hourly[column])

min_date = df_hourly["dteday"].min()
max_date = df_hourly["dteday"].max()

with st.sidebar:
    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = df_hourly[(df_hourly["dteday"] >= str(start_date)) &
                (df_hourly["dteday"] <= str(end_date))]

# Call DataFrame

atemp_df_daily = create_atemp_df_daily(main_df)
season_df_daily = create_season_df_daily(main_df)
weathersit_df_daily = create_weathersit_df_daily(main_df)
user = create_user(main_df)
user_df_hourly = create_user_df_hourly(main_df)
weekly_df_daily = create_weekly_df_daily(main_df)
week_merged_df = create_week_merged_df(main_df)
monthly_df_daily = create_monthly_df_daily(main_df)
monthly_atemp_df = create_monthly_atemp_df(main_df)

# Create Dashboard

st.header("Bike Sharing Rental Analysis Dashboard")

# Question 1
st.subheader('User: Casual vs Registered')

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.pie(
        user,
        autopct='%1.1f%%',
        colors = ["#000080", "#32CD32"],
        radius = 0.9,
        pctdistance= 1.2,
    )
    ax.legend(labels = ["Casual", "Registered"])
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize = (15, 5))
    ax.plot(
        monthly_df_daily["period"],
        monthly_df_daily["registered"],
        marker = ".",
        label = "Registered",
        color = "limegreen"
    )
    ax.plot(
        monthly_df_daily["period"],
        monthly_df_daily["casual"],
        marker = ".",
        label = "Casual",
        color = "navy"
    )
    st.pyplot(fig)

# Question 2
st.subheader('Weather and User Relationship')

plt.figure(figsize = (10, 5))

sns.barplot(
    y = "season",
    x = "cnt",
    data = season_df_daily,
    color = "royalblue"
)
plt.title("Average Number of User by Season", loc = "left", fontsize = 18)
plt.xlabel(None)
plt.ylabel(None)
st.pyplot(fig)

plt.figure(figsize = (10, 5))

sns.barplot(
    y = "weathersit",
    x = "cnt",
    data = weathersit_df_daily.sort_values(by = "cnt", ascending = False),
    color = "royalblue"
)
plt.title("Average Number of User by Weather Situation", loc = "left", fontsize = 18)
plt.xlabel(None)
plt.ylabel(None)
st.pyplot(fig)

plt.figure(figsize = (10, 5))
sns.barplot(
    y = "total",
    x = "atemp_celcius",
    data = atemp_df_daily,
    color = "royalblue"
)
plt.ylabel(None)
plt.xlabel("Temperature in Celcius (°C)", fontsize = 12)
plt.title("Average Number of Biker by Feels-Like Temperature", loc = "left", fontsize = 16)
st.pyplot(fig)

fig , ax = plt.subplots(nrows = 2, ncols = 1, figsize = (15, 10))
sns.barplot(
    x = "period",
    y = "atemp",
    data = monthly_atemp_df,
    color = "royalblue",
    width = 0.5,
    ax = ax[0]
)
ax[0].set_xticklabels(monthly_atemp_df["period"], rotation = 60)
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title("Averge Feels-Like Temperature by Month (2011-2012)", loc = "left", fontsize = 18)

plt.plot(
    monthly_df_daily["period"],
    monthly_df_daily["registered"],
    marker = ".",
    label = "Registered",
    color = "limegreen",
)
plt.plot(
    monthly_df_daily["period"],
    monthly_df_daily["casual"],
    marker = ".",
    label = "Casual",
    color = "navy",
)
ax[1].set_xticklabels(monthly_atemp_df["period"], rotation = 60)
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].legend()
ax[1].set_title("Total Number of Users by Month (2011-2012)", loc = "left", fontsize = 18)

plt.suptitle("Relationship between Feels-Like Temperature and Total Number of User by Month", fontsize = 20)
st.pyplot(fig)


# Question 3
st.subheader('User and Time Relationship')

plt.figure(figsize = (10, 5))
sns.barplot(
    x = "hr",
    y = "registered",
    data = user_df_hourly,
    label = "Registered",
    color = "limegreen"
)
sns.barplot(
    x = "hr",
    y = "casual",
    data = user_df_hourly,
    label = "Casual",
    color = "navy"
)
plt.legend()
plt.xticks(range(0,24))
plt.xlabel("Hours")
plt.title("Average Number of User by Hours", loc = "left", fontsize = 18)
st.pyplot(fig)

plt.figure(figsize = (10, 5))
sns.barplot(
    x = "weekday",
    y = "registered",
    data = weekly_df_daily,
    label = "Registered",
    color = "limegreen",
    width = 0.5
)
sns.barplot(
    x = "weekday",
    y = "casual",
    data = weekly_df_daily,
    label = "Casual",
    color = "navy",
    width = 0.5
)
plt.legend()
plt.title("Average Number of User by Week", loc = "left", fontsize = 18)
st.pyplot(fig)

plt.figure(figsize = (5, 8))
color = ["#000080", "#32CD32"]
sns.barplot(
    x = "status",
    y = "working_day",
    data = week_merged_df,
    palette = color,
    width = 0.5
)
plt.xlabel("Working Day")
plt.ylabel(None)
plt.title("Average Number of User During Working Day", loc = "left", fontsize = 15)
plt.yticks(np.arange(0, 4500, 500))
st.pyplot(fig)

plt.figure(figsize = (5, 8))
sns.barplot(
    x = "status",
    y = "weekend",
    data = week_merged_df,
    palette = color,
    width = 0.5
)
plt.xlabel("Holiday")
plt.ylabel(None)
plt.title("Average Number of User During Weekend", loc = "left", fontsize = 15)
plt.yticks(np.arange(0, 4500, 500))
st.pyplot(fig)

plt.figure(figsize = (5, 8))
sns.barplot(
    x = "status",
    y = "holiday",
    data = week_merged_df,
    palette = color,
    width = 0.5
)
plt.xlabel("Holiday")
plt.ylabel(None)
plt.title("Average Number of User During Holiday", loc = "left", fontsize = 15)
plt.yticks(np.arange(0, 4500, 500))
st.pyplot(fig)
