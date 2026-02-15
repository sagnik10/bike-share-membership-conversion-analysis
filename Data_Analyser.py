import os
import glob
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

BASE=r"C:\Users\nwp\Downloads\Use_Case_1"
OUT=os.path.join(BASE,"output")
GEO=os.path.join(BASE,"geo")

os.makedirs(OUT,exist_ok=True)
os.makedirs(GEO,exist_ok=True)

GEOJSON=os.path.join(GEO,"chicago.geojson")

if not os.path.exists(GEOJSON):
    url="https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/master/data/chicago-community-areas.geojson"
    open(GEOJSON,"wb").write(requests.get(url).content)

clean=os.path.join(OUT,"cleaned_data.csv")

if os.path.exists(clean):
    os.remove(clean)

files=[f for f in glob.glob(os.path.join(BASE,"**","*.csv"),recursive=True) if OUT not in f]

CHUNK=50000
MAX_DURATION=180
MAX_SAMPLE=300000
MAX_CLUSTER_POINTS=50000
MAX_ML_POINTS=400000
MAX_ANOMALY_POINTS=200000

plot_rows=[]
geo_points=[]
ml_rows=[]
corr_rows=[]

rows=0
START=time.time()

total_bytes=sum(os.path.getsize(f) for f in files)

with Progress("[progress.description]{task.description}",BarColumn(),"[progress.percentage]{task.percentage:>3.1f}%",TimeElapsedColumn(),TimeRemainingColumn()) as progress:

    task=progress.add_task("Processing Elite Cyclistic Dataset",total=total_bytes)

    for file in files:

        for chunk in pd.read_csv(file,chunksize=CHUNK,low_memory=False):

            progress.update(task,advance=chunk.memory_usage(deep=True).sum())

            cols={c.lower():c for c in chunk.columns}

            if "started_at" in cols:
                chunk["started_at"]=pd.to_datetime(chunk[cols["started_at"]],errors="coerce")
                if "ended_at" in cols:
                    chunk["ended_at"]=pd.to_datetime(chunk[cols["ended_at"]],errors="coerce")
                    chunk["ride_length"]=(chunk["ended_at"]-chunk["started_at"]).dt.total_seconds()/60
                if "start_lat" in cols:
                    chunk["start_lat"]=chunk[cols["start_lat"]]
                if "start_lng" in cols:
                    chunk["start_lng"]=chunk[cols["start_lng"]]

            elif "starttime" in cols:
                chunk["started_at"]=pd.to_datetime(chunk[cols["starttime"]],errors="coerce")
                if "tripduration" in cols:
                    chunk["ride_length"]=chunk[cols["tripduration"]]/60
                if "latitude" in cols:
                    chunk["start_lat"]=chunk[cols["latitude"]]
                if "longitude" in cols:
                    chunk["start_lng"]=chunk[cols["longitude"]]

            else:
                continue

            chunk=chunk.drop_duplicates()
            chunk=chunk[chunk.isnull().sum(axis=1)<=4]
            chunk=chunk[(chunk["ride_length"]>0)&(chunk["ride_length"]<1440)]

            chunk["hour"]=chunk["started_at"].dt.hour
            chunk["month"]=chunk["started_at"].dt.month

            chunk.to_csv(clean,mode="a",header=not os.path.exists(clean),index=False)

            plot_rows.append(chunk[["ride_length","hour","month"]])

            if "start_lat" in chunk.columns and "start_lng" in chunk.columns:
                geo_points.append(chunk[["start_lat","start_lng","ride_length"]].dropna())

            ml_rows.append(chunk[["hour","month","ride_length"]])
            corr_rows.append(chunk[["ride_length","hour","month"]])

            rows+=len(chunk)

plot_df=pd.concat(plot_rows,ignore_index=True)
plot_df=plot_df[(plot_df.ride_length>0)&(plot_df.ride_length<MAX_DURATION)]

if len(plot_df)>MAX_SAMPLE:
    plot_df=plot_df.sample(MAX_SAMPLE,random_state=42)

duration_all=plot_df["ride_length"].values
hour_df=plot_df[["hour","ride_length"]].rename(columns={"ride_length":"duration"})
month_df=plot_df[["month","ride_length"]].rename(columns={"ride_length":"duration"})
heat_df=plot_df.rename(columns={"ride_length":"duration"})

plt.style.use("dark_background")

hist_img=os.path.join(OUT,"hist.png")
plt.figure(figsize=(16,8))
plt.hist(duration_all,bins=120,log=True,color="#00FFFF",edgecolor="white")
plt.xlabel("Ride Duration (minutes)")
plt.ylabel("Ride Count (log scale)")
plt.title("Ride Duration Distribution")
plt.savefig(hist_img,dpi=300,facecolor="black",bbox_inches="tight")
plt.close()

hour_img=os.path.join(OUT,"hour.png")
hour_avg=hour_df.groupby("hour").mean()
plt.figure(figsize=(16,8))
plt.plot(hour_avg.index,hour_avg.duration,color="#00FFFF",linewidth=3)
plt.xlabel("Hour of Day")
plt.ylabel("Average Ride Duration (minutes)")
plt.title("Hourly Performance Curve")
plt.savefig(hour_img,dpi=300,facecolor="black",bbox_inches="tight")
plt.close()

month_img=os.path.join(OUT,"month.png")
month_avg=month_df.groupby("month").mean()
plt.figure(figsize=(16,8))
plt.plot(month_avg.index,month_avg.duration,color="#00FFFF",linewidth=3)
plt.xlabel("Month")
plt.ylabel("Average Ride Duration (minutes)")
plt.title("Seasonal Performance Curve")
plt.savefig(month_img,dpi=300,facecolor="black",bbox_inches="tight")
plt.close()

heat_img=os.path.join(OUT,"heat.png")
pivot=heat_df.pivot_table(values="duration",index="hour",columns="month",aggfunc="mean")
plt.figure(figsize=(16,10))
sns.heatmap(pivot,cmap="plasma")
plt.xlabel("Month")
plt.ylabel("Hour")
plt.title("Performance Heatmap")
plt.savefig(heat_img,dpi=300,facecolor="black",bbox_inches="tight")
plt.close()

geo_img=None

if geo_points:

    geo_df=pd.concat(geo_points,ignore_index=True)
    geo_df=geo_df.dropna()

    if len(geo_df)>MAX_CLUSTER_POINTS:
        geo_sample=geo_df.sample(MAX_CLUSTER_POINTS,random_state=42)
    else:
        geo_sample=geo_df

    coords=geo_sample[["start_lat","start_lng"]].values

    cluster=DBSCAN(eps=0.01,min_samples=30,n_jobs=-1).fit(coords)

    geo_sample["cluster"]=cluster.labels_

    geometry=[Point(xy) for xy in zip(geo_sample.start_lng,geo_sample.start_lat)]

    gdf=gpd.GeoDataFrame(geo_sample,geometry=geometry,crs="EPSG:4326")

    chicago=gpd.read_file(GEOJSON)

    gdf=gdf.to_crs(epsg=3857)
    chicago=chicago.to_crs(epsg=3857)

    geo_img=os.path.join(OUT,"geo.png")

    fig,ax=plt.subplots(figsize=(16,12))
    chicago.plot(ax=ax,alpha=0.3)
    gdf.plot(ax=ax,column="cluster",cmap="plasma",markersize=3)
    ax.set_title("Geospatial Ride Cluster Intelligence")
    ax.set_xlabel("Longitude (meters)")
    ax.set_ylabel("Latitude (meters)")
    ctx.add_basemap(ax)
    plt.savefig(geo_img,dpi=300,facecolor="black",bbox_inches="tight")
    plt.close()

ml_df=pd.concat(ml_rows,ignore_index=True)

if len(ml_df)>MAX_ML_POINTS:
    ml_df=ml_df.sample(MAX_ML_POINTS,random_state=42)

X=ml_df[["hour","month"]]
y=ml_df["ride_length"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

rf=RandomForestRegressor(n_estimators=60,max_depth=12,n_jobs=-1)
rf.fit(X_train,y_train)

pred=rf.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test,pred))
r2=r2_score(y_test,pred)

corr_df=pd.concat(corr_rows,ignore_index=True)

if len(corr_df)>MAX_ANOMALY_POINTS:
    corr_df=corr_df.sample(MAX_ANOMALY_POINTS,random_state=42)

iso=IsolationForest(contamination=0.01,random_state=42)
corr_df["anomaly"]=iso.fit_predict(corr_df)

TOTAL=time.time()-START
speed=rows/TOTAL

print("Processing complete")
print("Rows processed:",rows)
print("Processing speed:",speed)
print("RMSE:",rmse)
print("R2:",r2)
print("Histogram:",hist_img)
print("Hourly plot:",hour_img)
print("Monthly plot:",month_img)
print("Heatmap:",heat_img)
print("Geo plot:",geo_img)
print("Cleaned data:",clean)