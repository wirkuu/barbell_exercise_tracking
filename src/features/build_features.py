import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans



# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed.pkl")
predictor_columns = list(df.columns[:6])

#plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["font.size"] = 10
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df.info()
for col in predictor_columns:
    df[col]= df[col].interpolate()

df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] ==25]["acc_y"].plot()
df[df["set"] ==50]["acc_y"].plot()

duration =df[df["set"] ==25].index[-1] - df[df["set"] ==25].index[0]
duration.seconds

for s in df["set"].unique():
    start = df[df["set"] ==s].index[0]
    stop = df[df["set"] ==s].index[-1]
    duration = stop - start
    df.loc[df["set"] ==s, "duration"]  = duration.seconds

duration_df =df.groupby(["category"])["duration"].mean()    
duration_df.iloc[0]/ 5 #diving by the number of repeatition base on the num of participants
duration_df.iloc[1]/ 10


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()

LowPass = LowPassFilter()
fs = 1000/200
cutoff = 1.3
df_lowpass =LowPass.low_pass_filter(df_lowpass, "acc_y", fs,cutoff, order =5)

subset = df_lowpass[df_lowpass["set"] ==45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop =True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper right", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper right", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()
PC_values =PCA.determine_pc_explained_variance(df_pca, predictor_columns)

# optmal amount of PC number
plt.figure(figsize = (10,10))
plt.plot(range(1, len(predictor_columns)+1), PC_values)
plt.xlabel("Principal component number")
plt.ylabel("Explained variance")
plt.title("Explained variance by principal components")

# based on the plot, 3 is the PC optimal value as it is the value at the elbow
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)
subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes   (sqrt(x^2 + y^2 + z^2))
# --------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] **2 + df_squared["acc_y"] **2 + df_squared["acc_z"] **2
gyro_r = df_squared["gyro_x"] **2 + df_squared["gyro_y"] **2 + df_squared["gyro_z"] **2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyro_r"] = np.sqrt(gyro_r)

subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyro_r"]].plot(subplots=True)

df_squared

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
Numabs = NumericalAbstraction()
predictor_columns = predictor_columns + ["acc_r", "gyro_r"]


ws =int(1000/200)
for col in predictor_columns:
    df_temporal = Numabs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = Numabs.abstract_numerical(df_temporal, [col], ws, "std")
    
df_temporal_list = []

for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()

    for col in predictor_columns:
        subset = Numabs.abstract_numerical(subset, [col], ws, "mean")
        subset = Numabs.abstract_numerical(subset, [col], ws, "std")

    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyro_y", "gyro_y_temp_mean_ws_5", "gyro_y_temp_std_ws_5"]].plot()  

df_temporal.info()

# --------------------------------------------------------------
# Frequency features(discrete fourier transform DFT combination of diff sinusiols with diff freqs)
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()

freqAbs = FourierTransformation()

fs =int(1000/200)
ws = int(2000/200)

df_freq = freqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)


subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()
subset[["acc_y_max_freq", "acc_y_freq_weighted","acc_y_pse", "acc_y_freq_2.0_Hz_ws_10", "acc_y_freq_2.5_Hz_ws_10" ]].plot()


df_freq_list = []
for s in df_freq["set"].unique():
    print(f"applying fourier transform for {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = freqAbs.abstract_frequency(subset,predictor_columns,ws,fs)
    df_freq_list.append(subset)
    
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)




# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()

df_freq = df_freq.iloc[::2] #dropping every second row to remove overlapping windows



# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()
cluster_columns = ["acc_x","acc_y", "acc_z"]
k_values = range(2, 10)
inertia_values = [ ]
for k in k_values:
    subset =df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertia_values.append(kmeans.inertia_)


plt.figure(figsize=(10, 10))
plt.plot(k_values, inertia_values, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia or sum of squared distances")
plt.show()




kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset =df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot( projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"Cluster {c}")
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
ax.legend()
plt.show()



#clustering by labels

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot( projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"Cluster {l}")
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
ax.legend()
plt.show()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")