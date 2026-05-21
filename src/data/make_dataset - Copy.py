import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
single_file_gyro = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion......(*.csv or *.txt to specify the file type)
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/*.csv")
len(files)
# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
data_path = "../../data/raw/MetaMotion\\"
f = files[0]

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123MetaWear_2019")
# dataframe
df = pd.read_csv(f)
# add new columns to the dataframe with the extracted features
df["participant"] = participant
df["label"] = label
df["category"] = category

# participant
# level--> exercise
# category-->how heavy


# --------------------------------------------------------------
# for loop use to read all files and extract features from filename and add to dataframe since we are also extracting features from the name.
# --------------------------------------------------------------
acc_df = pd.DataFrame()
gyro_df = pd.DataFrame()

acc_set = 1
gyro_set = 1
# 1. extract data-->read f, 2. add data labels and put in the dataframe, acc_df
for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123MetaWear_2019")
    print(
        f"Processing file: {f} | Participant: {participant} | Label: {label} | Category: {category}"
    )
    df = pd.read_csv(f)
    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])
    elif "Gyroscope" in f:
        df["set"] = gyro_set
        gyro_set += 1
        gyro_df = pd.concat([gyro_df, df])

# to select a specific set of data, for example, set 94 from the accelerometer dataset
acc_df[acc_df["set"] == 94]

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------
gyro_df.info()
# convert epoch--> unix time (ms) to datetime
pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")


# data cleaning by deleting the columns that are not needed for analysis, in this case, we are deleting the epoch (ms), time (01:00), and elapsed (s) columns since we have already converted the epoch to datetime and set it as the index of the dataframe.
del acc_df["epoch (ms)"]
del gyro_df["epoch (ms)"]

del acc_df["time (01:00)"]
del gyro_df["time (01:00)"]
del acc_df["elapsed (s)"]
del gyro_df["elapsed (s)"]

# extract week,month, year, day, hour, minute, second from the datetime
# pd.to_datetime(df["time (01:00)"]).dt.month


# --------------------------------------------------------------
# Turn into function that be run all files ie doing the above steps in a function that can be run on all files in the directory and extract features from the filename and add to the dataframe, and also convert epoch to datetime and set it as the index of the dataframe, and also delete the columns that are not needed for analysis.
# --------------------------------------------------
# clean code

files = glob("../../data/raw/MetaMotion/*.csv")


def data_from_files(files):
    acc_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    acc_set = 1
    gyro_set = 1
    # 1. extract data-->read f, 2. add data labels and put in the dataframe, acc_df
    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123MetaWear_2019")
        print(
            f"Processing file: {f} | Participant: {participant} | Label: {label} | Category: {category}"
        )
        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        elif "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_set += 1
            gyro_df = pd.concat([gyro_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

    # data cleaning by deleting the columns that are not needed for analysis, in this case, we are deleting the epoch (ms), time (01:00), and elapsed (s) columns since we have already converted the epoch to datetime and set it as the index of the dataframe.
    del acc_df["epoch (ms)"]
    del gyro_df["epoch (ms)"]

    del acc_df["time (01:00)"]
    del gyro_df["time (01:00)"]
    del acc_df["elapsed (s)"]
    del gyro_df["elapsed (s)"]
    # when done give the back the cleaned dataframes
    return acc_df, gyro_df


#

# pass the list of files to the function and get back the cleaned dataframes
acc_df, gyro_df = data_from_files(files)
# --------------------------------------------------------------
# Merging datasets for selected columns(axis=1) (if needed)
# --------------------------------------------------

data_merged = pd.concat([acc_df.iloc[:, :3], gyro_df], axis=1)
data_merged.info()
# drop out any rows with missing values (NaN) since we are merging the datasets and there might be some missing values in the merged dataset, we can drop them to clean the data.
data_merged.dropna(inplace=True)


# rename columns
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "participant",
    "label",
    "category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# reduce the frequency so as to have data for every row
# the index needs to be of a datetime pandas object.
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
# sampling Dictionary to specify the aggregation function for each column when resampling the data. For example, we can take the mean of the accelerometer and gyroscope values for each 200ms interval, while keeping the participant, label, category, and set columns unchanged.
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyro_x": "mean",
    "gyro_y": "mean",
    "gyro_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}
data_merged[:1000].resample(rule="200ms").apply(
    sampling
)  # resample to 1 second frequency and take the mean of the values in each second

# so processor does not blow up , data is exported in chunks of 1000 rows
# split by day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

data_resampled.info()
data_resampled["set"] = data_resampled["set"].astype("int")
data_resampled
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resampled.to_pickle("../../data/interim/data_resampled.pkl ")
