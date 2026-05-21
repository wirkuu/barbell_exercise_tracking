import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/data_resampled.pkl")
df1 = pd.read_csv(
    "../../data/raw/MetaMotion/B-ohp-medium3-rpe9_MetaWear_2019-01-11T16.59.28.181_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)
# --------------------------------------------------------------
# Plot single columns
# the xxx.reset_index(drop=True) is used to view data based on the available number of samples, if not used, we get it plotted based on the time
set_df = df[df["set"] == 1]

plt.plot(set_df["acc_y"].reset_index(drop=True), label="acc_y")
plt.legend()

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
for label in df["label"].unique():
    subset = df[df["label"] == label]
    display(subset.head(2))
    fig, ax = plt.subplots()
    ax.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()


for label in df["label"].unique():
    subset = df[df["label"] == label]
    display(subset.head(2))
    fig, ax = plt.subplots()
    ax.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = [20, 5]
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# another way to create subsets using the query method and string matching
# --------------------------------------------------------------

category_df = (
    df.query("label=='squat'").query("participant == 'A'").reset_index(drop=True)
)
fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Compare participants
# sort so that all participant data is at the same part of the graph,
# reset so we plot not with respect to time as timing wasnt aranged
# --------------------------------------------------------------
participant_df = df.query("label=='bench'").sort_values("participant").reset_index()
fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label = "squat"
participant = "A"
all_axis_df = (
    df.query(f"label=='{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()
# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# AGyroscope data plots
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label=='{label}'")
            .query(f"participant == '{participant}'")
            .reset_index(drop=True)
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax)
            ax.set_ylabel("gyro_y")
            ax.set_xlabel("samples")
            ax.set_title(f"{label}-{participant}".title())
            plt.legend()
            plt.show()


# ACC_meter data plots
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label=='{label}'")
            .query(f"participant == '{participant}'")
            .reset_index(drop=True)
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            ax.set_title(f"{label}-{participant}".title())
            plt.legend()
            plt.show()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "row"
participant = "A"
combined_df = (
    df.query(f"label=='{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combined_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_df[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax[1])

ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].set_xlabel("samples")


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_df = (
            df.query(f"label=='{label}'")
            .query(f"participant == '{participant}'")
            .reset_index(drop=True)
        )
        if len(combined_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_df[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax[1])

            ax[0].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].set_xlabel("samples")
            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()
