from ais_predict.datasets.bigquery import download
import ais_predict.visualization.dyngp as dgp_plot
import ais_predict.visualization.plotting as plotting
import ais_predict.trajpred.dyngp as dyngp
import ais_predict.utilities.metrics as metrics
from ais_predict.trajpred.cvm import constant_velocity_model 

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import random
import seaborn as sns

sns.set()
latex_width = 369.88583
plt.rcParams["figure.figsize"] = plotting.set_size(latex_width)

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.latex.preamble": r'\usepackage{amsfonts}'
}

plt.rcParams.update(tex_fonts)




##### SIM CONFIG ####
sim_name = "straight_line"
use_pdaf=True
train_max_cog_dist = 10.0
train_max_sog_dist = 5.0
train_max_pos_dist = 1000.0

test_max_minutes = 30
test_min_minutes = 15
test_min_sog = 3
N_test = 350

try:
    df = pd.read_pickle(f"{sim_name}_cache.pkl")
    df = gpd.GeoDataFrame(df)
except:
    print("Downloading data...")
    df = download(limit=100000, shuffle=True, lead=10)
    df.to_pickle(f"{sim_name}_cache.pkl")
    print("complete")

### Preprocessing

df["date"] = df.timestamp.dt.date

minutes = ((df.timestamp_10 - df.timestamp).dt.seconds / 60)
msk = (minutes < test_max_minutes) & (minutes > test_min_minutes)
msk = msk & (df.sog > test_min_sog)   

d = df.loc[msk]

cog_sum = abs(d.cog_1 - d.cog)
cog_cols = d.columns.str.contains("cog")
for i in range(1, 10):
    diff = abs(d[f"cog_{i+1}"] - d[f"cog_{i}"])
    cog_sum += diff
d = d.loc[cog_sum > 40].copy()

grouping = d.groupby(["mmsi", "date"])
groups = list(grouping.groups.keys())
random.shuffle(groups)

# Use 1/3 for test and 2/3 for train
test_groups = groups[1::3]

test_df = d.groupby(["mmsi", "date"]).filter(lambda x: (x.mmsi.iloc[0], x.date.iloc[0]) in test_groups)
# Remove test mmsi/date pairs from training set to avoid leaking test trajectories 
train_df = df.groupby(["mmsi", "date"]).filter(lambda x: (x.mmsi.iloc[0], x.date.iloc[0]) not in test_groups)

print(f"Test Dataframe contains {len(test_df.index)} samples, sampling {N_test} randomly")

test_df = test_df.sample(350, replace=False)

train_pos = np.vstack([train_df.position.x, train_df.position.y]).T
test_pos = np.vstack([test_df.position.x, test_df.position.y]).T
pos_msk = np.linalg.norm(train_pos[np.newaxis,...] - test_pos.reshape((-1, 1, 2)), axis=-1) < 1000
cog_msk = abs(train_df.cog.to_numpy()[np.newaxis, ...] - test_df.cog.to_numpy()[..., np.newaxis]) < 10
sog_msk = abs(train_df.sog.to_numpy()[np.newaxis, ...] - test_df.sog.to_numpy()[..., np.newaxis]) < 2

msk = pos_msk & cog_msk & sog_msk
plt.ioff()
results_df = []
for i in range(msk.shape[0]):
    try:
        test = test_df.iloc[i:i+1]

        train = train_df.loc[msk[i]]
        pos_dist = train.position.distance(test.position.to_numpy()[0])
        pos_dist.sort_values(inplace=True)

        train = train.loc[pos_dist.index]

        train_x, train_y = dyngp.samples_from_lag_n_df(train, 10)
        # Remove any non-moving samples
        speed_msk = np.linalg.norm(train_y, axis=1) > 2
        train_x, train_y = train_x[speed_msk], train_y[speed_msk]

        # Pick first (assumed best) 500 samples - reduce computation time
        train_x = train_x[:500]
        train_y = train_y[:500]

        
        cog_train_x, cog_train_y = dyngp.samples_from_lag_n_df(train, 10, use_cog=True)
        # Remove any non-moving samples
        speed_msk = np.linalg.norm(cog_train_y, axis=1) > 2
        cog_train_x, cog_train_y = cog_train_x[speed_msk], cog_train_y[speed_msk]

        # Pick first (assumed best) 500 samples - reduce computation time
        cog_train_x = cog_train_x[:500]
        cog_train_y = cog_train_y[:500]

        test_x, _ = dyngp.samples_from_lag_n_df(test, 10)

        if train_x.shape[0] < 20:
            print("Too few valid samples for train, skipping")
            continue

        m = np.concatenate([train_x[:, :2], cog_train_x[:, :2]], axis=0).min(axis=0)

        train_x[:, :2] -= m
        cog_train_x[:, :2] -= m
        test_x[:, :2] -= m

        params = dyngp.get_default_mle_params(train_x, train_y, n_restarts_optimizer=5)
        dgp = dyngp.DynGP(train_x, train_y, params, normalize_y=True)
        dyngp_x_pdaf, dyngp_P_pdaf = res = dgp.kalman(test_x[0, :2], test_x[-1, -1], dt=10, pdaf_update=True)
        dyngp_x_no_pdaf, dyngp_P_no_pdaf = res_no_pdaf = dgp.kalman(test_x[0, :2], test_x[-1, -1], dt=10, pdaf_update=False)
        pred_x_cvm = constant_velocity_model(test_x[0, :2], dyngp_x_pdaf[:, -1], test.cog.to_numpy(), test.sog.to_numpy())

        cog_params = dyngp.get_default_mle_params(cog_train_x, cog_train_y, n_restarts_optimizer=5)
        cog_dgp = dyngp.DynGP(cog_train_x, cog_train_y, cog_params, normalize_y=True)
        cog_dyngp_x_pdaf, cog_dyngp_P_pdaf = cog_res_pdaf = cog_dgp.kalman(test_x[0, :2], test_x[-1, -1], dt=10, pdaf_update=True)
        cog_dyngp_x_no_pdaf, cog_dyngp_P_no_pdaf = cog_res_no_pdaf = cog_dgp.kalman(test_x[0, :2], test_x[-1, -1], dt=10, pdaf_update=False)

        # DynGP with PDAF
        dyngp_traj_subset_pdaf, dyngp_traj_err_pdaf = metrics.compare_trajectories(dyngp_x_pdaf, test_x)
        dyngp_path_err_pdaf = metrics.path_error(dyngp_x_pdaf, test_x)
        dyngp_nees_pdaf = metrics.nees(dyngp_x_pdaf, dyngp_P_pdaf, test_x)

        # DynGP without PDAF
        dyngp_traj_subset_no_pdaf, dyngp_traj_err_no_pdaf = metrics.compare_trajectories(dyngp_x_no_pdaf, test_x)
        dyngp_path_err_no_pdaf = metrics.path_error(dyngp_x_no_pdaf, test_x)
        dyngp_nees_no_pdaf = metrics.nees(dyngp_x_pdaf, dyngp_P_pdaf, test_x)

        # DynGP with PDAF - COG Data
        cog_dyngp_traj_subset_pdaf, cog_dyngp_traj_err_pdaf = metrics.compare_trajectories(cog_dyngp_x_pdaf, test_x)
        cog_dyngp_path_err_pdaf = metrics.path_error(cog_dyngp_x_pdaf, test_x)
        cog_dyngp_nees_pdaf = metrics.nees(cog_dyngp_x_pdaf, cog_dyngp_P_pdaf, test_x)

        # DynGP without PDAF - COG Data
        cog_dyngp_traj_subset_no_pdaf, cog_dyngp_traj_err_no_pdaf = metrics.compare_trajectories(cog_dyngp_x_no_pdaf, test_x)
        cog_dyngp_path_err_no_pdaf = metrics.path_error(cog_dyngp_x_no_pdaf, test_x)
        cog_dyngp_nees_no_pdaf = metrics.nees(cog_dyngp_x_pdaf, cog_dyngp_P_pdaf, test_x)

        # CVM
        cvm_traj_subset, cvm_traj_err = metrics.compare_trajectories(pred_x_cvm, test_x)
        cvm_path_err = metrics.path_error(pred_x_cvm, test_x)

        cvm_df = pd.DataFrame({
            "timestep": test_x[:, -1],
            "ground_truth": list(test_x),
            "prediction": list(cvm_traj_subset),
            "trajectory_error": list(cvm_traj_err),
            "path_error": list(cvm_path_err),
        })
        cvm_df["method"] = "cvm"
        cvm_df["y_source"] = "cog"

        dyngp_pdaf_df = pd.DataFrame({
            "timestep": test_x[:, -1],
            "ground_truth": list(test_x),
            "prediction": list(dyngp_traj_subset_pdaf),
            "trajectory_error": list(dyngp_traj_err_pdaf),
            "path_error": list(dyngp_path_err_pdaf),
            "nees": list(dyngp_nees_pdaf)
        })
        dyngp_pdaf_df["method"] = "dyngp_pdaf"
        dyngp_pdaf_df["y_source"] = "finite_difference"
        dyngp_pdaf_df["train_samples"] = train_x.shape[0]

        dyngp_no_pdaf_df = pd.DataFrame({
            "timestep": test_x[:, -1],
            "ground_truth": list(test_x),
            "prediction": list(dyngp_traj_subset_no_pdaf),
            "trajectory_error": list(dyngp_traj_err_no_pdaf),
            "path_error": list(dyngp_path_err_no_pdaf),
            "nees": list(dyngp_nees_no_pdaf)
        })
        dyngp_no_pdaf_df["method"] = "dyngp_no_pdaf"
        dyngp_no_pdaf_df["y_source"] = "finite_difference"
        dyngp_no_pdaf_df["train_samples"] = train_x.shape[0]


        cog_dyngp_pdaf_df = pd.DataFrame({
            "timestep": test_x[:, -1],
            "ground_truth": list(test_x),
            "prediction": list(cog_dyngp_traj_subset_pdaf),
            "trajectory_error": list(cog_dyngp_traj_err_pdaf),
            "path_error": list(cog_dyngp_path_err_pdaf),
            "nees": list(cog_dyngp_nees_pdaf)
        })
        cog_dyngp_pdaf_df["method"] = "dyngp_pdaf"
        cog_dyngp_pdaf_df["y_source"] = "cog"
        cog_dyngp_pdaf_df["train_samples"] = train_x.shape[0]

        cog_dyngp_no_pdaf_df = pd.DataFrame({
            "timestep": test_x[:, -1],
            "ground_truth": list(test_x),
            "prediction": list(cog_dyngp_traj_subset_no_pdaf),
            "trajectory_error": list(cog_dyngp_traj_err_no_pdaf),
            "path_error": list(cog_dyngp_path_err_no_pdaf),
            "nees": list(cog_dyngp_nees_no_pdaf)
        })
        cog_dyngp_no_pdaf_df["method"] = "dyngp_no_pdaf"
        cog_dyngp_no_pdaf_df["y_source"] = "cog"
        cog_dyngp_no_pdaf_df["train_samples"] = train_x.shape[0]



        temp_df = pd.concat([
            cvm_df,
            dyngp_pdaf_df,
            dyngp_no_pdaf_df,
            cog_dyngp_pdaf_df,
            cog_dyngp_no_pdaf_df
        ])


        
        temp_df["test_id"] = i
        results_df.append(temp_df)



        plt.clf()
        plt.plot(*pred_x_cvm.T[:2], color="green", label="CVM", alpha=0.5)
        dgp_plot.kalman_figure(dgp, res, test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/raw/run_{i}_pos_pdaf.pdf", format="pdf", bbox_inches='tight')
        plt.clf()
        dgp_plot.kalman_state_figure(dgp, res, test_x, label="GP")
        plt.savefig(f"{sim_name}/raw/run_{i}_state_pdaf.pdf", format="pdf", bbox_inches='tight')


        plt.clf()
        plt.plot(*pred_x_cvm.T[:2], color="green", label="CVM", alpha=0.5)
        dgp_plot.kalman_figure(dgp, res_no_pdaf, test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/raw/run_{i}_pos_no_pdaf.pdf", format="pdf", bbox_inches='tight')
        plt.clf()
        dgp_plot.kalman_state_figure(dgp, res_no_pdaf, test_x, label="GP")
        plt.savefig(f"{sim_name}/raw/run_{i}_state_no_pdaf.pdf", format="pdf", bbox_inches='tight')
        
        plt.clf()
        plt.plot(*pred_x_cvm.T[:2], color="green", label="CVM", alpha=0.5)
        dgp_plot.kalman_figure(cog_dgp, cog_res_pdaf, test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/raw/run_{i}_pos_pdaf_cog.pdf", format="pdf", bbox_inches='tight')
        plt.clf()
        dgp_plot.kalman_state_figure(cog_dgp, cog_res_pdaf, test_x, label="GP")
        plt.savefig(f"{sim_name}/raw/run_{i}_state_pdaf_cog.pdf", format="pdf", bbox_inches='tight')


        plt.clf()
        plt.plot(*pred_x_cvm.T[:2], color="green", label="CVM", alpha=0.5)
        dgp_plot.kalman_figure(cog_dgp, cog_res_no_pdaf, test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/run_{i}_pos_no_pdaf_cog.pdf", format="pdf", bbox_inches='tight')
        plt.clf()
        dgp_plot.kalman_state_figure(cog_dgp, cog_res_no_pdaf, test_x, label="GP")
        plt.savefig(f"{sim_name}/run_{i}_state_no_pdaf_cog.pdf", format="pdf", bbox_inches='tight')
        
        print("Completed", i, "with stats:")
        print("Completed", i, "with stats:")
        print(temp_df)
    except Exception as e:
        print(e)

out = pd.concat(results_df)
out.to_pickle(f"{sim_name}.pkl")

