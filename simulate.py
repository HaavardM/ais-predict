
# Disable BLAS paralellization
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from ais_predict.datasets.bigquery import download
from ais_predict.datasets.raw_ais import convert_ais_df_to_trajectories
import ais_predict.visualization.dyngp as dgp_plot
import ais_predict.visualization.plotting as plotting
import ais_predict.trajpred.dyngp as dyngp
import ais_predict.utilities.metrics as metrics
from ais_predict.trajpred.cvm import constant_velocity_model 
from ais_predict.trajpred import posgp
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import random
import seaborn as sns

sns.set()
latex_width = 369.88583
plt.rcParams["figure.figsize"] = plotting.set_size(latex_width, .6)

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
    df = download(limit=None, shuffle=True, crs="epsg:25832")
    df = convert_ais_df_to_trajectories(df)
    df.to_pickle(f"{sim_name}_cache.pkl")
    print("complete")

### Preprocessing

df["date"] = df.timestamp.dt.date
df = df.loc[~df.mmsi.isin((258323000, 257048700, 257374400))]

msk = (df.sog > test_min_sog)   

d = df.loc[msk]

cog_cols = df.columns[df.columns.str.contains("cog")]
cog_dist = (df[cog_cols].diff(axis=1) + 180) % 360 - 180
d = d.loc[cog_dist.abs().sum(axis=1) < 30].copy()

grouping = d.groupby(["mmsi", "date"])
groups = list(grouping.groups.keys())
random.shuffle(groups)

# Use 1/3 for test and 2/3 for train
test_groups = groups[1::3]

test_df = d.groupby(["mmsi", "date"]).filter(lambda x: (x.mmsi.iloc[0], x.date.iloc[0]) in test_groups)
print(f"Test Dataframe contains {len(test_df.index)} samples, sampling {N_test} randomly")
test_df = test_df.sample(frac=1)
test_groups = list(test_df.groupby(["mmsi", "date"]).groups.keys())
# Remove test mmsi/date pairs from training set to avoid leaking test trajectories 
train_df = df.groupby(["mmsi", "date"]).filter(lambda x: (x.mmsi.iloc[0], x.date.iloc[0]) not in test_groups)



train_pos = np.vstack([train_df.position.x, train_df.position.y]).T
test_pos = np.vstack([test_df.position.x, test_df.position.y]).T
pos_msk = np.linalg.norm(train_pos[np.newaxis,...] - test_pos.reshape((-1, 1, 2)), axis=-1) < 200
cog_dist = train_df.cog.to_numpy()[np.newaxis, ...] - test_df.cog.to_numpy()[..., np.newaxis]
cog_dist = ((cog_dist + 180) % 360) - 180
cog_msk = abs(cog_dist) < 20
sog_msk = abs(train_df.sog.to_numpy()[np.newaxis, ...] - test_df.sog.to_numpy()[..., np.newaxis]) < 4

msk = pos_msk & cog_msk & sog_msk

ix = msk.sum(axis=1) > 3
print(msk.shape, train_df.shape, test_df.shape)

test_df = test_df.loc[ix].iloc[:N_test]
msk = msk[ix][:N_test]

plt.ioff()
results_df = []

P_0 = np.eye(2) * 200**2
print(msk.shape[0])
def run(i):
    try:
        test = test_df.iloc[i:i+1]

        train = train_df.loc[msk[i]]
        pos_dist = train.position.distance(test.position.to_numpy()[0])
        pos_dist.sort_values(inplace=True)

        train = train.loc[pos_dist.index]

        train_x, train_y = dyngp.samples_from_lag_n_df(train, 19)
        # Remove any non-moving samples
        speed_msk = np.linalg.norm(train_y, axis=1) > 2
        train_x, train_y = train_x[speed_msk], train_y[speed_msk]

        # Pick first (assumed best) 500 samples - reduce computation time
        train_x = train_x[:1000]
        train_y = train_y[:1000]

    
        cog_train_x, cog_train_y = dyngp.samples_from_lag_n_df(train, 19, use_cog=True)
        # Remove any non-moving samples
        speed_msk = np.linalg.norm(cog_train_y, axis=1) > 2
        cog_train_x, cog_train_y = cog_train_x[speed_msk], cog_train_y[speed_msk]

        # Pick first (assumed best) 500 samples - reduce computation time
        cog_train_x = cog_train_x[:1000]
        cog_train_y = cog_train_y[:1000]


        posgp_train_x, posgp_train_y = posgp.samples_from_lag_n_df(train, 19)

        dyngp_test_x, _ = dyngp.samples_from_lag_n_df(test, 19)
        posgp_test_x, posgp_test_y = posgp.samples_from_lag_n_df(test, 19)

        posgp_train_x, posgp_train_y = posgp_train_x[:1000], posgp_train_y[:1000]

        if train_x.shape[0] < 20:
            print("Too few valid samples for train, skipping")
            return None

        m = np.concatenate([train_x[:, :2], cog_train_x[:, :2], posgp_train_y[:, :2]], axis=0).min(axis=0)

        train_x[:, :2] -= m
        cog_train_x[:, :2] -= m
        posgp_train_x[:, :2] -= m
        posgp_train_y[:, :2] -= m
        dyngp_test_x[:, :2] -= m
        posgp_test_x[:, :2] -= m
        posgp_test_y[:, :2] -= m

        params = dyngp.get_default_mle_params(train_x, train_y, n_restarts_optimizer=10)
        dgp = dyngp.DynGP(train_x, train_y, params, normalize_y=True)
        dyngp_x_pdaf, dyngp_P_pdaf = res_pdaf = dgp.kalman(dyngp_test_x[0, :2], dyngp_test_x[-1, -1], dt=10, pdaf_update=True, synthetic_update=False, P_0=P_0, update_P=True)
        dyngp_x_pdaf_no_P, dyngp_P_pdaf_no_P = res_pdaf_no_P = dgp.kalman(dyngp_test_x[0, :2], dyngp_test_x[-1, -1], dt=10, pdaf_update=True, synthetic_update=False, P_0=P_0, update_P=False)
        dyngp_x_syn, dyngp_P_syn = res_syn = dgp.kalman(dyngp_test_x[0, :2], dyngp_test_x[-1, -1], dt=10, pdaf_update=False, synthetic_update=True, P_0=P_0, update_P=True)
        dyngp_x_syn_no_P, dyngp_P_syn_no_P = res_syn_no_P = dgp.kalman(dyngp_test_x[0, :2], dyngp_test_x[-1, -1], dt=10, pdaf_update=False, synthetic_update=True, P_0=P_0, update_P=False)
        dyngp_x_no_pdaf, dyngp_P_no_pdaf = res = dgp.kalman(dyngp_test_x[0, :2], dyngp_test_x[-1, -1], dt=10, pdaf_update=False, synthetic_update=False, P_0=P_0, update_P=True)
        pred_x_cvm = constant_velocity_model(dyngp_test_x[0, :2], dyngp_x_pdaf[:, -1], test.cog.to_numpy(), test.sog.to_numpy())

        cog_params = dyngp.get_default_mle_params(cog_train_x, cog_train_y, n_restarts_optimizer=10)
        cog_dgp = dyngp.DynGP(cog_train_x, cog_train_y, cog_params, normalize_y=True)
        cog_dyngp_x_pdaf, cog_dyngp_P_pdaf = cog_res_pdaf = cog_dgp.kalman(dyngp_test_x[0, :2], dyngp_test_x[-1, -1], dt=10, pdaf_update=True, synthetic_update=False, P_0=P_0, update_P=True)
        cog_dyngp_x_pdaf_no_P, cog_dyngp_P_pdaf_no_P = cog_res_pdaf_no_P = cog_dgp.kalman(dyngp_test_x[0, :2], dyngp_test_x[-1, -1], dt=10, pdaf_update=True, synthetic_update=False, P_0=P_0, update_P=False)
        cog_dyngp_x_syn, cog_dyngp_P_syn = cog_res_syn = cog_dgp.kalman(dyngp_test_x[0, :2], dyngp_test_x[-1, -1], dt=10, pdaf_update=False, synthetic_update=True, P_0=P_0, update_P=True)
        cog_dyngp_x_syn_no_P, cog_dyngp_P_syn_no_P = cog_res_syn_no_P = cog_dgp.kalman(dyngp_test_x[0, :2], dyngp_test_x[-1, -1], dt=10, pdaf_update=False, synthetic_update=True, P_0=P_0, update_P=False)
        cog_dyngp_x_no_pdaf, cog_dyngp_P_no_pdaf = cog_res = cog_dgp.kalman(dyngp_test_x[0, :2], dyngp_test_x[-1, -1], dt=10, pdaf_update=False, synthetic_update=False, P_0=P_0, update_P=True)

        pgp = posgp.PosGP(posgp_train_x, posgp_train_y)
        x = posgp_test_x[0][np.newaxis,...].repeat(100, axis=0)
        x[:, -1] = np.linspace(0, posgp_test_x[-1, -1], 100)
        x = np.append(x, posgp_test_x, axis=0)
        six = np.argsort(x[:, -1])
        x = x[six]
        pred_y_posgp, pred_P_posgp = pgp(x)



        # DynGP with PDAF
        dyngp_traj_subset_pdaf, dyngp_traj_err_pdaf = metrics.compare_trajectories(dyngp_x_pdaf, dyngp_test_x)
        dyngp_path_err_pdaf = metrics.path_error(dyngp_x_pdaf, dyngp_test_x)
        dyngp_nees_pdaf = metrics.nees(dyngp_x_pdaf, dyngp_P_pdaf, dyngp_test_x)

        # DynGP with PDAF
        dyngp_traj_subset_pdaf_no_P, dyngp_traj_err_pdaf_no_P = metrics.compare_trajectories(dyngp_x_pdaf_no_P, dyngp_test_x)
        dyngp_path_err_pdaf_no_P = metrics.path_error(dyngp_x_pdaf_no_P, dyngp_test_x)
        dyngp_nees_pdaf_no_P = metrics.nees(dyngp_x_pdaf_no_P, dyngp_P_pdaf_no_P, dyngp_test_x)


        # DynGP with syn
        dyngp_traj_subset_syn, dyngp_traj_err_syn = metrics.compare_trajectories(dyngp_x_syn, dyngp_test_x)
        dyngp_path_err_syn = metrics.path_error(dyngp_x_syn, dyngp_test_x)
        dyngp_nees_syn = metrics.nees(dyngp_x_syn, dyngp_P_syn, dyngp_test_x)

        # DynGP with syn - no P
        dyngp_traj_subset_syn_no_P, dyngp_traj_err_syn_no_P = metrics.compare_trajectories(dyngp_x_syn_no_P, dyngp_test_x)
        dyngp_path_err_syn_no_P = metrics.path_error(dyngp_x_syn_no_P, dyngp_test_x)
        dyngp_nees_syn_no_P = metrics.nees(dyngp_x_syn_no_P, dyngp_P_syn_no_P, dyngp_test_x)

        # DynGP with PDAF
        cog_dyngp_traj_subset_pdaf_no_P, cog_dyngp_traj_err_pdaf_no_P = metrics.compare_trajectories(cog_dyngp_x_pdaf_no_P, dyngp_test_x)
        cog_dyngp_path_err_pdaf_no_P = metrics.path_error(cog_dyngp_x_pdaf_no_P, dyngp_test_x)
        cog_dyngp_nees_pdaf_no_P = metrics.nees(cog_dyngp_x_pdaf_no_P, cog_dyngp_P_pdaf_no_P, dyngp_test_x)


        # DynGP with syn
        cog_dyngp_traj_subset_syn, cog_dyngp_traj_err_syn = metrics.compare_trajectories(cog_dyngp_x_syn, dyngp_test_x)
        cog_dyngp_path_err_syn = metrics.path_error(cog_dyngp_x_syn, dyngp_test_x)
        cog_dyngp_nees_syn = metrics.nees(cog_dyngp_x_syn, cog_dyngp_P_syn, dyngp_test_x)

        # DynGP with syn - no P
        cog_dyngp_traj_subset_syn_no_P, cog_dyngp_traj_err_syn_no_P = metrics.compare_trajectories(cog_dyngp_x_syn_no_P, dyngp_test_x)
        cog_dyngp_path_err_syn_no_P = metrics.path_error(cog_dyngp_x_syn_no_P, dyngp_test_x)
        cog_dyngp_nees_syn_no_P = metrics.nees(cog_dyngp_x_syn_no_P, cog_dyngp_P_syn_no_P, dyngp_test_x)

        # DynGP without PDAF
        dyngp_traj_subset_no_pdaf, dyngp_traj_err_no_pdaf = metrics.compare_trajectories(dyngp_x_no_pdaf, dyngp_test_x)
        dyngp_path_err_no_pdaf = metrics.path_error(dyngp_x_no_pdaf, dyngp_test_x)
        dyngp_nees_no_pdaf = metrics.nees(dyngp_x_pdaf, dyngp_P_no_pdaf, dyngp_test_x)

        # DynGP with PDAF - COG Data
        cog_dyngp_traj_subset_pdaf, cog_dyngp_traj_err_pdaf = metrics.compare_trajectories(cog_dyngp_x_pdaf, dyngp_test_x)
        cog_dyngp_path_err_pdaf = metrics.path_error(cog_dyngp_x_pdaf, dyngp_test_x)
        cog_dyngp_nees_pdaf = metrics.nees(cog_dyngp_x_pdaf, cog_dyngp_P_pdaf, dyngp_test_x)

        # DynGP without PDAF - COG Data
        cog_dyngp_traj_subset_no_pdaf, cog_dyngp_traj_err_no_pdaf = metrics.compare_trajectories(cog_dyngp_x_no_pdaf, dyngp_test_x)
        cog_dyngp_path_err_no_pdaf = metrics.path_error(cog_dyngp_x_no_pdaf, dyngp_test_x)
        cog_dyngp_nees_no_pdaf = metrics.nees(cog_dyngp_x_pdaf, cog_dyngp_P_no_pdaf, dyngp_test_x)

        # CVM
        cvm_traj_subset, cvm_traj_err = metrics.compare_trajectories(pred_x_cvm, dyngp_test_x)
        cvm_path_err = metrics.path_error(pred_x_cvm, dyngp_test_x)

        # PosGP
        pos_gp_traj_subset, pos_gp_traj_err = metrics.compare_trajectories(pred_y_posgp, posgp_test_y)
        pos_gp_path_err = metrics.path_error(pred_y_posgp, posgp_test_y) 
        pos_gp_nees = metrics.nees(pred_y_posgp, pred_P_posgp, posgp_test_y)



        cvm_df = pd.DataFrame({
            "timestep": dyngp_test_x[:, -1],
            "ground_truth": list(dyngp_test_x),
            "prediction": list(cvm_traj_subset),
            "trajectory_error": list(cvm_traj_err),
            "path_error": list(cvm_path_err),
        })
        cvm_df["method"] = "cvm"
        cvm_df["y_source"] = "cog"

        dyngp_pdaf_df = pd.DataFrame({
            "timestep": dyngp_test_x[:, -1],
            "ground_truth": list(dyngp_test_x),
            "prediction": list(dyngp_traj_subset_pdaf),
            "trajectory_error": list(dyngp_traj_err_pdaf),
            "path_error": list(dyngp_path_err_pdaf),
            "nees": list(dyngp_nees_pdaf)
        })
        dyngp_pdaf_df["method"] = "dyngp_pdaf"
        dyngp_pdaf_df["y_source"] = "finite_difference"
        dyngp_pdaf_df["train_samples"] = train_x.shape[0]

        dyngp_pdaf_no_P_df = pd.DataFrame({
            "timestep": dyngp_test_x[:, -1],
            "ground_truth": list(dyngp_test_x),
            "prediction": list(dyngp_traj_subset_pdaf_no_P),
            "trajectory_error": list(dyngp_traj_err_pdaf_no_P),
            "path_error": list(dyngp_path_err_pdaf_no_P),
            "nees": list(dyngp_nees_pdaf_no_P)
        })
        dyngp_pdaf_no_P_df["method"] = "dyngp_pdaf_no_P"
        dyngp_pdaf_no_P_df["y_source"] = "finite_difference"
        dyngp_pdaf_no_P_df["train_samples"] = train_x.shape[0]

        dyngp_syn_df = pd.DataFrame({
            "timestep": dyngp_test_x[:, -1],
            "ground_truth": list(dyngp_test_x),
            "prediction": list(dyngp_traj_subset_syn),
            "trajectory_error": list(dyngp_traj_err_syn),
            "path_error": list(dyngp_path_err_syn),
            "nees": list(dyngp_nees_syn)
        })
        dyngp_syn_df["method"] = "dyngp_syn"
        dyngp_syn_df["y_source"] = "finite_difference"
        dyngp_syn_df["train_samples"] = train_x.shape[0]

        dyngp_syn_no_P_df = pd.DataFrame({
            "timestep": dyngp_test_x[:, -1],
            "ground_truth": list(dyngp_test_x),
            "prediction": list(dyngp_traj_subset_syn_no_P),
            "trajectory_error": list(dyngp_traj_err_syn_no_P),
            "path_error": list(dyngp_path_err_syn_no_P),
            "nees": list(dyngp_nees_syn_no_P)
        })
        dyngp_syn_no_P_df["method"] = "dyngp_syn_no_P"
        dyngp_syn_no_P_df["y_source"] = "finite_difference"
        dyngp_syn_no_P_df["train_samples"] = train_x.shape[0]

        dyngp_no_pdaf_df = pd.DataFrame({
            "timestep": dyngp_test_x[:, -1],
            "ground_truth": list(dyngp_test_x),
            "prediction": list(dyngp_traj_subset_no_pdaf),
            "trajectory_error": list(dyngp_traj_err_no_pdaf),
            "path_error": list(dyngp_path_err_no_pdaf),
            "nees": list(dyngp_nees_no_pdaf)
        })
        dyngp_no_pdaf_df["method"] = "dyngp_no_pdaf"
        dyngp_no_pdaf_df["y_source"] = "finite_difference"
        dyngp_no_pdaf_df["train_samples"] = train_x.shape[0]


        cog_dyngp_pdaf_df = pd.DataFrame({
            "timestep": dyngp_test_x[:, -1],
            "ground_truth": list(dyngp_test_x),
            "prediction": list(cog_dyngp_traj_subset_pdaf),
            "trajectory_error": list(cog_dyngp_traj_err_pdaf),
            "path_error": list(cog_dyngp_path_err_pdaf),
            "nees": list(cog_dyngp_nees_pdaf)
        })
        cog_dyngp_pdaf_df["method"] = "dyngp_pdaf"
        cog_dyngp_pdaf_df["y_source"] = "cog"
        cog_dyngp_pdaf_df["train_samples"] = train_x.shape[0]

        cog_dyngp_pdaf_no_P_df = pd.DataFrame({
            "timestep": dyngp_test_x[:, -1],
            "ground_truth": list(dyngp_test_x),
            "prediction": list(cog_dyngp_traj_subset_pdaf_no_P),
            "trajectory_error": list(cog_dyngp_traj_err_pdaf_no_P),
            "path_error": list(cog_dyngp_path_err_pdaf_no_P),
            "nees": list(cog_dyngp_nees_pdaf_no_P)
        })
        cog_dyngp_pdaf_no_P_df["method"] = "dyngp_pdaf_no_P"
        cog_dyngp_pdaf_no_P_df["y_source"] = "cog"
        cog_dyngp_pdaf_no_P_df["train_samples"] = train_x.shape[0]

        cog_dyngp_syn_df = pd.DataFrame({
            "timestep": dyngp_test_x[:, -1],
            "ground_truth": list(dyngp_test_x),
            "prediction": list(cog_dyngp_traj_subset_syn),
            "trajectory_error": list(cog_dyngp_traj_err_syn),
            "path_error": list(cog_dyngp_path_err_syn),
            "nees": list(cog_dyngp_nees_syn)
        })
        cog_dyngp_syn_df["method"] = "dyngp_syn"
        cog_dyngp_syn_df["y_source"] = "cog"
        cog_dyngp_syn_df["train_samples"] = train_x.shape[0]

        cog_dyngp_syn_no_P_df = pd.DataFrame({
            "timestep": dyngp_test_x[:, -1],
            "ground_truth": list(dyngp_test_x),
            "prediction": list(cog_dyngp_traj_subset_syn_no_P),
            "trajectory_error": list(cog_dyngp_traj_err_syn_no_P),
            "path_error": list(cog_dyngp_path_err_syn_no_P),
            "nees": list(cog_dyngp_nees_syn_no_P)
        })
        cog_dyngp_syn_no_P_df["method"] = "dyngp_syn_no_P"
        cog_dyngp_syn_no_P_df["y_source"] = "cog"
        cog_dyngp_syn_no_P_df["train_samples"] = train_x.shape[0]

        cog_dyngp_no_pdaf_df = pd.DataFrame({
            "timestep": dyngp_test_x[:, -1],
            "ground_truth": list(dyngp_test_x),
            "prediction": list(cog_dyngp_traj_subset_no_pdaf),
            "trajectory_error": list(cog_dyngp_traj_err_no_pdaf),
            "path_error": list(cog_dyngp_path_err_no_pdaf),
            "nees": list(cog_dyngp_nees_no_pdaf)
        })
        cog_dyngp_no_pdaf_df["method"] = "dyngp_no_pdaf"
        cog_dyngp_no_pdaf_df["y_source"] = "cog"
        cog_dyngp_no_pdaf_df["train_samples"] = train_x.shape[0]

        pos_gp_df = pd.DataFrame({
            "timestep": posgp_test_x[:, -1],
            "ground_truth": list(posgp_test_y),
            "prediction": list(pos_gp_traj_subset),
            "trajectory_error": list(pos_gp_traj_err),
            "path_error": list(pos_gp_path_err),
            "nees": list(pos_gp_nees)
        })
        pos_gp_df["method"] = "posgp"
        pos_gp_df["y_source"] = "position"
        pos_gp_df["train_samples"] = posgp_train_x.shape[0]



        temp_df = pd.concat([
            cvm_df,
            dyngp_pdaf_df,
            dyngp_pdaf_no_P_df,
            dyngp_syn_df,
            dyngp_syn_no_P_df,
            dyngp_no_pdaf_df,
            cog_dyngp_pdaf_df,
            cog_dyngp_pdaf_no_P_df,
            cog_dyngp_syn_df,
            cog_dyngp_syn_no_P_df,
            cog_dyngp_no_pdaf_df,
            pos_gp_df
        ])
        temp_df["test_id"] = i

        plt.close()
        dgp_plot.kalman_figure(dgp, res_pdaf, dyngp_test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/raw/run_{i}_pos_pdaf.pdf", format="pdf", bbox_inches='tight')
        plt.close()

        plt.clf()
        dgp_plot.kalman_state_figure(dgp, res_pdaf, dyngp_test_x, label="GP")
        plt.savefig(f"{sim_name}/raw/run_{i}_state_pdaf.pdf", format="pdf", bbox_inches='tight')
        plt.close()

        plt.clf()
        dgp_plot.kalman_figure(dgp, res_pdaf_no_P, dyngp_test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/raw/run_{i}_pos_pdaf_no_P.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        plt.clf()
        dgp_plot.kalman_state_figure(dgp, res_pdaf_no_P, dyngp_test_x, label="GP")
        plt.savefig(f"{sim_name}/raw/run_{i}_state_pdaf_no_P.pdf", format="pdf", bbox_inches='tight')
        plt.close()

        
        plt.clf()
        dgp_plot.kalman_figure(dgp, res_syn, dyngp_test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/raw/run_{i}_pos_syn.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        plt.clf()
        dgp_plot.kalman_state_figure(dgp, res_syn, dyngp_test_x, label="GP")
        plt.savefig(f"{sim_name}/raw/run_{i}_state_syn.pdf", format="pdf", bbox_inches='tight')
        plt.close()

        plt.clf()
        dgp_plot.kalman_figure(dgp, res_syn_no_P, dyngp_test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/raw/run_{i}_pos_syn_no_P.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        plt.clf()
        dgp_plot.kalman_state_figure(dgp, res_syn_no_P, dyngp_test_x, label="GP")
        plt.savefig(f"{sim_name}/raw/run_{i}_state_syn_no_P.pdf", format="pdf", bbox_inches='tight')
        plt.close()

        plt.clf()
        dgp_plot.kalman_figure(dgp, res, dyngp_test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/raw/run_{i}_pos_no_pdaf.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        plt.clf()
        dgp_plot.kalman_state_figure(dgp, res, dyngp_test_x, label="GP")
        plt.savefig(f"{sim_name}/raw/run_{i}_state_no_pdaf.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        
        plt.clf()
        dgp_plot.kalman_figure(cog_dgp, cog_res_pdaf, dyngp_test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/raw/run_{i}_pos_pdaf_cog.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        plt.clf()
        dgp_plot.kalman_state_figure(cog_dgp, cog_res_pdaf, dyngp_test_x, label="GP")
        plt.savefig(f"{sim_name}/raw/run_{i}_state_pdaf_cog.pdf", format="pdf", bbox_inches='tight')
        plt.close()

        plt.clf()
        dgp_plot.kalman_figure(cog_dgp, cog_res_pdaf_no_P, dyngp_test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/raw/run_{i}_pos_pdaf_no_P_cog.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        plt.clf()
        dgp_plot.kalman_state_figure(cog_dgp, cog_res_pdaf_no_P, dyngp_test_x, label="GP")
        plt.savefig(f"{sim_name}/raw/run_{i}_state_pdaf_no_P_cog.pdf", format="pdf", bbox_inches='tight')
        plt.close()

        plt.clf()
        dgp_plot.kalman_figure(cog_dgp, cog_res_syn, dyngp_test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/raw/run_{i}_pos_syn_cog.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        plt.clf()
        dgp_plot.kalman_state_figure(cog_dgp, cog_res_syn, dyngp_test_x, label="GP")
        plt.savefig(f"{sim_name}/raw/run_{i}_state_syn_cog.pdf", format="pdf", bbox_inches='tight')
        plt.close()

        plt.clf()
        dgp_plot.kalman_figure(cog_dgp, cog_res_syn_no_P, dyngp_test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/raw/run_{i}_pos_syn_no_P_cog.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        plt.clf()
        dgp_plot.kalman_state_figure(cog_dgp, cog_res_syn_no_P, dyngp_test_x, label="GP")
        plt.savefig(f"{sim_name}/raw/run_{i}_state_syn_no_P_cog.pdf", format="pdf", bbox_inches='tight')
        plt.close()

        plt.clf()
        dgp_plot.kalman_figure(cog_dgp, cog_res, dyngp_test_x, label="GP", plot_P=True)
        plt.savefig(f"{sim_name}/raw/run_{i}_pos_no_pdaf_cog.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        plt.clf()
        dgp_plot.kalman_state_figure(cog_dgp, cog_res, dyngp_test_x, label="GP")
        plt.savefig(f"{sim_name}/raw/run_{i}_state_no_pdaf_cog.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        
        print("Completed", i, "with stats:")
        #print(temp_df)
    except Exception as e:
        print(e)
        return None
    return temp_df.copy()

with Pool(8) as pool:
    results_df = pool.map(run, list(range(msk.shape[0])))
print(results_df)

out = pd.concat(results_df)
out.to_pickle(f"{sim_name}.pkl")

