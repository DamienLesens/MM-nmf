import numpy as np
import NMF_KL as nmf_kl
import plotly.express as px
import pandas as pd
import scipy.io
from shootout.methods.runners import run_and_track
from shootout.methods import post_processors as pp
from shootout.methods.post_processors import df_to_convergence_df, interpolate_time_and_error, median_convergence_plot
from shootout.methods.plotters import line, rename_axis
import sys
import plotly.io as pio
from utils import opt_scaling
pio.kaleido.scope.mathjax = None
pio.templates.default= "plotly_white"


'''
    We load an hyperspectral image called Urban. It has 162 clean spectral bands, and 307x307 pixels. We also load a set of good endmembers considered as ``Ground Truth'' (rank=6 spectra), and we define subsets of the image that are likely to contain pure pixels.

    For the NNLS part, we can use either W as the ground truth (tall matrix) or candidates pixels (if more than 162, fat matrix), and estimate abundances H that should be very sparse.

    For the NMF, we estimate both W and H, and plot the results. In this experiment, the Frobenius norm is well adapted.
    
    LONG RUNTIME --> cut it ?
'''

#-------------------------------------------------------------------------
# Data import and preprocessing

# Loading the data
dico = scipy.io.loadmat('./data_and_scripts/Urban.mat')

# dict is a python dictionnary. It contains the matrix we want to NMF
M = np.transpose(dico['A']) # permutation because we like spectra in W
m, n = M.shape

# It can be nice to normalize the data
M = M/np.max(M)
#M = M/np.linalg.norm(M, 'fro')

# Ground truth import
# https://gitlab.com/nnadisic/giant.jl/-/blob/master/xp/data/Urban_Ref.mat
Wref = scipy.io.loadmat('./data_and_scripts/Urban_Ref.mat')
Wref = np.transpose(Wref['References'])

# ground truth rank is 6
# for good init
#Href = fista(Wref.T@M,Wref.T@Wref,tol=1e-16,n_iter_max=500)


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Solving with nonnegative least squares
# from tensorly.tenalg.proximal import fista
if len(sys.argv) == 1 or int(sys.argv[1]) == 0:
    seeds = []  # no run
    skip = True
else:
    seeds = list(np.arange(int(sys.argv[1])))
    skip = False

variables = {
    "NbIter" : 800,
    "NbIter_inner" : 10,
    "delta" : 0,
    "epsilon" : 1e-16,
    "seed" : seeds,
    "tol" : 0,
    "NbIter_SN": 100,  #un peu long
    "NbIter_inner_SN": 5,
}

algs = ["AMU", "AMUSOM", "ASN CCD", "AmSOM"]
#algs = ["fastMU_Fro", "fastMU_Fro_ex", "GD_Fro", "NeNMF_Fro", "MU_Fro", "HALS", "MU_KL", "fastMU_KL", "trueMU"]
#algs = ["AMU", "AmSOM", "AMUSOM", "APGD", "NeNMF", "AHALS"]

name = "hsi_nmf_kl_19-01-2026"

@run_and_track(
    algorithm_names=algs, 
    path_store="Results/",
    name_store=name,
    skip=skip,
    **variables
)
def one_run(rank = 6,
            NbIter = 200,
            NbIter_inner = 10,
            NbIter_SN = 50,
            NbIter_inner_SN = 50,
            delta = 0,
            epsilon = 1e-16,
            tol=0,
            seed=1
            ):
    # Print
    verbose=True
    # Seeding
    rng = np.random.RandomState(seed+20)
    # Init
    Wini = Wref + 0.1*np.random.rand(m, rank) # TODO le mentionner
    Hini = rng.rand(rank, n)
    lamb = opt_scaling(M, Wini@Hini)
    Hini = lamb*Hini
   
    # Init with one step of Lee-Seung KL to avoid bad inits
    _, Wini, Hini, _, _ = nmf_kl.Lee_Seung_KL(M, Wini, Hini, NbIter=1, nb_inner=NbIter_inner, tol=0, verbose=verbose, epsilon=epsilon, print_it=1)

    # KL algorithms 
    error7, W7, H7, toc7, cnt7 = nmf_kl.Lee_Seung_KL(M, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, epsilon=epsilon, print_it=20)
    error8, W8, H8, toc8, cnt8 = nmf_kl.Proposed_KL(M, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, gamma=1.9, epsilon=epsilon, method="AMUSOM", print_it=20)
    error9, W9, H9, toc9, cnt9 = nmf_kl.ScalarNewton(M, Wini, Hini, NbIter=NbIter_SN, nb_inner=NbIter_inner_SN, tol=tol, verbose=verbose,  epsilon=epsilon, method="CCD", print_it=5)
    error10, W10, H10, toc10, cnt10 = nmf_kl.Proposed_KL(M, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, gamma=1.9, epsilon=epsilon, print_it=20)


    return {
        "errors": [error7, error8, error9, error10],
        "timings": [toc7, toc8, toc9, toc10],
    }


# -------------------- Post-Processing ------------------- #
df = pd.read_pickle("Results/"+name)
scale, scale_text = 1.5, 3.1  # size of the plots
quantile = 0.5  # for the error bars, 0.5 is median, 1 is all
threshold_points = 50
variables_plot = []
n_cols = 1  # for the threshold plots
meanplot = "min"  # TODO typical that works

# ------------- Performance plots -------
algorithms = df["algorithm"][:len(algs)]

fig = pp.performance_profiles(df, variables=variables_plot, n_cols=n_cols, threshold_points=threshold_points, algorithms=algorithms)

fig.update_layout(
    title="HSI KL NMF, winner profiles",
    xaxis_type="log",
    template="plotly_white",
    font_size=8*scale_text,
    height=350*scale,  # adjust figure height
    width=450*scale,            # adjust figure width
    # when next to conv plot
    showlegend=False,
    title_font_size=9*scale_text,
)
fig.update_annotations(font_size=8*scale_text)
fig.show()
# interpolation
ovars_iterp = ["algorithm"]
df = interpolate_time_and_error(df, npoints=df["NbIter"][0], adaptive_grid=True, groups=ovars_iterp)

# Making a convergence plot dataframe
ovars=["seed"]
df_kl_conv = df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars, err_name="errors_interp", time_name="timings_interp")
df_kl_conv = df_kl_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})

df_kl_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars)
df_kl_conv_median_time = median_convergence_plot(df_kl_conv, type_x="timings", mean=meanplot, quantile=quantile)
df_kl_conv_median_it = pp.median_convergence_plot(df_kl_conv_it, type_x="iterations", mean=meanplot, quantile=quantile)

# dirty hack to get the naming of algorithms in the same order as the NLS problem
# plotly takes the order of the algorithms as they appear in the dataframe, so we reorder them
# we swap row 5 with row 3, 3 with 4 and 4 with 5
#a = df_kl_conv_median_time
#b, c, d = a.iloc[3], a.iloc[4], a.iloc[5]

#temp1 = b.copy()
#temp2 = c.copy()
#a.iloc[3] = d
#a.iloc[4] = b
#a.iloc[5] = c


# ----------------------- Plot --------------------------- #
# Convergence plots with all runs
pxfig = line(
            data_frame=df_kl_conv_median_time,
            #line_group="groups",
            x="timings",
            y="errors",
            color='algorithm',
            #line_dash='algorithm',
            log_y=True,
            log_x=True,
            error_y_mode="band",
            error_y="q_errors_p", 
            error_y_minus="q_errors_m", 
            category_orders={
                "algorithm": algs
                }
            )

pxfigit = line(
            data_frame=df_kl_conv_median_it,  #_median_it, 
            x="it", 
            y="errors", 
            color='algorithm',
            #line_dash='algorithm',
            log_y=True,
            #line_group="groups",
            error_y_mode="band",
            error_y="q_errors_p", 
            error_y_minus="q_errors_m"
)

#rename_axis(pxfig, scale=scale, xtext="Time (s)", ytext="Loss")
pxfig.layout.title.text = "HSI KL NMF, convergence plots"
pxfig.layout.title.font.size = 9*scale_text
pxfig.update_layout(margin_t=100)

# Final touch
pxfig.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfig.update_layout(
    font_size = 8*scale_text,
    width=450*scale, # in px
    height=350*scale,
    xaxis=dict(title_text="Time (s)"),
    yaxis=dict(title_text="Loss")
)

pxfig.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfig.update_yaxes(
    matches=None,
    showticklabels=True
)

pxfigit.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfigit.update_layout(
    font_size = 8*scale_text,
    width=450*scale, # in px
    height=350*scale,
    #xaxis=dict(range=[0,1.0], title_text="Time (s)"),
    #yaxis=dict(title_text="Fit")
)

pxfigit.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfigit.update_yaxes(
    matches=None,
    showticklabels=True
)
pxfig.write_image("Results/"+name+".pdf")
pxfig.write_image("Results/"+name+".pdf")
#pxfigit.write_image("Results/"+name+"_it.pdf")
fig.write_image("Results/"+name+"_performance.pdf")
pxfig.show()
#pxfigit.show()