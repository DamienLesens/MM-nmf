import numpy as np
from matplotlib import pyplot as plt
import NMF_Frobenius as nmf_f 
from nn_fac.nmf import nmf
#import tensorly as tl #perso branch
#from tensorly.decomposition import non_negative_parafac_hals
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import plotly.io as pio
import time
pio.kaleido.scope.mathjax = None
from utils import opt_scaling_fro, sparsify

# Personnal comparison toolbox
# you can get it at 
# https://github.com/cohenjer/shootout
from shootout.methods.runners import run_and_track
import shootout.methods.post_processors as pp
from shootout.methods.plotters import line, rename_axis

plt.close('all')
# --------------------- Choose parameters for grid tests ------------ #
if len(sys.argv)==1 or int(sys.argv[1])==0:
    seeds = [] #no run
    skip=True
else:
    seeds = list(np.arange(int(sys.argv[1])))
    skip=False

variables = {
    "add_track": {"distribution" : "uniform"},
    "mnr": [[1000, 400, 20]],
    "NbIter": [200],  # for Lee and Seung also
    "NbIter_HALS": [100],
    "NbIter_inner": 10,
    "setup": ["dense", "sparse"],
    "unbalanced_scale": [0, -3],  # SNR is incorrect ?
    "SNR": [100],
    "delta": 0,
    "seed": seeds,
    "distribution": "uniform",
    "show_it": 100,
    "epsilon": 1e-8,
    "tol": 0
}

name = "l2_run-27-01-2026_unbalanced_sparse"
#algs = ["AMU", "APGD", "ANeNMF", "AHALS", "AmSOM", "AMUSOM"]
algs = ["AHALS", "AMU", "AMUSOM", "ANeNMF", "APGD", "AmSOM"]

@run_and_track(algorithm_names=algs, path_store="Results/", name_store=name,
                skip=skip, **variables)
def one_run(**cfg):
    m, n, r = cfg["mnr"]
    # Fixed the signal 
    rng = np.random.RandomState(cfg["seed"]+20)
    Worig = rng.rand(m, r) 
    Horig = rng.rand(r, n)  
    Vorig = Worig.dot(Horig)
    
    match cfg["setup"]:
        case "dense":  # Dense
            # Rescaling the components to degrade the conditionning of the problem
            Worig = Worig * np.logspace(0, cfg["unbalanced_scale"], r)[np.newaxis, :]
            # Data generation            
            Vorig = Worig.dot(Horig)  # densified
        case "sparse":  # sparse factors and data
            Worig = sparsify(Worig, s=0.5, epsilon=cfg["epsilon"])
            Horig = sparsify(Horig, s=0.5, epsilon=cfg["epsilon"])
            # Rescaling the components to degrade the conditionning of the problem
            Worig = Worig * np.logspace(0, cfg["unbalanced_scale"], r)[np.newaxis, :]
            # Data generation
            Vorig = Worig.dot(Horig)  #+ 0.1 # densified

    # prints
    verbose = True
    
    # adding Gaussian noise to the observed data
    N = rng.randn(m,n)
    sigma = 10**(-cfg["SNR"]/20)*np.linalg.norm(Vorig)/np.linalg.norm(N)
    V = Vorig + sigma*N

    # Initialization for H0 as a random matrix
    Hini = rng.rand(r, n)
    Wini = rng.rand(m, r)  #sparse.random(rV, cW, density=0.25).toarray() 
    Hini = opt_scaling_fro(V, Wini@Hini)*Hini
    
    # AHALS
    # Fewer max iter because too slow
    _, _, error0, toc0, cnt0 = nmf(V, r, init="random", n_iter_max=cfg["NbIter_HALS"], tol=cfg["tol"], return_costs=True, NbIter_inner=cfg["NbIter_inner"], delta=cfg["delta"], verbose=verbose)
    
    # AMU
    error1, W1, H1, toc1, cnt1 = nmf_f.NMF_Lee_Seung(V,  Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"],tol=cfg["tol"], legacy=False, delta=cfg["delta"], verbose=verbose)
    
    # AMUSOM 
    error2, W2, H2, toc2, cnt2 = nmf_f.NMF_proposed_Frobenius(V, Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"], tol=cfg["tol"], delta=cfg["delta"], verbose=verbose, gamma=1.9, method="AMUSOM")
    
    # ANeNMF
    error3, W3, H3, toc3, cnt3  = nmf_f.NeNMF(V, Wini, Hini, tol=cfg["tol"], nb_inner=cfg["NbIter_inner"], itermax=cfg["NbIter"], delta=cfg["delta"], verbose=verbose)
    
    # APGD
    error4, W4, H4, toc4, cnt4  = nmf_f.Grad_descent(V , Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"], tol=cfg["tol"], delta=cfg["delta"], verbose=verbose)

    # AmSOM
    error5, W5, H5, toc5, cnt5 = nmf_f.NMF_proposed_Frobenius(V, Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"], tol=cfg["tol"], delta=cfg["delta"], verbose=verbose, gamma=1.9, method="AmSOM")

    #   algs = ["MU_Fro","fastMU_Fro_ex","GD_Fro", "NeNMF_Fro", "HALS", "fastMU_Fro", "trueMU_Fro"]
    return {"errors" : [error0, error1, error2, error3, error4, error5], 
            "timings" : [toc0, toc1, toc2, toc3, toc4, toc5],
            "cnt" : [cnt0[::10], cnt1[::10], cnt2[::10], cnt3[::10], cnt4[::10], cnt5[::10]]
            }


# -------------------- Post-Processing ------------------- #
pio.templates.default= "plotly_white"
scale, scale_text = 1.5, 2.2  # size of the plots
quantile = 0.75  # for the error bars, 0.5 is median, 1 is all
threshold_points = 50
variables_plot = ["setup", "unbalanced_scale"]
n_cols = 2  # for the threshold plots
meanplot = "median"  # TODO typical that works

df = pd.read_pickle("Results/"+name)

# Check that algs are correctly defined
algorithms = df["algorithm"][:len(algs)]

# ----------- Performance profiles for all setups and SNRs -------------- #
fig = pp.performance_profiles(df, variables=variables_plot, n_cols=n_cols, threshold_points=threshold_points, algorithms=algorithms)

# Update layout
fig.update_layout(
    title="Synthetic Frobenius NMF unbalanced sparse, winner profiles",
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
# Change unbalanced_scale to scale in annotations
for ann in fig.layout.annotations:
    if ann.text.find("unbalanced_scale")!=-1:
        ann.text = ann.text.replace("unbalanced_scale","scale")

# Interpolating time (choose fewer points for better vis), adaptive grid since time varies across plots
ovars_interp =  ["setup", "unbalanced_scale", "algorithm"]
df = pp.interpolate_time_and_error(df, npoints=df["NbIter"][0], adaptive_grid=True, groups=ovars_interp)

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
ovars = ["setup", "unbalanced_scale", "seed"]
df_conv = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars, err_name="errors_interp", time_name="timings_interp")
df_conv = df_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
df_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars)

# Median plot
df_conv_median_time = pp.median_convergence_plot(df_conv, type_x="timings", mean=meanplot, quantile=quantile)
df_conv_median_it = pp.median_convergence_plot(df_conv, type_x="iterations", mean=meanplot, quantile=quantile)

# Merge SNR and mnr in one column for facetting
df_conv_median_time["setup"] = df_conv_median_time["setup"] + ", scale=" + df_conv_median_time["unbalanced_scale"].astype(str)


# Convergence plots with all runs
pxfig = line(
            data_frame=df_conv_median_time,
            x="timings", 
            y= "errors", 
            color='algorithm',
            #line_dash='algorithm',
            facet_col="setup",
            #facet_row="SNR",
            facet_col_wrap=2,
            log_y=True,
            log_x=True,
            facet_col_spacing=0.12,
            facet_row_spacing=0.2,
            #log_x=True,
            error_y_mode="band",
            error_y="q_errors_p", 
            error_y_minus="q_errors_m", 
            category_orders={
                "algorithm": algs,
                "setup": ["dense, scale=-3", "dense, scale=0", "sparse, scale=-3", "sparse, scale=0"]
                }
)
# Final touch
pxfig.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfig.update_layout(
    font_size = 8*scale_text,
    width=450*scale, # in px
    height=350*scale
)

pxfig.update_xaxes(
    matches = None,
    #showticklabels = True
)
pxfig.update_yaxes(
    matches=None,
    showticklabels=True
)

# updating titles
#for ann in pxfig.layout.annotations:
    #if ann.text[:3]=="mnr":
        #ann.text = "[M,N,R]=["+ ", ".join(re.findall(r"\((\d+)\)", ann.text)) + "]" + ann.text[ann.text.find(", SNR="):]

# Final touch
rename_axis(pxfig, scale=scale_text, xtext="Time (s)", ytext="n. Loss")
pxfig.layout.title.text = "Synthetic Frobenius NMF unbalanced sparse, convergence plots"
pxfig.layout.title.font.size = 9*scale_text
pxfig.update_layout(margin_t=100)


# Convergence plots with all runs
pxfigit = line(
            data_frame=df_conv_median_it,
            x="it", 
            y= "errors", 
            color='algorithm',
            #line_dash='algorithm',
            facet_col="setup",
            facet_row="unbalanced_scale",
            log_y=True,
            facet_col_spacing=0.12,
            facet_row_spacing=0.2,
            #log_x=True,
            error_y_mode="band",
            error_y="q_errors_p", 
            error_y_minus="q_errors_m", 
)
# Final touch
pxfigit.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfigit.update_layout(
    font_size = 8*scale_text,
    width=450*scale, # in px
    height=350*scale,
    #xaxis1=dict(range=[0,3],title_text="Time (s)"),
    #xaxis2=dict(range=[0,0.2],title_text="Time (s)"),
    yaxis1=dict(title_text="n. Loss"),
    yaxis2=dict(title_text=""),
    yaxis3=dict(title_text="n. Loss"),
    yaxis4=dict(title_text="")
)

pxfigit.update_xaxes(
    matches = None,
    #showticklabels = True
)
pxfigit.update_yaxes(
    matches=None,
    showticklabels=True
)
# updating titles
#for i,ann in enumerate(pxfigit.layout.annotations):
    #if ann.text[:3]=="mnr":
        #ann.text="[M,N,R]"+ann.text[3:] 
pxfig.write_image("Results/"+name+".pdf")
pxfig.write_image("Results/"+name+".pdf")
pxfigit.write_image("Results/"+name+"_it.pdf")
fig.write_image("Results/"+name+"_performance.pdf")
fig.show()
pxfig.show()
#pxfigit.show()