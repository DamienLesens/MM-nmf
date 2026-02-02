from importlib.resources import path
import numpy as np
from scipy.linalg import hadamard
import NMF_Frobenius as nmf_f 
import NMF_KL as nmf_kl
import matplotlib.pyplot as plt
import nn_fac
import pandas as pd
import soundfile as sf
from scipy import signal
import plotly.express as px
# personal toolbox
from shootout.methods.runners import run_and_track
import shootout.methods.post_processors as pp
from shootout.methods.plotters import line, rename_axis
import sys
from utils import opt_scaling_fro
import plotly.io as pio
pio.kaleido.scope.mathjax = None
pio.templates.default= "plotly_white"
'''
 W is a dictionary of 88 columns and 4097 frequency bins. Each column was obtained by performing a rank-one NMF (todo correct) on the recording of a single note in the MAPS database, on a Yamaha Disklavier with close microphones, the note was played mezzo forte and the loss was beta-divergence with beta=1.

 By performing matrix NNLS from the Power STFT Y of 30s of a (relatively simple) song in MAPS, recorded with the same piano in similar conditions and processes in the same way, we expect that $Y \approx WH$, where H are the activations of each note in the recording. A good loss to measure discrepencies here is the beta-divergence with beta=1.

 For the purpose of this toy experiment, only one song from MAPS is selected. We then perform NMF, and look at the activations as a piano roll.

 For the NMF part, we simply discard the provided templates, and estimate both the templates and the activations. Again it is best to use KL divergence. We can initialize with the provided template to get a initial dictionary.
'''

#-------------------------------------------------------------------------
# Modeling/audio data

# Importing data and computing STFT using the Attack-Decay paper settings
# Read the song (you can use your own!)
the_signal, sampling_rate_local = sf.read('./data_and_scripts/MAPS_MUS-bach_846_AkPnBcht.wav')
# Using the settings of the Attack-Decay transcription paper
the_signal = the_signal[:,0] # left channel only
frequencies, time_atoms, Y = signal.stft(the_signal, fs=sampling_rate_local, nperseg=4096, nfft=8192, noverlap=4096 - 882)
time_step = time_atoms[1] #20 ms
freq_step = frequencies[1] #5.3 hz
# Taking the amplitude spectrogram
Y = np.abs(Y)
# Cutting silence, end song and high frequencies (>5300 Hz)
cutf = 1000 
cutt_in = int(1/time_step) # song beginning after 1 second
cutt_out = int(30/time_step)# 30seconds with 20ms steps #time_atoms.shape[0]
Y = Y[:cutf, cutt_in:cutt_out]
# normalization
Y = Y/np.max(Y)  # does not change much

df = pd.DataFrame()

if len(sys.argv)==1 or int(sys.argv[1])==0:
    seeds = []  # no run
    skip=True
else:
    seeds = list(np.arange(int(sys.argv[1])))
    skip=False

# TODO More iterations? show all runs ?
# TODO sigma larger ?
variables = {
    "NbIter": 100,
    "NbIter_inner": 10,
    "delta": 0,
    "epsilon": 1e-16,
    "rank": [2, 11, 23, 45],
    "seed": seeds,
    "sigma": 0.1,  # 1 ?
    "cutf": cutf,
    "tol": 0
}

#name = "audio_test_01-06-2024"
name = "audio_fro_19-01-2026"

#algs = ["AmSOM", "APGD", "NeNMF", "AMU", "HALS", "AMU_kl", "AmSOM_kl", "AMUSOM_kl", "ASN CCD"]
#algs = ["AMU", "AmSOM", "AMUSOM", "ASN CCD"]
#algs = ["AMU", "AmSOM", "AMUSOM", "APGD", "NeNMF", "AHALS"]
algs = ["AHALS", "AMU", "AMUSOM", "ANeNMF", "APGD", "AmSOM"]

# TODO: better error message when algs dont match

@run_and_track(
    algorithm_names=algs, 
    path_store="Results/",
    name_store=name,
    skip=skip,
    **variables
)
def one_run(**cfg):
    # Print
    verbose=True
    print_it=10
    # Importing a good dictionnary for the NNLS part
    Wgt = np.load('./data_and_scripts/attack_dict_piano_AkPnBcht_beta_1_stftAD_True_intensity_M.npy')
    Wgt = Wgt[:cfg["cutf"], :]

    Wgt = Wgt[:,27:(27+cfg["rank"])] # octaves in the middle
    # Normalization by max
    Wgt = Wgt/np.max(Wgt, axis=0)

    #------------------------------------------------------------------
    # Computing the NMF to try and recover activations and templates
    m, n = Y.shape
    
    # Run options
    print_it = 10
    
    # Perturbing the initialization for randomization
    rng = np.random.RandomState(cfg["seed"]+20)
    Wini = Wgt + cfg["sigma"]*rng.rand(m, cfg["rank"])
    Hini = rng.rand(cfg["rank"], n)
    lamb = opt_scaling_fro(Y, Wini@Hini)
    Hini = lamb*Hini
        
    # AHALS
    # Fewer max iter because too slow
    _, _, error0, toc0, cnt0 = nn_fac.nmf.nmf(Y, cfg["rank"], init="random", n_iter_max=cfg["NbIter"], tol=cfg["tol"], return_costs=True, NbIter_inner=cfg["NbIter_inner"], delta=cfg["delta"], verbose=verbose, print_it=print_it)
    
    # AMU
    error1, W1, H1, toc1, cnt1 = nmf_f.NMF_Lee_Seung(Y,  Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"],tol=cfg["tol"], legacy=False, delta=cfg["delta"], verbose=verbose, print_it=print_it, epsilon=cfg["epsilon"])
    
    # AMUSOM 
    error2, W2, H2, toc2, cnt2 = nmf_f.NMF_proposed_Frobenius(Y, Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"], tol=cfg["tol"], delta=cfg["delta"], verbose=verbose, gamma=1.9, method="AMUSOM", print_it=print_it, epsilon=cfg["epsilon"])
    
    # ANeNMF
    error3, W3, H3, toc3, cnt3  = nmf_f.NeNMF(Y, Wini, Hini, tol=cfg["tol"], nb_inner=cfg["NbIter_inner"], itermax=cfg["NbIter"], delta=cfg["delta"], verbose=verbose, print_it=print_it, epsilon=cfg["epsilon"])
    
    # APGD
    error4, W4, H4, toc4, cnt4  = nmf_f.Grad_descent(Y , Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"], tol=cfg["tol"], delta=cfg["delta"], verbose=verbose, print_it=print_it, epsilon=cfg["epsilon"])

    # AmSOM
    error5, W5, H5, toc5, cnt5 = nmf_f.NMF_proposed_Frobenius(Y, Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"], tol=cfg["tol"], delta=cfg["delta"], verbose=verbose, gamma=1.9, method="AmSOM", print_it=print_it, epsilon=cfg["epsilon"])

    return {"errors" : [error0, error1, error2, error3, error4, error5], 
            "timings" : [toc0, toc1, toc2, toc3, toc4, toc5],
            "cnt" : [cnt0[::10], cnt1[::10], cnt2[::10], cnt3[::10], cnt4[::10], cnt5[::10]]
            }


    #error0, W0, H0, toc0, cnt0 = nmf_f.NMF_Lee_Seung(Y,  Wini, Hini, NbIter, NbIter_inner,tol=tol, legacy=False, epsilon=epsilon, verbose=verbose, delta=delta, print_it=print_it)   
    #error1, W1, H1, toc1, cnt1 = nmf_f.NMF_proposed_Frobenius(Y, Wini, Hini, NbIter, NbIter_inner, tol=tol, delta=delta, verbose=verbose, print_it=print_it, gamma=1.9)
    #error2, W2, H2, toc2, cnt2 = nmf_f.NMF_proposed_Frobenius(Y, Wini, Hini, NbIter, NbIter_inner, tol=tol, method="AMUSOM", delta=delta, verbose=verbose, print_it=print_it, gamma=1.9)
    ##error3, W3, H3, toc3, cnt3  = nmf_f.NeNMF_optimMajo(M, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner, epsilon=epsilon, verbose=verbose, delta=delta, print_it=print_it, gamma=1)
    #error4, W4, H4, toc4, cnt4  = nmf_f.Grad_descent(Y, Wini, Hini, NbIter, NbIter_inner, tol=tol, epsilon=epsilon, verbose=verbose, delta=delta, print_it=print_it)
    #error5, W5, H5, toc5, cnt5  = nmf_f.NeNMF(Y, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter, epsilon=epsilon, verbose=verbose, delta=delta, print_it=print_it)
    #W6, H6, error6, toc6, cnt6 = nn_fac.nmf.nmf(Y, rank, init="custom", U_0=np.copy(Wini), V_0=np.copy(Hini), n_iter_max=NbIter, tol=tol, update_rule='hals', beta=2, return_costs=True, NbIter_inner=NbIter_inner, verbose=verbose, delta=delta, print_it=print_it)

    #return {
        #"errors": [error0, error1, error2, error4, error5, error6], #, error7, error8, error9, error10],
        #"timings": [toc0, toc1, toc2, toc4, toc5, toc6],
        ##"loss": 5*["l2"]+4*["kl"],
            #}
    

# ----------- Results -----------
df = pd.read_pickle("Results/"+name)
scale, scale_text = 1.5, 2.2  # size of the plots
quantile = 0.3  # for the error bars, 0.5 is median, 1 is all
threshold_points = 50
variables_plot = ["rank"]
n_cols = 2  # for the threshold plots
meanplot = "min"  # TODO typical that works

# ------------- Performance plots -------
algorithms = df["algorithm"][:len(algs)]

fig = pp.performance_profiles(df, variables=variables_plot, n_cols=n_cols, threshold_points=threshold_points, algorithms=algorithms)

fig.update_layout(
    title="Audio Frobenius NMF, winner profiles",
    xaxis_type="log",
    template="plotly_white",
    font_size = 8*scale_text,
    height=350*scale,  # adjust figure height
    width=450*scale,   # adjust figure width
    # when next to conv plot
    showlegend=False,
    title_font_size=10*scale_text,
)
fig.update_annotations(font_size=8*scale_text)
fig.show()
# Remove extrapolation
#df = df[df["algorithm"] != "fastMU_Fro_ex"]

ovars_iterp = ["algorithm", "rank"]
df = pp.interpolate_time_and_error(df, npoints=df["NbIter"][0], adaptive_grid=True, groups=ovars_iterp)

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
#df_l2_conv = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               #filters={"loss":"l2"}, err_name="errors_interp", time_name="timings_interp")
#df_l2_conv = df_l2_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
#df_l2_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               #filters={"loss":"l2"})
ovars = ["rank", "seed"]
df_fro_conv = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars,
                               err_name="errors_interp", time_name="timings_interp", exclude_zero=True)
df_fro_conv = df_fro_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
df_fro_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars)

#df_l2_conv_median_time = pp.median_convergence_plot(df_l2_conv, type_x="timings")
df_fro_conv_median_time = pp.median_convergence_plot(df_fro_conv, type_x="timings", mean=meanplot, quantile=quantile)
#df_l2_conv_median_it = pp.median_convergence_plot(df_l2_conv_it)
df_fro_conv_median_it = pp.median_convergence_plot(df_fro_conv_it, type_x="iterations", mean=meanplot, quantile=quantile)
# ----------------------- Plot --------------------------- #
pxfig2 = line(
            data_frame=df_fro_conv_median_time, #line_group="groups", 
            x="timings", 
            y= "errors", 
            color='algorithm',
            #line_dash='algorithm',
            facet_col="rank",
            facet_col_wrap=2,
            facet_col_spacing=0.07,
            facet_row_spacing=0.17,
            log_y=True,
            log_x=True,
            error_y_mode="band",
            error_y="q_errors_p", 
            error_y_minus="q_errors_m",
            category_orders={
                "algorithm": algs,
                "rank": sorted(df_fro_conv_median_time["rank"].unique())
                }
            )

pxfig2it = line(
            data_frame=df_fro_conv_median_it, 
            x="it", 
            y="errors", 
            color='algorithm',
            #line_dash='algorithm',
            log_y=True,
            #log_x=True,
            #line_group="groups",
            facet_col="rank",
            facet_col_wrap=2,
            facet_col_spacing=0.07,
            facet_row_spacing=0.17,
            error_y_mode="band",
            error_y="q_errors_p", 
            error_y_minus="q_errors_m", 
)

# Final touch
pxfig2.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfig2.update_layout(
    title_text = "NMF",
    font_size = 8*scale_text,
    width=450*scale, # in px
    height=350*scale,
    #xaxis=dict(range=np.log10([2, 100]), title_text="Time (s)"),
    #yaxis=dict(range=np.log10([450, 550]), title_text="Loss"),
)
rename_axis(pxfig2, scale=scale_text, xtext="Time (s)", ytext="n. Loss")
pxfig2.layout.title.text = "Audio Frobenius NMF, convergence plots"
pxfig2.layout.title.font.size = 10*scale_text
pxfig2.update_layout(margin_t=100)

pxfig2.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfig2.update_yaxes(
    matches=None,
    showticklabels=True
)

pxfig2it.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfig2it.update_layout(
    title_text = "NMF",
    font_size = 8*scale_text,
    #width=450*1.62/2, # in px
    #height=450,
    #xaxis=dict(range=[0,4],title_text="Time (s)"),
    #yaxis=dict(title_text="Fit")
)

pxfig2it.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfig2it.update_yaxes(
    matches=None,
    showticklabels=True
)

pxfig2.write_image("Results/"+name+".pdf")
pxfig2it.write_image("Results/"+name+"_it.pdf")
fig.write_image("Results/"+name+"_performance.pdf")
pxfig2.show()
#pxfig2it.show()
# Convergence plots with all runs
#pxfig = px.line(df_l2_conv_median_time, #line_group="groups",
                #x="timings", y= "errors", color='algorithm', 
            #line_dash='algorithm',
            #log_y=True)

#pxfigit = px.line(df_l2_conv_median_it, 
            #x="it", 
            #y= "errors", 
            #color='algorithm',
            #line_dash='algorithm',
            #log_y=True,
            ##error_y="q_errors_p", 
            ##error_y_minus="q_errors_m", 
#)
## Final touch
#pxfig.update_traces(
    #selector=dict(),
    #line_width=2.5,
    ##error_y_thickness = 0.3,
#)

#pxfig.update_layout(
    #title_text = "NMF",
    #font_size = 12,
    #width=450*1.62/2, # in px
    #height=450,
    ##xaxis=dict(range=[0,10], title_text="Time (s)"),
    ##yaxis=dict(range=np.log10([5e-11,1e-7]), title_text="Fit")
#)

#pxfig.update_xaxes(
    #matches = None,
    #showticklabels = True
#)
#pxfig.update_yaxes(
    #matches=None,
    #showticklabels=True
#)
#pxfigit.update_traces(
    #selector=dict(),
    #line_width=2.5,
    ##error_y_thickness = 0.3,
#)

#pxfigit.update_layout(
    #title_text = "NMF",
    #font_size = 12,
    #width=450*1.62/2, # in px
    #height=450,
    ##xaxis=dict(range=[0,0.5], title_text="Time (s)"),
    ##yaxis=dict(range=np.log10([2e-7,7e-7]), title_text="Fit")
#)

#pxfigit.update_xaxes(
    #matches = None,
    #showticklabels = True
#)
#pxfigit.update_yaxes(
    #matches=None,
    #showticklabels=True
#)




#pxfig.write_image("Results/"+name+"_fro.pdf")
#pxfig.write_image("Results/"+name+"_fro.pdf")
#pxfigit.write_image("Results/"+name+"_fro_it.pdf")
#pxfig.show()
#pxfigit.show()

