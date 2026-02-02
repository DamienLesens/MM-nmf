from os import system
import sys

if len(sys.argv)<2:
    nbseeds = 0
else:
    nbseeds=sys.argv[1]

# note: 100 runs for synthetic and NLS real, 10 for realistic NMF
names_100 = [
    "synthetic_comparisons_Frobenius_nls.py",
    "synthetic_comparisons_Frobenius_nmf.py",
    "synthetic_comparisons_Frobenius_nmf_sparse.py",
    "synthetic_comparisons_KL_nls_sparse.py",
    "synthetic_comparisons_KL_nmf_sparse.py",
    "synthetic_comparisons_KL_nmf_sparse_unbalanced.py",
    "test_audio_nls_KL.py",
    "test_audio_nls_fro.py",
    "test_hyperspectral_nls_KL.py",
    "test_hyperspectral_nls_fro.py"
]
names_10 = [
    "test_audio_nmf_KL.py",
    "test_audio_nmf_fro.py",
    "test_hyperspectral_nmf_fro.py",
    "test_hyperspectral_nmf_KL.py",  
]
# run audio nmf by hand
for name in names_100:
    print(name+" running\n")
    #system("python "+name+" "+str(nbseeds))
    #system("python "+name+" "+str(100))
    system("python "+name+" "+str(0))
    
for name in names_10:
    print(name+" running\n")
    #system("python "+name+" "+str(nbseeds))
    #system("python "+name+" "+str(10))
    system("python "+name+" "+str(0))
