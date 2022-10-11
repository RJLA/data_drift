from evidently.analyzers.stattests.jensenshannon import jensenshannon_stat_test
from evidently.analyzers.stattests.kl_div import kl_div_stat_test
from evidently.analyzers.stattests.ks_stattest import ks_stat_test
from evidently.analyzers.stattests.psi import psi_stat_test
from evidently.analyzers.stattests.wasserstein_distance_norm import wasserstein_stat_test
import pandas as pd

def check_numerical_dd(reference, received, threshold_nks = 0.1, threshold_ks = 0.05):

    reference = pd.read_csv(reference)
    received = pd.read_csv(received)

    ddt = pd.DataFrame(index = [
        "Numerical_Features_Drift_Table:Drift_detected=True",
        "JensenShannon", 
        "KullbackLeibler", 
        "PopulationStabilityIndex", 
        "WassersteinDistance", 
        "KolmogorovSmirnov"
        ],
        columns = reference.select_dtypes(include = 'number').columns.tolist())

    for cols in ddt.columns:
        js_res = jensenshannon_stat_test(reference[cols], 
                                        received[cols],
                                        "num",
                                        threshold = threshold_nks).drifted
        
        kl_res = kl_div_stat_test(reference[cols], 
                                        received[cols],
                                        "num",
                                        threshold = threshold_nks).drifted

        psi_res = psi_stat_test(reference[cols], 
                                        received[cols],
                                        "num",
                                        threshold = threshold_nks).drifted     

        emd_res = wasserstein_stat_test(reference[cols], 
                                        received[cols],
                                        "num",
                                        threshold = threshold_nks).drifted

        ks_res = ks_stat_test(reference[cols], 
                                        received[cols],
                                        "num",
                                        threshold = threshold_ks).drifted
        ddt.loc["JensenShannon", cols] = js_res
        ddt.loc["KullbackLeibler", cols] = kl_res
        ddt.loc["PopulationStabilityIndex", cols] = psi_res
        ddt.loc["WassersteinDistance", cols] = emd_res
        ddt.loc["KolmogorovSmirnov", cols] = ks_res
        ddt.loc["Numerical_Features_Drift_Table:Drift_detected=True", cols] = "-"
    
    # print("Numerical Features Drift Table: Drift detected = True")
    return ddt