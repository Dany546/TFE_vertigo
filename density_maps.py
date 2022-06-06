from dipy.io.gradients import read_bvals_bvecs
from dipy.io.streamline import load_tck, save_trk, save_tck, load_trk
from dipy.io.image import load_nifti_data, load_nifti, save_nifti
from AFQ.segmentation import Segmentation
from AFQ.segmentation import clean_bundle as seg_clean_bundle
from AFQ.api.bundle_dict import BundleDict, RECO_BUNDLES_80, BUNDLES, CALLOSUM_BUNDLES
from AFQ.definitions import scalar  
from joblib import Parallel, delayed
# from cython.view import memoryview, array
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import VotingClassifier
from dipy.tracking.streamline import length
import AFQ.data as afd  
import sys 
import seaborn as sns
import gc
import warnings
import time
import joblib
import itertools
import scipy.optimize
import scipy.stats as sct
import bebi103
# from tracto import dipy_tracto
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.decomposition import PCA
from dipy.denoise.enhancement_kernel import EnhancementKernel
from dipy.tracking.fbcmeasures import FBCMeasures
from AFQ.utils.streamlines import bundles_to_tgram
# from AFQ.viz.fury_backend import visualize_bundles 
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import dipy.tracking.utils as dtu
from dipy.stats.analysis import afq_profile, gaussian_weights
from multiprocessing import Process, Queue
from dipy.tracking.streamline import set_number_of_points, values_from_volume
import nibabel as nib
import dipy.tracking.streamlinespeed as dps
import dipy.tracking.streamline as dts
import AFQ.registration as reg
from AFQ.tasks.utils import get_default_args
# from AFQ.viz.plotly_backend import single_bundle_viz 
import dipy.core.gradients as dpg
from cmtk_stolen import *
from dipy.align.streamlinear import whole_brain_slr
import copy
import os
import json
import pickle, json
import bz2 
import _pickle as cPickle
import os.path as op
import pandas as pd
import numpy as np
from dipy.align.bundlemin import distance_matrix_mdf
from dipy.io.image import load_nifti_data, load_nifti, save_nifti
from dipy.io.streamline import load_tck, load_trk, save_trk, save_tck 
from dipy.direction import peaks 
import matplotlib.pyplot as plt
import dipy.data as dpd
import copy 
from wm_query import query
import multiprocessing 
from joblib import Parallel, delayed
import dipy.tracking.streamline as dts
import dipy.tracking.streamlinespeed as dps
from dipy.viz import window, actor
from scipy.ndimage import binary_dilation, binary_erosion
from dipy.align.imaffine import (transform_centers_of_mass, AffineMap, MutualInformationMetric, AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D) 
from dipy.data import get_sphere 
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel, auto_response_ssst, response_from_mask_ssst, mask_for_response_ssst, recursive_response)
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion, CmcStoppingCriterion
import dipy.direction.peaks as dp 
from dipy.reconst import sfm 
import AFQ.registration as dipy_syn_reg
import dipy.core.gradients as dpg
from dipy.tracking.streamline import set_number_of_points
from dipy.direction import ProbabilisticDirectionGetter, BootDirectionGetter
from dipy.data import default_sphere
from dipy.core.geometry import cart2sphere
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs 
from dipy.reconst import shm
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking, ParticleFilteringTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines  
from cmtk_stolen import *
from pathlib import Path
import pandas as pd
import threading
import numpy as np
import datetime
import subprocess
import copy as cp
import pickle, json
import bz2 
import _pickle as cPickle
import sys
import os 
from dipy.tracking.utils import density_map
from dipy.stats.analysis import orient_by_streamline, values_from_volume 

sys.setrecursionlimit(20971052)     
threading.stack_size(134217728)  
sys.settrace 
my_f_path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/"

my_f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset_new/PROJECT/" 
my_f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/elikopy_subset_new2/PROJECT/"  
# f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset/PROJECT/"  

def _resample_tg(tg, n_points):
    # reformat for dipy's set_number_of_points
    if isinstance(tg, np.ndarray):
        if len(tg.shape) > 2:
            streamlines = tg.tolist()
            streamlines = [np.asarray(item) for item in streamlines]
    elif isinstance(tg, list):
        print(tg)
        if not isinstance(tg[0][0],list):
            tg = [tg]
        streamlines = [np.asarray(item) for item in tg]
        print(streamlines)
    else:
        streamlines = [np.asarray(item) for item in list(tg.streamlines)] 

    return dps.set_number_of_points(streamlines, n_points)
    
def reorient(streamlines, nifti, f_path=my_f_path, trk_type="prob"):
 
    ext = "" if trk_type=="prob" else "_det"
    bundle = pd.read_csv("/auto/home/users/d/r/drimez/orders%s/labels.txt"%ext, sep=" ") 
     
    bundle = pd.DataFrame([ [lign__[0], float(lign__[1].replace(",","").replace("[","")), 
                             float(lign__[2].replace("]",""))] 
                            for lign__ in bundle.values], columns=bundle.columns)
     
    streamlines.to_vox()
    streamlines = streamlines.streamlines 
    start = np.nanmean([streamline[0] for streamline in streamlines], axis=0)
    end = np.nanmean([streamline[-1] for streamline in streamlines], axis=0)
     
    return_none = False
    if len(streamlines[0])>3:
        streamlines = [sl for sl in streamlines if len(sl)>2]
    elif isinstance(list,streamlines[0][0]):
        streamlines = [sl for sl in streamlines if len(sl)>2]
    elif len(streamlines)>2:
        streamlines = streamlines 
    else:
        return_none = True
     
    if not (np.linalg.norm(start-bundle.values[0,1])+np.linalg.norm(end-bundle.values[0,2])<np.linalg.norm(start-bundle.values[0,2])+np.linalg.norm(end-bundle.values[0,1])): 
        if len(streamlines[0])>3:
            streamlines = [sl[::-1] for sl in streamlines if len(sl)>2]
        elif isinstance(list,streamlines[0][0]):
            streamlines = [sl[::-1] for sl in streamlines if len(sl)>2]
        elif len(streamlines)>2:
            streamlines = streamlines[::-1]  
        else:
            return_none = True
    
    if return_none: 
        return None, None
    else:
        trk = StatefulTractogram( streamlines,  
                                  nifti, 
                                  Space.VOX   )
        trk.to_rasmm()
             
        return trk, trk.streamlines

def aggregate(list_of_dicts):
 
    to_return = {str(key___).replace("'",""):[val___] for key___, val___ in list_of_dicts[0].items()}
    keys_list = [*to_return.keys()]
    for nd, dict_ in enumerate(list_of_dicts[1:]):
        dict_ = {str(key___).replace("'",""):val___ for key___, val___ in dict_.items()}
        this_keys_list = np.array([*dict_.keys()])
        for one_key in keys_list: 
            if one_key in this_keys_list:
                to_return[one_key].append(dict_[one_key])
            else:
                to_return[one_key].append(np.nan)
        if len([_ for _ in this_keys_list if not (_ in keys_list)])>0:
            for new_key in [_ for _ in this_keys_list if not (_ in keys_list)]:
                to_return[new_key] = [np.nan for uuu in range(nd+1)]
                to_return[new_key].append(dict_[new_key])
            keys_list = [*to_return.keys()]
               
    return to_return

def plot_performances(path, data_type, target):
 
    perfs = [] 
    if os.path.isdir(path) and len(path.split(data_type))==2 and len(path.split(target))==2:
        list_dir = []
        with os.scandir(path) as _:
            for __ in _:
                list_dir.append(__.path)
        reliable_perfs = None ; params = None ; metrics = None
        for file_path in list_dir:
            if len(file_path.split("reliable"))==2 and (not len(file_path.split("unreliable"))==2): 
                reliable_perfs = pd.read_csv( file_path, sep="},", skiprows=0,
                                              names=["fitted params","perfs"]).drop_duplicates()  
                 
                perfs = [] 
                if params is None:
                    params = [key for key in [_.split(": ")[0].replace("'","") for _ in (reliable_perfs.values[1,0]+"}").replace("{","").replace("}","").split(", ")]]
                    metrics = [key for key in [_.split(": ")[0].replace("'","") for _ in (reliable_perfs.values[1,1]+"}").replace("{","").replace("}","").split(", ")]]
                for par, lign in zip(reliable_perfs.values[1:,0], reliable_perfs.values[1:,1]): 
                    perf_params = {key:val for key, val in zip([_.split(": ")[0].replace("'","") for _ in (par+"}").replace("{","").replace("}","").split(", ")],
                                                              [_.split(": ")[1].replace("'","") for _ in (par+"}").replace("{","").replace("}","").split(", ")]) }
                    perf_metrics = {key:float(val) for key, val in zip([_.split(": ")[0].replace("'","") for _ in lign.replace("{","").replace("}","").split(", ")],
                                                          [_.split(": ")[1].replace("'","") for _ in lign.replace("{","").replace("}","").split(", ")]) }
                    perf_dict = dict( **perf_params,
                                      **perf_metrics )
                    perfs.append(perf_dict) 
                    
                    for new_key in [_ for _ in [*perf_params.keys()] if not (_ in params)]:
                        params.append(new_key)
                    for new_key in [_ for _ in [*perf_metrics.keys()] if not (_ in metrics)]:
                        metrics.append(new_key)
                     
    perfs = aggregate(perfs)
    perfs_df = pd.DataFrame(perfs,columns=[*perfs.keys()]) 
    for x_axis in params:
        if not x_axis=="scaler":
            for y_axis in metrics:
                print(x_axis,y_axis)
                try:
                    fig, ax = plt.subplots(1,1)
                    sns.boxplot(x=x_axis, y=y_axis, hue="scaler", ax=ax)
                    if not os.path.exists("/auto/home/users/d/r/drimez/Classify_results/plots/%s/"%data_type):
                        os.makedirs("/auto/home/users/d/r/drimez/Classify_results/plots/%s/"%data_type)
                    plt.savefig("/auto/home/users/d/r/drimez/Classify_results/plots/%s/%s_vs_%s.png"%(data_type,x_axis, y_axis))
                except Exception as err:
                    print(err) 
                          
def compare_models(root_path, target, age=True, threshold=0.05, type__="prob"):
 
    perfs = [] ; params = None ; metrics = None ; models = [] ; figname = target
    for age in [False,True]: #  
        for data_type in ["scale_pca","select"]: 
            with os.scandir(root_path) as my_iterator:
                for my_entry in my_iterator:
                    path = my_entry.path  
                    with_or_without_age = "_without_age" if age else "_"
                    this_data_type = my_entry.name.replace("_elast","").replace("_svm","").replace("_sgl","").replace("_"+target,"")
                    this_data_type = this_data_type.replace(with_or_without_age,"") if age else this_data_type
                    if os.path.isdir(path) and len(path.split(target))==2:
                        if len(this_data_type.split(data_type))==2:
                            split_list = my_entry.name.replace("_"+target+"_","").split("_")
                            model = (split_list[-1] + "_" + data_type + with_or_without_age).replace("scale_","")
                            print(my_entry.name)
                            models = np.unique(np.append(models,model))
                            list_dir = []
                            with os.scandir(path) as _:
                                for __ in _:
                                    list_dir.append(__.path)
                            reliable_perfs = None 
                            for file_path in list_dir: 
                                if len(file_path.split("reliable"))==2 and (not len(file_path.split("unreliable"))==2) \
                                   and np.any([True if len(_file_.split("results"))==2 else False for _file_ in list_dir]): 
                                   
                                    print("one pass:",file_path)
                                    reliable_perfs = pd.read_csv( file_path, sep="},", skiprows=0,
                                                                  names=["fitted params","perfs"]).drop_duplicates()  
                                    
                                    this_params = []
                                    for this_par in reliable_perfs.values[1:,0]:
                                        params_string = ""
                                        for par_name, params_val in zip([_.split(": ")[0] for _ in (this_par+"}").replace("{","").replace("}","").split(", ")],
                                                                        [_.split(": ")[1] for _ in (this_par+"}").replace("{","").replace("}","").split(", ")]):
                                            params_string += str(par_name) + "=" + str(params_val) + "_"
                                        this_params.append(params_string)
                                     
                                    results_folder = np.array([True if len(_file_.split("results"))==2 else False for _file_ in list_dir]) 
                                    success_folder = np.array(list_dir)[results_folder]
                                    if isinstance(success_folder,list) or isinstance(success_folder,np.ndarray):
                                        success_folder = success_folder[0]
                                        
                                    success_file = os.listdir(success_folder) ; pvalues = []
                                    for __pval in success_file: 
                                        for a_param in this_params:  
                                            if str(__pval).replace(".txt","").replace("b'","").replace("'","")==a_param.replace("'",""): 
                                                pvalues.append(pd.read_csv(success_folder+"/"+str(__pval).replace("b'","").replace("'","")).values[2,-1])
                                    print(pvalues)
                                    if params is None:
                                        params = [key for key in [_.split(": ")[0].replace("'","").replace(" ","") for _ in (reliable_perfs.values[1,0]+"}").replace("{","").replace("}","").split(", ")]] 
                                        metrics = [key for key in [_.split(": ")[0].replace("'","").replace(" ","") for _ in (reliable_perfs.values[1,1]+"}").replace("{","").replace("}","").split(", ")]] 
                                    for par, lign, pvalue in zip(reliable_perfs.values[1:,0], reliable_perfs.values[1:,1],pvalues): 
                                        if pvalue<=threshold:
                                            perf_params = {key:val for key, val in zip([_.split(": ")[0].replace("'","").replace(" ","") for _ in (par+"}").replace("{","").replace("}","").split(", ")],
                                                                                      [_.split(": ")[1].replace("'","").replace(" ","") for _ in (par+"}").replace("{","").replace("}","").split(", ")]) }
                                            perf_metrics = {key:float(val) for key, val in zip([_.split(": ")[0].replace("'","").replace(" ","") for _ in lign.replace("{","").replace("}","").split(", ")],
                                                                                  [_.split(": ")[1].replace("'","").replace(" ","") for _ in lign.replace("{","").replace("}","").split(", ")]) }
                                            perf_dict = dict( **{"model":model},
                                                              **perf_params,
                                                              **perf_metrics )
                                            perfs.append(perf_dict) 
                                            
                                            for new_key in [_ for _ in [*perf_params.keys()] if not (_ in params)]:
                                                params.append(new_key.replace("'","").replace(" ","")) 
                                            for new_key in [_ for _ in [*perf_metrics.keys()] if not (_ in metrics)]:
                                                metrics.append(new_key.replace("'","").replace(" ","")) 
                            
    if threshold==0.05:                                 
        if len(perfs)>0:
            metrics_test  = [_ for _ in metrics if len(_.split("test"))>1]
            metrics_train = [_.replace("test","train") for _ in metrics_test]
            metrics  = [_.replace("test_","").replace("mean_","") for _ in metrics_test]
            perfs = aggregate(perfs)
            perfs_df = pd.DataFrame.from_dict(perfs)    
            if not os.path.exists("/auto/home/users/d/r/drimez/Classify_results/comp_reliable/%s/%s/"%(type__,figname)):
                os.makedirs("/auto/home/users/d/r/drimez/Classify_results/comp_reliable/%s/%s/"%(type__,figname))
            for y_axis_train, y_axis_test, y_name in zip(metrics_train,metrics_test,metrics): 
                try:
                    fig, ax = plt.subplots(1,2,figsize=(15,4))
                    ax = ax.flatten()
                    temp_df = pd.DataFrame(np.array([perfs_df[y_axis_train].values.flatten(),
                                                     perfs_df["model"].values.flatten()]).T,columns=[y_axis_train.replace("mean_",""),"model"]).dropna() 
                    print(temp_df)
                    ax[0] = sns.boxplot(x="model", y=y_axis_train.replace("mean_",""), data=temp_df, ax=ax[0]) 
                    temp_df = pd.DataFrame(np.array([perfs_df[y_axis_test].values.flatten(),
                                                     perfs_df["model"].values.flatten()]).T,columns=[y_axis_test.replace("mean_",""),"model"]).dropna() 
                    print(temp_df)
                    ax[1] = sns.boxplot(x="model", y=y_axis_test.replace("mean_",""), data=temp_df, ax=ax[1])  
                    try:
                        ax[0].legend([],[], frameon=False)
                        sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
                    except Exception:
                        pass
                    plt.savefig("/auto/home/users/d/r/drimez/Classify_results/comp_reliable/%s/%s/%s_vs_%s.png"%(type__,figname,"model", y_name))
                except Exception as err:
                    print(err) 
            for x_axis in params:  
                for y_axis_train, y_axis_test, y_name in zip(metrics_train,metrics_test,metrics): 
                    try:
                        fig, ax = plt.subplots(1,2,figsize=(15,4))
                        ax = ax.flatten()
                        temp_df = pd.DataFrame(np.array([perfs_df[x_axis].values.flatten(),
                                                         perfs_df[y_axis_train].values.flatten(),
                                                         perfs_df["model"].values.flatten()]).T,columns=[x_axis,y_axis_train.replace("mean_",""),"model"]).dropna() 
                        print(temp_df)
                        ax[0] = sns.boxplot(x=x_axis, y=y_axis_train.replace("mean_",""), hue="model", data=temp_df, ax=ax[0]) 
                        ax[0].legend([],[], frameon=False)
                        temp_df = pd.DataFrame(np.array([perfs_df[x_axis].values.flatten(),
                                                         perfs_df[y_axis_test].values.flatten(),
                                                         perfs_df["model"].values.flatten()]).T,columns=[x_axis,y_axis_test.replace("mean_",""),"model"]).dropna() 
                        print(temp_df)
                        ax[1] = sns.boxplot(x=x_axis, y=y_axis_test.replace("mean_",""), hue="model", data=temp_df, ax=ax[1]) 
                        try:
                            ax[0].legend([],[], frameon=False)
                            sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
                        except Exception:
                            pass
                        plt.legend(labels = perfs_df["model"].values.flatten(),loc = 2, bbox_to_anchor = (1,1))
                        plt.savefig("/auto/home/users/d/r/drimez/Classify_results/comp_reliable/%s/%s/%s_vs_%s.png"%(type__,figname,x_axis, y_name))
                    except Exception as err:
                        print(err)  
        else:
            print("no reliable models")
        compare_models(root_path, data_type, target, threshold=1)
    else:                               
        if len(perfs)>0:
            metrics_test  = [_ for _ in metrics if len(_.split("test"))>1]
            metrics_train = [_.replace("test","train") for _ in metrics_test]
            metrics  = [_.replace("test_","").replace("mean_","") for _ in metrics_test]
            perfs = aggregate(perfs)
            perfs_df = pd.DataFrame.from_dict(perfs)   
            if not os.path.exists("/auto/home/users/d/r/drimez/Classify_results/comp_unreliable/%s/%s/"%(type__,figname)):
                os.makedirs("/auto/home/users/d/r/drimez/Classify_results/comp_unreliable/%s/%s/"%(type__,figname))
            for y_axis_train, y_axis_test, y_name in zip(metrics_train,metrics_test,metrics): 
                try:
                    fig, ax = plt.subplots(1,2,figsize=(15,4))
                    ax = ax.flatten()
                    temp_df = pd.DataFrame(np.array([perfs_df[y_axis_train].values.flatten(),
                                                     perfs_df["model"].values.flatten()]).T,columns=[y_axis_train.replace("mean_",""),"model"]).dropna() 
                    print(temp_df)
                    ax[0] = sns.boxplot(x="model", y=y_axis_train.replace("mean_",""), data=temp_df, ax=ax[0]) 
                    temp_df = pd.DataFrame(np.array([perfs_df[y_axis_test].values.flatten(),
                                                     perfs_df["model"].values.flatten()]).T,columns=[y_axis_test.replace("mean_",""),"model"]).dropna() 
                    print(temp_df)
                    ax[1] = sns.boxplot(x="model", y=y_axis_test.replace("mean_",""), data=temp_df, ax=ax[1])  
                    try:
                        ax[0].legend([],[], frameon=False)
                        sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
                    except Exception:
                        pass
                    plt.savefig("/auto/home/users/d/r/drimez/Classify_results/comp_unreliable/%s/%s/%s_vs_%s.png"%(type__,figname,"model", y_name))
                except Exception as err:
                    print(err) 
            for x_axis in params:  
                for y_axis_train, y_axis_test, y_name in zip(metrics_train,metrics_test,metrics): 
                    try:
                        fig, ax = plt.subplots(1,2,figsize=(15,4))
                        ax = ax.flatten()
                        temp_df = pd.DataFrame(np.array([perfs_df[x_axis].values.flatten(),
                                                         perfs_df[y_axis_train].values.flatten(),
                                                         perfs_df["model"].values.flatten()]).T,columns=[x_axis,y_axis_train.replace("mean_",""),"model"]).dropna() 
                        print(temp_df)
                        ax[0] = sns.boxplot(x=x_axis, y=y_axis_train.replace("mean_",""), hue="model", data=temp_df, ax=ax[0]) 
                        ax[0].legend([],[], frameon=False)
                        temp_df = pd.DataFrame(np.array([perfs_df[x_axis].values.flatten(),
                                                         perfs_df[y_axis_test].values.flatten(),
                                                         perfs_df["model"].values.flatten()]).T,columns=[x_axis,y_axis_test.replace("mean_",""),"model"]).dropna() 
                        print(temp_df)
                        ax[1] = sns.boxplot(x=x_axis, y=y_axis_test.replace("mean_",""), hue="model", data=temp_df, ax=ax[1]) 
                        try:
                            ax[0].legend([],[], frameon=False)
                            sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
                        except Exception:
                            pass
                        plt.legend(labels = perfs_df["model"].values.flatten(),loc = 2, bbox_to_anchor = (1,1))
                        plt.savefig("/auto/home/users/d/r/drimez/Classify_results/comp_unreliable/%s/%s/%s_vs_%s.png"%(type__,figname,x_axis, y_name))
                    except Exception as err:
                        print(err) 
        else:
            print("no models")
                                  

def parc_rois(wmparc_path, selected_list, f_path=my_f_path, dens=None, path_prefix=None, folder__=None):
  
    wmparc, affine = load_nifti(wmparc_path)
    LUT = pd.read_csv("/auto/home/users/d/r/drimez/LUT.txt",sep="  ",header=None,index_col=False,names=["id","name","r","g","b","a"])  
    names = np.array([_.split('-')[-1] for _ in LUT["name"].values])
    lut_index = np.arange(len(LUT["id"].values))
    metrics = ["pc0","pc1"] if len(selected_list)==2 else ["FA","MD","AD","RD"]
    folder = path_prefix + folder__
    for ise, selected_rois in zip(metrics,selected_list):
        new_wmparc = np.zeros_like(wmparc).astype(float)
        if selected_rois=={}:
            return None
        for roi, value in selected_rois.items():  
            if not "_".join(roi.split("_")[:-2]) in ("UnsegmentedWhiteMatter","unknown"):
                corresponding = lut_index[np.array(names=="_".join(roi.split("_")[:-2]))] 
                corresponding_indx = int(roi.split("_")[-2])
                corresponding = corresponding[corresponding_indx]  
                label = LUT["id"].values[corresponding] 
                new_wmparc[wmparc==label] = value 
        
        if not os.path.isdir("/auto/home/users/d/r/drimez/Classify_results/wmparc/"):
            os.makedirs("/auto/home/users/d/r/drimez/Classify_results/wmparc/")
        
        new_wmparc_1 = copy.copy(new_wmparc)
        new_wmparc_1[new_wmparc<0] = 0
        min_1, max_1 = np.abs([new_wmparc_1.min(),new_wmparc_1.max()])
        
        new_wmparc_2 = copy.copy(new_wmparc)
        new_wmparc_2[new_wmparc>0] = 0
        new_wmparc_2 = -new_wmparc_2 
        new_wmparc_2 += new_wmparc_1
        min_2, max_2 = np.abs([new_wmparc_2.min(),new_wmparc_2.max()])
        max_ = np.max([max_1,max_2])
         
        path_prefix_ = "" if path_prefix is None else path_prefix + "_"
        path_1 = "/auto/home/users/d/r/drimez/Classify_results/" + path_prefix_ + "wmparc_significant"
        path_2 = "/auto/home/users/d/r/drimez/Classify_results/" + path_prefix_ + "wmparc_unsignificant"
        save_nifti(path_1+".nii.gz",new_wmparc_1,affine)
        save_nifti(path_2+".nii.gz",new_wmparc_2,affine)
        
        if not os.path.isdir("/auto/home/users/d/r/drimez/Classify_results/wmparc/"+folder+"/"):
            os.makedirs("/auto/home/users/d/r/drimez/Classify_results/wmparc/"+folder+"/")
        
        t1_path = "/".join(wmparc_path.split("/")[:-3]) + "/%s_T1_corr_projected"%wmparc_path.split("/")[-4]
        t1_path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/elikopy_subset_new2/PROJECT/subjects/C_1/T1/C_1_T1_corr_projected" 
        bshcmd = "cd /auto/home/users/d/r/drimez/Classify_results/wmparc/ ;"
        bshcmd += "fsleyes render --scene ortho --size 2000 2000 --hideCursor --crop 20 -of " + folder + "/results_ortho_%s.png "%ise + t1_path + " -a 50 " + path_1 + " -cm red-yellow -a 20 -dr %s %s "%(0,max_) + path_2 + " -cm blue-lightblue -a 70 -dr %s %s -in linear --logScale ; "%(0,max_)
        bshcmd += "fsleyes render --scene lightbox --size 8000 8000 --hideCursor --crop 20 -zx X -ss 2 -nr 20 -nc 5 -of " + folder + "/side_results_light_%s.png "%ise + t1_path + " -a 50 " + path_1 + " -cm red-yellow -a 20 -dr %s %s "%(0,max_) + path_2 + " -cm blue-lightblue -a 70 -dr %s %s -in linear --logScale ; "%(0,max_)
        bshcmd += "fsleyes render --scene lightbox --size 8000 8000 --hideCursor --crop 20 -zx Y -ss 3 -nr 20 -nc 5 -of " + folder + "/ant_results_light_%s.png "%ise + t1_path + " -a 50 " + path_1 + " -cm red-yellow -a 20 -dr %s %s "%(0,max_) + path_2 + " -cm blue-lightblue -a 70 -dr %s %s -in linear --logScale ; "%(0,max_) 
        
        bshcmd.split()
        process = subprocess.Popen(bshcmd, universal_newlines=True, shell=True,
                                   stdout=sys.stdout, stderr=sys.stdout) 
        outs, errs = process.communicate() 
 
from dipy.align.reslice import reslice

def density_map(streamlines, affine, vol_dims):
    affine = np.array(affine, dtype=float)
    inv_affine = np.linalg.inv(affine)
    lin_T = inv_affine[:3, :3].T.copy()
    offset = inv_affine[:3, 3] + .5 
    counts = np.zeros(vol_dims, 'int')
    i,j,k = 0,0,0
    for sl in streamlines: 
        try:
            inds = np.dot(sl, lin_T) 
            inds += offset
            if inds.min().round(decimals=6) < 0:
                print(inds)
                continue
            else:
                i, j, k = inds.T.astype(int)
                counts[i, j, k] += 1
        except Exception as err: 
            print(i,j,k)
            print(err)
    return counts
 
def split_segments_age(trk_path, nifti, selected_list, f_path=my_f_path, dens=None, type_="prob", path_prefix="", separator="_"):
  
    print("segments")
    selected_seg_list = [{} for tt in selected_list]
    for ise, selected_segments in enumerate(selected_list): 
        new_dens = np.zeros_like(dens).astype(float)
        if selected_segments=={}:
            return None
        for roi, value in selected_segments.items():    
            roi = roi.replace("(","").replace(")","").replace("'","").replace(" ","")
            bdle = separator.join(roi.split(separator)[:-2])    
            seg = roi.split(separator)[-2]    
            if not (bdle in selected_seg_list[ise].keys()):
                selected_seg_list[ise][bdle] = {} 
            selected_seg_list[ise][bdle][seg] = value
    
    print("",file=open("/auto/home/users/d/r/drimez/afq.txt","w"))
    print("Density") 
    dens = [None for _ in selected_seg_list]
    for selected_seg, ise in zip(selected_seg_list,["FA","MD","AD","RD"]):
     
        path_prefix_ = "" if path_prefix is None else path_prefix + "_"
        path_1 = "/auto/home/users/d/r/drimez/Classify_results/%s/"%type_ + path_prefix_ + "significant_" + ise
        path_2 = "/auto/home/users/d/r/drimez/Classify_results/%s/"%type_ + path_prefix_ + "unsignificant_" + ise
        
        if not os.path.exists("/auto/home/users/d/r/drimez/Classify_results/%s/"%type_ + path_prefix + "/ant_results_light_%s.png"%ise):
            if not os.path.exists(path_2): 
                vol_dims = None ; affine = None ; dens = None ; save_affine = None
                with os.scandir(trk_path) as iterator:
                    for entry in iterator:
                        print(entry.name)
                        if "trk" in entry.name.split("."):  
                            new_entry_name = entry.name.replace(".trk","").replace("_","")
                            print(new_entry_name)
                            if new_entry_name in selected_seg.keys():
                                streamlines = load_trk(entry.path,nifti) 
                                trk, streamlines = reorient(streamlines, nifti, trk_type=type_)
                                if trk is None:
                                    print(new_entry_name,"not found")
                                else:
                                    img, affine, vox_sizes = load_nifti(nifti, return_voxsize=True)  
                                    new_vox_sizes = np.array([___/4 for ___ in vox_sizes])
                                    vol_dims = np.array([img.shape[0]*4,img.shape[1]*4,img.shape[2]*4]).flatten()
                                    
                                    _, affine = reslice(img, affine, vox_sizes, new_vox_sizes) 
                                      
                                    # affine[:3,-1] += new_vox_sizes*np.array([_/4 for _ in vol_dims])*affine[:3,-1]/abs(affine[:3,-1])
                                    # print(np.linalg.inv(affine),file=open("/auto/home/users/d/r/drimez/afq.txt","a"))
                                    """
                                    save_affine = copy.copy(affine)
                                    # affine[:3,:3] *= (np.ones((3,3)) - np.eye(3)*(3/4)) 
                                    affine[:3,:3] /= (np.ones((3,3)) + np.eye(3)*3) 
                                    affine[:3,-1] += (np.ones((3,3)) + np.eye(3)*3) 
                                    save_affine[:3,:3] *= (np.ones((3,3)) + np.eye(3)*3) 
                                    print(affine)
                                    """ 
                                    sl_size = len(streamlines) 
                                    this_sl = streamlines
                                     
                                    n_seg = len([0 for _ in pd.read_csv("/auto/home/users/d/r/drimez/data_all_prob.csv").columns.values if _.split(",")[0][2:-1]==new_entry_name and _.split(",")[-1][2:-2]=="FA"]) 
                                    lengths = length(this_sl) ; mean_length = None
                                    if not isinstance(lengths,np.ndarray):
                                        mean_length = lengths
                                        lengths = np.array([lengths])
                                    else: 
                                        standard = this_sl[np.argmin(abs(lengths-np.quantile(lengths,0.625)))] # sets reference streamline to the mid-second-quartile length (long enough to be in the bundle, short enough to be "straight")
                                        this_sl = orient_by_streamline(this_sl, standard, n_points=100)        # reorient streamlines for correct estimation of mean bundle 
                                        mean_length = np.mean(lengths)
                                         
                                    seglen = mean_length/n_seg 
                                    print(sl_size,np.array([len(this_) for this_ in this_sl]).min(),mean_length,n_seg)
                                           
                                    new_sl = None ; centroids = None
                                    if len(this_sl)!=0:
                                        if isinstance(this_sl[0],list) or isinstance(this_sl[0],np.ndarray):
                                            if not (isinstance(this_sl[0][0],list) or isinstance(this_sl[0][0],np.ndarray)): # it is not a list of streamlines but a single streamline
                                                this_sl = [this_sl]
                                       
                                    if len(this_sl)>1: 
                                        new_sl_tg = _resample_tg(trk,1 + 2*n_seg)  
                                        new_sl = np.array(new_sl_tg)  
                                        centroids = np.mean(new_sl[:,1::2],0)  # centrers of segments
                                    elif len(this_sl)>50:                      
                                        temp_tg = StatefulTractogram.from_sft( this_sl, trk)  
                                        new_sl_tg = np.array(_resample_tg(temp_tg,1 + 2*n_seg))
                                        this_prof_weights = gaussian_weights(new_sl_tg, n_points=n_seg) 
                                        new_sl = np.array(new_sl_tg) 
                                        centroids = np.mean((new_sl*this_prof_weights)[:,1::2],0)  # centrers of segments
                                    else: 
                                        this_sl = [this_sl]
                                        temp_tg = StatefulTractogram.from_sft( this_sl, trk) 
                                        new_sl_tg = _resample_tg(temp_tg,1 + 2*n_seg)  
                                        new_sl = np.array([new_sl_tg])
                                        centroids = new_sl[0,1::2]    # centrers of segments
                                        
                                    lengths_ = [np.linalg.norm(end__-start__) for end__, start__ in zip(np.mean(new_sl[:,::2],0)[:-1],np.mean(new_sl[:,::2],0)[1:])] # start and ends of segments
                                    assignments_ = []
                                    for s in this_sl: 
                                        assignments_.append([np.argmin([np.linalg.norm(si-centroid) for centroid in centroids]) for si in s])   
                                    new_sl = None 
                                    
                                    seg_sl_assignments = assignments_
                                    segments = [[] for nnn_seg in range(n_seg)] 
                                    for assigned_sl in range(len(streamlines)):
                                        split_sl = this_sl[assigned_sl]
                                        for nnn_seg in range(n_seg): 
                                            to_append = split_sl[np.array(seg_sl_assignments[assigned_sl])==nnn_seg]
                                            if len(to_append)>0:
                                                if not (isinstance(to_append[0],list) or isinstance(to_append[0],np.ndarray)):
                                                    to_append = [to_append]  
                                            to_append = [_ for _ in to_append if len(_)>0]
                                            if len(to_append)>0:
                                                segments[nnn_seg].append(to_append)   
                                       
                                    for iseg, segment in enumerate(segments): 
                                        if str(iseg) in [*selected_seg[new_entry_name].keys()]: 
                                            tdm = density_map(segment, affine, vol_dims) 
                                            if dens is None: 
                                                dens = tdm*selected_seg[new_entry_name][str(iseg)]/sl_size
                                            else:
                                                dens += tdm*selected_seg[new_entry_name][str(iseg)]/sl_size
                print("fsleyes")  
                
                new_1 = copy.copy(dens)
                new_1[dens<0] = 0
                min_1, max_1 = np.abs([new_1.min(),new_1.max()])
                
                new_2 = copy.copy(dens)
                new_2[dens>0] = 0
                new_2 = -new_2 
                min_2, max_2 = np.abs([new_2.min(),new_2.max()])
                max_ = np.max([max_1,max_2])
                print(max_,file=open("/auto/home/users/d/r/drimez/afq.txt","a"))
                
                if not os.path.isdir("/auto/home/users/d/r/drimez/Classify_results/%s/"%type_):
                    os.makedirs("/auto/home/users/d/r/drimez/Classify_results/%s/"%type_)
                    
                save_nifti(path_1+".nii.gz",new_1, affine) 
                save_nifti(path_2+".nii.gz",new_2, affine) 
         
            t1_path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/elikopy_subset_new2/PROJECT/subjects/C_1/T1/C_1_T1_corr_projected" 
            # bshcmd = "cp " + t1_path + ".nii.gz /auto/home/users/d/r/drimez/Classify_results/%s/ ; "%type_
             
            t1_path = "/auto/home/users/d/r/drimez/Classify_results/%s_old/C_1_T1_corr_projected"%type_
            bshcmd = "cd /auto/home/users/d/r/drimez/Classify_results/%s/ ; "%type_ 
            bshcmd += "fsleyes render --scene ortho --size 2000 2000 --crop 20 -of " + path_prefix + "/results_ortho_%s.png --hideCursor "%ise + t1_path + " -a 40 "
            
            bsh = copy.copy(bshcmd)
            bsh.split()
            process = subprocess.Popen(bsh, universal_newlines=True, shell=True,
                                       stdout=sys.stdout, stderr=sys.stdout) 
            outs, errs = process.communicate()  
            
            if max_1>0:
                bshcmd += path_1 + " -cm red-yellow -a 20 -dr %s %s -in linear "%(0,max_)  
            if max_2>0:
                bshcmd += path_2 + " -cm blue-lightblue -a 50 -dr %s %s -in linear ; "%(0,max_) 
            
            bshcmd.split()
            process = subprocess.Popen(bshcmd, universal_newlines=True, shell=True,
                                       stdout=sys.stdout, stderr=sys.stdout) 
            outs, errs = process.communicate()  
            
            bshcmd = "cd /auto/home/users/d/r/drimez/Classify_results/%s/ ; "%type_ 
            bshcmd += "fsleyes render --scene lightbox --size 8000 8000 --hideCursor --crop 20 -zx X -ss 2 -nr 20 -nc 5 -of " + path_prefix + "/side_results_light_%s.png "%ise + t1_path + " -a 40 " + path_1 + " -cm red-yellow -a 20 -dr %s %s -in linear "%(0,max_)  + path_2 + " -cm blue-lightblue -a 50 -dr %s %s -in linear ; "%(0,max_) 
            
            bshcmd.split()
            process = subprocess.Popen(bshcmd, universal_newlines=True, shell=True,
                                       stdout=sys.stdout, stderr=sys.stdout) 
            outs, errs = process.communicate()  
            
            bshcmd = "cd /auto/home/users/d/r/drimez/Classify_results/%s/ ; "%type_ 
            bshcmd += "fsleyes render --scene lightbox --size 8000 8000 --hideCursor --crop 20 -zx Y -ss 3 -nr 20 -nc 5 -of " + path_prefix + "/ant_results_light_%s.png "%ise + t1_path + " -a 40 " + path_1 + " -cm red-yellow -a 20 -dr %s %s -in linear "%(0,max_)  + path_2 + " -cm blue-lightblue -a 50 -dr %s %s -in linear ; "%(0,max_)   
            # bshcmd += "rm /auto/home/users/d/r/drimez/Classify_results/%s/%s.nii.gz ; "%(type_,t1_path)
            bshcmd += "rm %s.nii.gz %s.nii.gz ; "%(path_1,path_2)
            
            bshcmd.split()
            process = subprocess.Popen(bshcmd, universal_newlines=True, shell=True,
                                       stdout=sys.stdout, stderr=sys.stdout) 
            outs, errs = process.communicate()  
         
def split_segments(trk_path, nifti, selected_list, f_path=my_f_path, dens=None, type_="prob", folder__="", path_prefix="", separator="_"):
  
    print("segments")
    print(selected_list)
    selected_seg_list = [{} for tt in selected_list]
    separator = "_" if len(selected_seg_list)==2 else ","
    for ise, selected_segments in enumerate(selected_list): 
        new_dens = np.zeros_like(dens).astype(float)
        if selected_segments=={}:
            return None
        for roi, value in selected_segments.items():    
            roi = roi.replace("(","").replace(")","").replace("'","").replace(" ","")
            bdle = separator.join(roi.split(separator)[:-2])    
            seg = roi.split(separator)[-2]    
            if not (bdle in selected_seg_list[ise].keys()):
                selected_seg_list[ise][bdle] = {} 
            selected_seg_list[ise][bdle][seg] = value
    
    print("",file=open("/auto/home/users/d/r/drimez/afq_1.txt","w"))
    print("Density") 
    dens = [None for _ in selected_seg_list] ; metrics = ["pc0","pc1"] if len(selected_seg_list)==2 else ["FA","MD","AD","RD"]
    for selected_seg, ise in zip(selected_seg_list,metrics):
     
        path_prefix_ = "" if path_prefix is None else path_prefix + "_"
        folder = path_prefix_ + folder__
        path_1 = "/auto/home/users/d/r/drimez/Classify_results/%s/"%type_ + folder + "significant_" + ise
        path_2 = "/auto/home/users/d/r/drimez/Classify_results/%s/"%type_ + folder + "unsignificant_" + ise
        
        if not os.path.exists("/auto/home/users/d/r/drimez/Classify_results/%s/"%type_ + folder + "/ant_results_light_%s.png"%ise) and selected_seg!={}:
            to_pass = False
            if not os.path.exists(path_2): 
                vol_dims = None ; affine = None ; dens = None ; save_affine = None
                with os.scandir(trk_path) as iterator:
                    for entry in iterator:
                        print(entry.name)
                        if "trk" in entry.name.split("."):  
                            new_entry_name = entry.name.replace(".trk","").replace("_"+ise,"")[1:]
                            print(new_entry_name)
                            if np.any([len(new_entry_name.split(____))==2 for ____ in selected_seg.keys()]):
                                streamlines = load_trk(entry.path,nifti) 
                                trk, streamlines = reorient(streamlines, nifti, trk_type=type_)
                                if trk is None:
                                    print(new_entry_name,"not found")
                                else:
                                    try:
                                        img, affine, vox_sizes = load_nifti(nifti, return_voxsize=True)  
                                        new_vox_sizes = np.array([___/4 for ___ in vox_sizes])
                                        vol_dims = np.array([img.shape[0]*4,img.shape[1]*4,img.shape[2]*4]).flatten()
                                        
                                        _, affine = reslice(img, affine, vox_sizes, new_vox_sizes) 
                                          
                                        # affine[:3,-1] += new_vox_sizes*np.array([_/4 for _ in vol_dims])*affine[:3,-1]/abs(affine[:3,-1])
                                        # print(np.linalg.inv(affine),file=open("/auto/home/users/d/r/drimez/afq.txt","a"))
                                        """
                                        save_affine = copy.copy(affine)
                                        # affine[:3,:3] *= (np.ones((3,3)) - np.eye(3)*(3/4)) 
                                        affine[:3,:3] /= (np.ones((3,3)) + np.eye(3)*3) 
                                        affine[:3,-1] += (np.ones((3,3)) + np.eye(3)*3) 
                                        save_affine[:3,:3] *= (np.ones((3,3)) + np.eye(3)*3) 
                                        print(affine)
                                        """ 
                                        this_sl = [np.array(sl_) for sl_ in streamlines if (len(sl_)>=2 and (isinstance(sl_[0],list) or isinstance(sl_[0],np.ndarray)) )] 
                                        sl_size = len(this_sl) 
                                        print(sl_size) 
                                         
                                        n_seg = len([0 for _ in pd.read_csv("/auto/home/users/d/r/drimez/data_all_prob.csv").columns.values if len(_.split(new_entry_name))==2 and _.split(",")[-1][2:-2]=="FA"]) 
                                        lengths = length(this_sl) ; mean_length = None ; this_sl = np.array(this_sl)
                                        if not isinstance(lengths,np.ndarray):
                                            mean_length = lengths
                                            lengths = np.array([lengths])
                                        else: 
                                            standard = this_sl[np.argmin(abs(lengths-np.quantile(lengths,0.625)))] # sets reference streamline to the mid-second-quartile length (long enough to be in the bundle, short enough to be "straight")
                                            this_sl = orient_by_streamline(this_sl, standard, n_points=100)        # reorient streamlines for correct estimation of mean bundle 
                                            mean_length = np.mean(lengths)
                                             
                                        seglen = mean_length/n_seg 
                                        print(mean_length,n_seg)
                                               
                                        new_sl = None ; centroids = None
                                        if len(this_sl)!=0:
                                            if isinstance(this_sl[0],list) or isinstance(this_sl[0],np.ndarray):
                                                if not (isinstance(this_sl[0][0],list) or isinstance(this_sl[0][0],np.ndarray)): # it is not a list of streamlines but a single streamline
                                                    this_sl = [this_sl]
                                            
                                        if len(this_sl)>50:                      
                                            temp_tg = StatefulTractogram.from_sft( this_sl, trk)  
                                            new_sl_tg = _resample_tg(temp_tg,1 + 2*n_seg)  
                                            this_prof_weights = gaussian_weights(new_sl_tg, n_points=n_seg) 
                                            new_sl = np.array(new_sl_tg) 
                                            centroids = np.mean((new_sl*this_prof_weights)[:,1::2],0)  # centers of segments
                                        elif len(this_sl)>1:  
                                            trk = StatefulTractogram.from_sft( this_sl, trk) 
                                            new_sl_tg = _resample_tg(trk,1 + 2*n_seg)  
                                            new_sl = np.array(new_sl_tg)  
                                            centroids = np.mean(new_sl[:,1::2],0)  # centers of segments
                                        else: 
                                            this_sl = [this_sl]
                                            temp_tg = StatefulTractogram.from_sft( this_sl, trk) 
                                            new_sl_tg = _resample_tg(temp_tg,1 + 2*n_seg)  
                                            new_sl = np.array([new_sl_tg])
                                            centroids = new_sl[0,1::2]    # centrers of segments
                                            
                                        lengths_ = [np.linalg.norm(end__-start__) for end__, start__ in zip(np.mean(new_sl[:,::2],0)[:-1],np.mean(new_sl[:,::2],0)[1:])] # start and ends of segments
                                        assignments_ = []
                                        for s in this_sl: 
                                            assignments_.append([np.argmin([np.linalg.norm(si-centroid) for centroid in centroids]) for si in s])   
                                        new_sl = None 
                                        
                                        seg_sl_assignments = assignments_
                                        segments = [[] for nnn_seg in range(n_seg)] 
                                        for assigned_sl in range(sl_size):
                                            split_sl = this_sl[assigned_sl]
                                            for nnn_seg in range(n_seg): 
                                                to_append = split_sl[np.array(seg_sl_assignments[assigned_sl])==nnn_seg]
                                                if len(to_append)>0:
                                                    if not (isinstance(to_append[0],list) or isinstance(to_append[0],np.ndarray)):
                                                        to_append = [to_append]  
                                                to_append = [_ for _ in to_append if len(_)>0]
                                                if len(to_append)>0:
                                                    segments[nnn_seg].append(to_append)   
                                           
                                        for iseg, segment in enumerate(segments): 
                                            if str(iseg) in [*selected_seg[new_entry_name].keys()]: 
                                                tdm = density_map(segment, affine, vol_dims) 
                                                if dens is None: 
                                                    dens = tdm*selected_seg[new_entry_name][str(iseg)]/sl_size
                                                else:
                                                    dens += tdm*selected_seg[new_entry_name][str(iseg)]/sl_size
                                    except Exception as err_:
                                        raise
                
                if not (dens is None):   
                    new_1 = copy.copy(dens)
                    new_1[dens<0] = 0
                    min_1, max_1 = np.abs([new_1.min(),new_1.max()])
                    
                    new_2 = copy.copy(dens)
                    new_2[dens>0] = 0
                    new_2 = -new_2 
                    new_2 += new_1
                    min_2, max_2 = np.abs([new_2.min(),new_2.max()])
                    max_ = np.max([max_1,max_2])
                    print(max_,file=open("/auto/home/users/d/r/drimez/afq_1.txt","a"))
                    
                    if not os.path.isdir("/auto/home/users/d/r/drimez/Classify_results/%s/"%type_):
                        os.makedirs("/auto/home/users/d/r/drimez/Classify_results/%s/"%type_)
                        
                    save_nifti(path_1+".nii.gz",new_1, affine) 
                    save_nifti(path_2+".nii.gz",new_2, affine) 
                else:
                    to_pass = True
            
            if not to_pass:
                print("fsleyes")
                t1_path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/elikopy_subset_new2/PROJECT/subjects/C_1/T1/C_1_T1_corr_projected" 
                bshcmd = "cp " + t1_path + ".nii.gz /auto/home/users/d/r/drimez/Classify_results/%s/ ; "%type_
                 
                t1_path = "/auto/home/users/d/r/drimez/Classify_results/%s_old/C_1_T1_corr_projected"%type_
                bshcmd = "cd /auto/home/users/d/r/drimez/Classify_results/%s/ ; "%type_ 
                 
                if not os.path.isdir("/auto/home/users/d/r/drimez/Classify_results/%s/%s/"%(type_,folder)):
                    os.makedirs("/auto/home/users/d/r/drimez/Classify_results/%s/%s/"%(type_,folder))
                bshcmd += "fsleyes render --scene ortho --size 2000 2000 --crop 20 -of " + folder + "/results_ortho_%s.png --hideCursor "%ise + t1_path + " -a 40 "
                
                bsh = copy.copy(bshcmd)
                bsh.split()
                process = subprocess.Popen(bsh, universal_newlines=True, shell=True,
                                           stdout=sys.stdout, stderr=sys.stdout) 
                outs, errs = process.communicate()  
                
                if max_1>0:
                    bshcmd += path_1 + " -cm blue-lightblue -a 50 -dr %s %s -in linear "%(0,max_)  
                if max_2>0:
                    bshcmd += path_2 + " -cm red-yellow -a 70 -dr %s %s -in linear --logScale ; "%(0,max_) 
                
                bshcmd.split()
                process = subprocess.Popen(bshcmd, universal_newlines=True, shell=True,
                                           stdout=sys.stdout, stderr=sys.stdout) 
                outs, errs = process.communicate()  
                
                bshcmd = "cd /auto/home/users/d/r/drimez/Classify_results/%s/ ; "%type_ 
                bshcmd += "fsleyes render --scene lightbox --size 8000 8000 --hideCursor --crop 20 -zx X -ss 2 -nr 20 -nc 5 -of " + folder + "/side_results_light_%s.png "%ise + t1_path + " -a 40 " + path_1 + " -cm blue-lightblue -a 50 -dr %s %s -in linear "%(0,max_)  + path_2 + " -cm red-yellow -a 70 -dr %s %s -in linear --logScale ; "%(0,max_) 
                
                bshcmd.split()
                process = subprocess.Popen(bshcmd, universal_newlines=True, shell=True,
                                           stdout=sys.stdout, stderr=sys.stdout) 
                outs, errs = process.communicate()  
                
                bshcmd = "cd /auto/home/users/d/r/drimez/Classify_results/%s/ ; "%type_ 
                bshcmd += "fsleyes render --scene lightbox --size 8000 8000 --hideCursor --crop 20 -zx Y -ss 3 -nr 20 -nc 5 -of " + folder + "/ant_results_light_%s.png "%ise + t1_path + " -a 40 " + path_1 + " -cm blue-lightblue -a 50 -dr %s %s -in linear "%(0,max_)  + path_2 + " -cm red-yellow -a 70 -dr %s %s -in linear --logScale ; "%(0,max_)   
                # bshcmd += "rm /auto/home/users/d/r/drimez/Classify_results/%s/%s.nii.gz ; "%(type_,t1_path)
                bshcmd += "rm %s.nii.gz %s.nii.gz ; "%(path_1,path_2)
                
                bshcmd.split()
                process = subprocess.Popen(bshcmd, universal_newlines=True, shell=True,
                                           stdout=sys.stdout, stderr=sys.stdout) 
                outs, errs = process.communicate()  
        
def selection(path, data_type, pval, n_roi=10000):

    thresh = 0.001 if len(path.split("wmparc"))==2 else 0.0001
    selected_features = [] ; neg_selected_features = [] ; less_than_10 = [] ; neg_less_than_10 = []  ; perfs = []
    metrics = ["pc0","pc1"] if "pca" in data_type.split("_") else ["FA","MD","AD","RD"]
    for feature_type in metrics:
        selected_features.append({}) ; neg_selected_features.append({}) ; less_than_10.append({}) ; neg_less_than_10.append({}) ; success_list_dir = []
        if os.path.isdir(path) and len(path.split(data_type))>=2:
            list_dir = [] ; print(path)
            with os.scandir(path) as _:
                for __ in _:
                    list_dir.append(__.path)
            reliable_perfs = None  
            for file_path in list_dir:
                if len(file_path.split("reliable"))==2 and (not len(file_path.split("unreliable"))==2) \
                   and np.any([True if len(_file_.split("results"))==2 else False for _file_ in list_dir]): 
                    reliable_perfs = pd.read_csv( file_path, sep="},", skiprows=0,
                                                  names=["fitted params","perfs"]).drop_duplicates()  
                    
                    this_params = []
                    for this_par in reliable_perfs.values[1:,0]:
                        params_string = ""
                        for par_name, params_val in zip([_.split(": ")[0] for _ in (this_par+"}").replace("{","").replace("}","").split(", ")],
                                                        [_.split(": ")[1] for _ in (this_par+"}").replace("{","").replace("}","").split(", ")]):
                            params_string += str(par_name) + "=" + str(params_val) + "_"
                        this_params.append(params_string)
                     
                    results_folder = np.array([True if len(_file_.split("results"))==2 else False for _file_ in list_dir]) 
                    success_folder = np.array(list_dir)[results_folder]
                    if isinstance(success_folder,list) or isinstance(success_folder,np.ndarray):
                        success_folder = success_folder[0]
                        
                    success_file = os.listdir(success_folder) ; pvalues = [] ; temp_success_list_dir = []
                    for __pval in success_file: 
                        for a_param in this_params:  
                            if str(__pval).replace(".txt","").replace("b'","").replace("'","")==a_param.replace("'",""): 
                                temp_success_list_dir.append(success_folder+"/"+str(__pval).replace("b'","").replace("'",""))
                                pvalues.append(pd.read_csv(success_folder+"/"+str(__pval).replace("b'","").replace("'","")).values[2,-1])
                    
                    print(pvalues)
                    for par, lign, pvalue, file__ in zip(reliable_perfs.values[1:,0], reliable_perfs.values[1:,1],pvalues,temp_success_list_dir): 
                        if pvalue<=0.06:   
                            perf_metrics = {key:float(val) for key, val in zip([_.split(": ")[0].replace("'","") for _ in lign.replace("{","").replace("}","").split(", ")],
                                                                  [_.split(": ")[1].replace("'","") for _ in lign.replace("{","").replace("}","").split(", ")]) } 
                            perfs.append(perf_metrics) 
                            success_list_dir.append(file__) 
                                 
            print(perfs)       
            print(success_list_dir)         
            for mask, success_file in enumerate(success_list_dir):
                results = pd.read_csv(success_file)    
                if np.sum(results["true_mean"].values.flatten()>=thresh)<=n_roi:
                    imp = results["true_mean"].values
                    ten_or_less = False
                    if np.sum(imp>0)<=10:
                        ten_or_less = True
                    for name, lign in zip(results.values[:,0].flatten(),results["true_mean"].values.flatten()):  
                        if len(name.split(feature_type))==2:  
                            diff_type = pval.values[:,1][pval.values[:,0]==name] 
                            sign = 1 if np.any(diff_type<=0.05) else -1  
                            if lign>=thresh:
                                if not name in selected_features[-1].keys():
                                    selected_features[-1][name] = sign*lign*float(perfs[mask]["mean_test_balanced_accuracy"])
                                else:
                                    selected_features[-1][name] += sign*lign*float(perfs[mask]["mean_test_balanced_accuracy"])
                                if ten_or_less:
                                    if not name in less_than_10[-1].keys():
                                        less_than_10[-1][name] = sign*lign*float(perfs[mask]["mean_test_balanced_accuracy"])
                                    else:
                                        less_than_10[-1][name] += sign*lign*float(perfs[mask]["mean_test_balanced_accuracy"])
                            elif lign<=-thresh:
                                if not name in neg_selected_features[-1].keys():
                                    neg_selected_features[-1][name] = sign*lign*float(perfs[mask]["mean_test_balanced_accuracy"])
                                else:
                                    neg_selected_features[-1][name] += sign*lign*float(perfs[mask]["mean_test_balanced_accuracy"])
                                if ten_or_less:
                                    if not name in neg_less_than_10[-1].keys():
                                        neg_less_than_10[-1][name] = sign*lign*float(perfs[mask]["mean_test_balanced_accuracy"])
                                    else:
                                        neg_less_than_10[-1][name] += sign*lign*float(perfs[mask]["mean_test_balanced_accuracy"]) 
                            
    return selected_features, neg_selected_features, less_than_10, neg_less_than_10
    
    
def select_age(path, wmparc_path, path_prefix="age"):

    age_coeffs = pd.read_csv(path,sep=": ").dropna()
    selected_features = [{} for i in range(4)]
    for name, coeff in age_coeffs.values:
        if len(name.split("FA"))==2 and not len(name.split("std"))==2:
            selected_features[0][name.split("'")[3]] = float(coeff) 
        elif len(name.split("MD"))==2 and not len(name.split("std"))==2:
            selected_features[1][name.split("'")[3]] = float(coeff)
        elif len(name.split("AD"))==2 and not len(name.split("std"))==2:
            selected_features[2][name.split("'")[3]] = float(coeff)
        elif len(name.split("RD"))==2 and not len(name.split("std"))==2:
            selected_features[3][name.split("'")[3]] = float(coeff)
    
    wmparc, affine = load_nifti(wmparc_path)
        
    LUT = pd.read_csv("/auto/home/users/d/r/drimez/LUT.txt",sep="  ",header=None,index_col=False,names=["id","name","r","g","b","a"])  
    names = np.array([_ for _ in LUT["name"].values])
    lut_index = np.arange(len(LUT["id"].values))
      
    new_wmparc = np.zeros_like(wmparc).astype(float)
    selected_features_list = selected_features
    for selected_features, ise in zip(selected_features_list,["FA","MD","AD","RD"]):
        for roi, value in selected_features.items():   
            corresponding = lut_index[np.array(names==roi)]  
            # corresponding = corresponding[0]  
            label = LUT["id"].values[corresponding] 
            new_wmparc[wmparc==label] = value 
        
        if not os.path.isdir("/auto/home/users/d/r/drimez/Classify_results/wmparc/"):
            os.makedirs("/auto/home/users/d/r/drimez/Classify_results/wmparc/")
        
        new_wmparc_1 = copy.copy(new_wmparc)
        new_wmparc_1[new_wmparc<0] = 0
        min_1, max_1 = np.abs([new_wmparc_1.min(),new_wmparc_1.max()])
        
        new_wmparc_2 = copy.copy(new_wmparc)
        new_wmparc_2[new_wmparc>0] = 0
        new_wmparc_2 = -new_wmparc_2 
        min_2, max_2 = np.abs([new_wmparc_2.min(),new_wmparc_2.max()])
        max_ = np.max([max_1,max_2])
         
        path_prefix_ = "" if path_prefix is None else path_prefix + "_"
        path_1 = "/auto/home/users/d/r/drimez/Classify_results/" + path_prefix_ + "wmparc_significant"
        path_2 = "/auto/home/users/d/r/drimez/Classify_results/" + path_prefix_ + "wmparc_unsignificant"
        save_nifti(path_1+".nii.gz",new_wmparc_1,affine)
        save_nifti(path_2+".nii.gz",new_wmparc_2,affine)
        
        if not os.path.isdir("/auto/home/users/d/r/drimez/Classify_results/wmparc/"+path_prefix+"/"):
            os.makedirs("/auto/home/users/d/r/drimez/Classify_results/wmparc/"+path_prefix+"/")
        
        t1_path = "/".join(wmparc_path.split("/")[:-3]) + "/%s_T1_corr_projected"%wmparc_path.split("/")[-4]
        t1_path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/elikopy_subset_new2/PROJECT/subjects/C_1/T1/C_1_T1_corr_projected" 
        bshcmd = "cd /auto/home/users/d/r/drimez/Classify_results/wmparc/ ;"
        bshcmd += "fsleyes render --scene ortho --size 2000 2000 --hideCursor --crop 20 -of " + path_prefix + "/results_ortho_%s.png "%ise + t1_path + " -a 50 " + path_1 + " -cm red-yellow -a 50 -dr %s %s "%(0,max_) + path_2 + " -cm blue-lightblue -a 50 -dr %s %s ; "%(0,max_)
        bshcmd += "fsleyes render --scene lightbox --size 8000 8000 --hideCursor --crop 20 -zx X -ss 2 -nr 20 -nc 5 -of " + path_prefix + "/side_results_light_%s.png "%ise + t1_path + " -a 50 " + path_1 + " -cm red-yellow -a 50 -dr %s %s "%(0,max_) + path_2 + " -cm blue-lightblue -a 50 -dr %s %s ; "%(0,max_)
        bshcmd += "fsleyes render --scene lightbox --size 8000 8000 --hideCursor --crop 20 -zx Y -ss 3 -nr 20 -nc 5 -of " + path_prefix + "/ant_results_light_%s.png "%ise + t1_path + " -a 50 " + path_1 + " -cm red-yellow -a 50 -dr %s %s "%(0,max_) + path_2 + " -cm blue-lightblue -a 50 -dr %s %s ; "%(0,max_) 
        
        bshcmd.split()
        process = subprocess.Popen(bshcmd, universal_newlines=True, shell=True,
                                   stdout=sys.stdout, stderr=sys.stdout) 
        outs, errs = process.communicate() 
         
def select_age_trk(path, trk_path, nifti, type_="prob", path_prefix="age"):

    age_coeffs = pd.read_csv(path,sep=": ").dropna()
    selected_features = [{} for i in range(4)]
    for name, coeff in age_coeffs.values:
        if len(name.split("FA"))==2 and not len(name.split("std"))==2:
            selected_features[0][name] = float(coeff)*1000
        elif len(name.split("MD"))==2 and not len(name.split("std"))==2:
            selected_features[1][name] = float(coeff)*1000
        elif len(name.split("AD"))==2 and not len(name.split("std"))==2:
            selected_features[2][name] = float(coeff)*1000
        elif len(name.split("RD"))==2 and not len(name.split("std"))==2:
            selected_features[3][name] = float(coeff)*1000
     
    if not os.path.isdir("/auto/home/users/d/r/drimez/Classify_results/%s/age/"%type_):
        os.makedirs("/auto/home/users/d/r/drimez/Classify_results/%s/age/"%type_)
            
    split_segments_age(trk_path, nifti, selected_features, type_=type_, path_prefix=path_prefix, separator=",")
           
        
wmparc_path = my_f_path + "subjects/C_1/tracking/preproc/C_1_reg_wmparc.nii.gz"
data_type = "scale_pca_without_age" ; model_type = "VHIT"
pvalues = pd.read_csv("pvalues_wmparc_scale_pca_without_age.txt")

select_age("/auto/home/users/d/r/drimez/wmparc_age_coef.txt", wmparc_path, path_prefix="age") 

trk_path = my_f_path + "subjects/C_1/tracking/AFQ/tracks_dipy_prob/"
select_age_trk("/auto/home/users/d/r/drimez/prob_age_coef.txt", trk_path, nifti, type_="prob", path_prefix="age") 
"""
# 
import gc
for with_or_without_age in ["_without_age"]:
    for data_type in ["scale_pca","select"]:  
        this_data_type = "_"+data_type if data_type!="select" else ""
        this_data_type += with_or_without_age 
        try:
            pvalues = pd.read_csv("pvalues_%s%s.txt"%("wmparc",this_data_type))
            for model_type in ["vertigo","cVemp_R","cVemp_L","oVemp_R","oVemp_L","VHIT_ant_R","VHIT_ant_L","VHIT_lat_R",
                               "VHIT_lat_L","VHIT_post_R","VHIT_post_L","VNG_cal_R","VNG_cal_L","VNG_rot","Posturo"]:
                with os.scandir("/auto/home/users/d/r/drimez/Classify_wmparc_final/") as this_iterator:    
                    for target_entry in this_iterator: 
                        if os.listdir(target_entry.path) and len(target_entry.name.split(model_type))==2:
                            gc.collect()
                            selected_features, neg_selected_features, less_than_10, neg_less_than_10 = selection(target_entry.path, data_type, pvalues)
                             
                            target_type = "_".join(target_entry.name.replace("_"+data_type,"").split("_")[1:-1])
                            parc_rois(wmparc_path, selected_features, path_prefix="", folder__=model_type+with_or_without_age+"_"+data_type)
                            parc_rois(wmparc_path, neg_selected_features, path_prefix="neg", folder__=model_type+with_or_without_age+"_"+data_type)
                            parc_rois(wmparc_path, less_than_10, path_prefix="10", folder__=model_type+with_or_without_age+"_"+data_type)
                            parc_rois(wmparc_path, neg_less_than_10, path_prefix="neg_10", folder__=model_type+with_or_without_age+"_"+data_type)
        except Exception as err:
            print(err)
            raise
""" 
"""
import gc
trk_path = my_f_path + "subjects/C_1/tracking/AFQ/tracks_dipy_prob/"
nifti = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/elikopy_subset_new2/PROJECT/subjects/C_1/T1/C_1_T1_corr_projected.nii.gz"  

for target in ["vertigo","cVemp_R","cVemp_L","oVemp_R","oVemp_L","VHIT_ant_R","VHIT_ant_L","VHIT_lat_R",
                 "VHIT_lat_L","VHIT_post_R","VHIT_post_L","VNG_cal_R","VNG_cal_L","VNG_rot","Posturo"]: 
    gc.collect()
    # compare_models("/auto/home/users/d/r/drimez/Classify_tract_prob_final/", target, type__="prob")  
    gc.collect()
    compare_models("/auto/home/users/d/r/drimez/Classify_wmparc_final/", target, type__="wmparc") 
""" 
""" 
""" 
""" 
import gc
def plot_trk(data_type, with_or_without_age, trk_type,trk_path,nifti,model_type):
    this_data_type = data_type if data_type!="select" else "all"
    this_data_type += with_or_without_age 
    try:
        pvalues = pd.read_csv("pvalues_%s_%s.txt"%(trk_type,this_data_type)) 
        with os.scandir("/auto/home/users/d/r/drimez/Classify_tract_%s_final/"%trk_type) as this_iterator:    
            for target_entry in this_iterator: 
                if os.listdir(target_entry.path) and len(target_entry.name.split(model_type))==2 \
                   and len(target_entry.name.split(data_type))==2 and len(target_entry.name.split(with_or_without_age+"_"))>=2: 
                    gc.collect()
                    selected_features, neg_selected_features, less_than_10, neg_less_than_10 = selection(target_entry.path, data_type+"_"+with_or_without_age, pvalues)
                    print("Displaying") 
                    split_segments(trk_path, nifti, selected_features, type_=trk_type, folder__=model_type+with_or_without_age+"_"+data_type, path_prefix="")
                    split_segments(trk_path, nifti, neg_selected_features, type_=trk_type, folder__=model_type+with_or_without_age+"_"+data_type, path_prefix="neg") 
                    split_segments(trk_path, nifti, less_than_10, type_=trk_type, folder__=model_type+with_or_without_age+"_"+data_type, path_prefix="10")  
                    split_segments(trk_path, nifti, neg_less_than_10, type_=trk_type, folder__=model_type+with_or_without_age+"_"+data_type, path_prefix="neg_10") 
    except Exception as err:
        print(err)
        raise

nifti = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/elikopy_subset_new2/PROJECT/subjects/C_1/T1/C_1_T1_corr_projected.nii.gz" 
data_type = "scale_pca_without_age" ; model_type = "vertigo"

trk_type = "prob" ; data_type = "select" # "scale_pca" _without_age
with_or_without_age = ""
trk_path = my_f_path + "subjects/C_1/tracking/AFQ/tracks_dipy_prob/" # if trk_type=="prob" else my_f_path + "subjects/C_1/tracking/AFQ/tracks_dipy/" 
            
Parallel(n_jobs=-1,verbose=10,pre_dispatch='2*n_jobs',require="sharedmem")(
                 delayed(plot_trk)(data_type, with_or_without_age, trk_type,trk_path,nifti,model_type) for model_type in ["vertigo","cVemp_R","cVemp_L","oVemp_R","oVemp_L","VHIT_ant_R","VHIT_ant_L","VHIT_lat_R", "VHIT_lat_L","VHIT_post_R","VHIT_post_L","VNG_cal_R","VNG_cal_L","VNG_rot","Posturo"] 
      )     
 
"""
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        