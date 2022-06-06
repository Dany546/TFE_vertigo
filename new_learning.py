# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 01:43:24 2022

@author: rimez
"""
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
from numpy import nan
import os
import sys
from sklearn.multioutput import MultiOutputClassifier
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io
import random
import pandas as pd
import tables
import seaborn as sns 
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from matplotlib.gridspec import  GridSpec
from matplotlib.legend_handler import HandlerTuple  
from sklearn.metrics import accuracy_score, r2_score, median_absolute_error, roc_auc_score
from string import Template
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from textwrap import wrap 
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import seaborn as sns
import bz2 
import multiprocessing 
from joblib import Parallel, delayed
import _pickle as cPickle
from afqinsight.datasets import download_sarica, load_afq_data
from afqinsight import make_afq_classifier_pipeline, cross_validate_checkpoint
from groupyr_stolen import *
from groupyr_stolen import _stringify_sequence
import groupyr
from groupyr import LogisticSGLCV
from groupyr.decomposition import GroupPCA 
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
import afqinsight as afqi
import joblib 
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle
from itertools import product
import seaborn as sns
from abc import ABCMeta, abstractmethod
from sklearn.ensemble import *
from sklearn.ensemble._base import BaseEnsemble
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
import copy 
import gc
from sklearn.ensemble._base import BaseEnsemble, _partition_estimators
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state, column_or_1d, deprecated
from sklearn.utils import indices_to_mask
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import has_fit_parameter, check_is_fitted, _check_sample_weight
from sklearn.utils.fixes import delayed
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import (
    TimeSeriesSplit,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    GroupShuffleSplit,
    GroupKFold,
    StratifiedShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import numbers
from typing import List
from sklearn.inspection import permutation_importance
from sklearn.model_selection import permutation_test_score
import numpy as np
from sklearn.utils import *
from sklearn.base import *
from joblib import effective_n_jobs
from skopt import BayesSearchCV
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
import warnings
warnings.filterwarnings("ignore")
import faulthandler; faulthandler.enable()
 
refresh = False

"""
checkpoint_path = "/auto/home/users/d/r/drimez/Classify%s/checkpoints"%"_bagging"
checkpoint_path_list = [checkpoint_path + "_SVM", checkpoint_path + "_SGL", checkpoint_path + "_RandForest", 
                        checkpoint_path + "_Lasso", checkpoint_path + "_Ridge", checkpoint_path + "_ElasticNet"]
for checkpt_path in checkpoint_path_list:
    if not os.path.isdir(checkpt_path):
        os.mkdir(checkpt_path)
        
checkpoint_path = "/auto/home/users/d/r/drimez/Classify%s/checkpoints"%""
checkpoint_path_list = [checkpoint_path + "_SVM", checkpoint_path + "_SGL", checkpoint_path + "_RandForest", 
                        checkpoint_path + "_Lasso", checkpoint_path + "_Ridge", checkpoint_path + "_ElasticNet"]
for checkpt_path in checkpoint_path_list:
    if not os.path.isdir(checkpt_path):
        os.mkdir(checkpt_path)
"""
f_path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/" 
f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset/PROJECT/"  

patient_list = [ 'C_0', 'C_1', 'C_11', 'C_12', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6',
                 'C_7', 'C_8', 'C_9', 'H_0', 'H_1', 'H_2', 'H_3', 'H_5', 'H_6',
                 'V_0', 'V_1', 'V_10', 'V_11', 'V_13', 'V_14', 'V_15', 'V_16',
                 'V_17', 'V_18', 'V_19', 'V_2', 'V_20', 'V_21', 'V_22', 'V_23',
                 'V_24', 'V_25', 'V_26', 'V_27', 'V_28', 'V_29', 'V_3', 'V_30',
                 'V_32', 'V_33', 'V_34', 'V_35', 'V_36', 'V_37', 'V_38', 'V_39',
                 'V_4', 'V_40', 'V_41', 'V_42', 'V_43', 'V_44', 'V_45', 'V_46',
                 'V_47', 'V_48', 'V_49', 'V_5', 'V_50', 'V_51', 'V_52', 'V_53',
                 'V_6', 'V_7', 'V_8', 'V_9' ]

MAX_INT = np.iinfo(np.int32).max
  
def traverse(D): # inspired from https://stackoverflow.com/questions/41523543/how-can-i-create-a-list-of-possible-combination-in-a-dict
    if not isinstance(D,list):
        D = [D]
    for d in D:
        K,V = zip(*d.items())
        for v in product(*(v if isinstance(v,list) else traverse(v) for v in V)):
            yield dict(zip(K,v))

def compressed_pickle(title, data):
   with bz2.BZ2File(title + '.pbz2', 'w') as f: 
       cPickle.dump(data, f)
       f.close()
       
def decompress_pickle(filename):
   data = bz2.BZ2File(filename, 'rb')
   data = cPickle.load(data)
   return data
"""  
def generate_data(subjects=patient_list,f_path=f_path,old_subjects=old_subjects):
    new_df = None ; errors = []
    if os.path.exists("metrics_new.csv"):
        new_df = pd.read_csv("metrics_new.csv", index_col=False)
    else: 
        df = pd.DataFrame()
        for i_sub, sub in enumerate(old_subjects):
            csv_path = f_path + "subjects/" + sub + "/tracking/ROIs/metrics.csv"
            if os.path.exists(csv_path):
                this_df = pd.read_csv(csv_path,header=0)
                this_df['subjectID'].values[:] = subjects[i_sub]
                df = pd.concat((df,this_df),axis=0)
            else:
                errors.append(sub)
                
        new_df = pd.DataFrame()
        for n_col,col in enumerate(df.columns.values.flatten()):
            if n_col>=3:
                column_data = np.array([[elem.split(',')[0][1:],elem.split(',')[-1][:-1]] for elem in df[col].values])  
                new_df[col] = pd.DataFrame(column_data[:,0], index=df.index)     
                new_df['std_'+col] = pd.DataFrame(column_data[:,1], index=df.index)  
            else:  
                new_df[col] = df[col] 
    
        new_df['nodeID'] = new_df['tractID'].values[:]
        for lign_id, lign in enumerate(df['tractID'].values):
            if np.any([el in ("Cbm","cbm","cerebellum","Cerebellum") for el in lign.split("-")]):
                new_df['tractID'].iloc[lign_id] = "cbm"
            elif np.any([el in ("White","wm","Chiasm","UnsegmentedWhiteMatter","CC","WM","Stem") for el in lign.split("-")])\
                 or "CC" in lign.split("_"):
                new_df['tractID'].iloc[lign_id] = "wm"
            elif np.any([el in ("ctx","Cortex") for el in lign.split("-")]):
                new_df['tractID'].iloc[lign_id] = "ctx"
            elif np.any([el in ("CSF","VentralDC","vessel","choroid","ventricle","Ventricle","Vent") for el in lign.split("-")]):
                new_df['tractID'].iloc[lign_id] = "others"
            else:
                new_df['tractID'].iloc[lign_id] = "subctx"
                
        new_df.to_csv("/auto/home/users/d/r/drimez/metrics_new.csv", line_terminator="\n", sep=",", index=False)
    
    if len(errors)>=1:
        print("Subjects not found: "+str(errors))
        
    return new_df

def generate_afq_data(subjects=patient_list,f_path=f_path,old_subjects=old_subjects):
    new_df = None ; errors_det = [] ; errors_prob = []
    if os.path.exists("metrics_afq.csv"):
        new_df = pd.read_csv("metrics_afq.csv", index_col=False)
    else: 
        df_det = pd.DataFrame() ; df_prob = pd.DataFrame()
        for i_sub, sub in enumerate(subjects):
            
            csv_paths = f_path + "subjects/" + sub + "/tracking/AFQ/" + sub + "_dipy.csv"
            if os.path.exists(csv_path):
               this_df = pd.read_csv(csv_path,header=0)
               this_df['subjectID'].values[:] = subjects[i_sub]
               df_det = pd.concat((df_det,this_df),axis=0)
            else:
               errors_det.append(sub)
                
            csv_paths = f_path + "subjects/" + sub + "/tracking/AFQ/" + sub + "_dipy_prob.csv"
            if os.path.exists(csv_path):
               this_df = pd.read_csv(csv_path,header=0)
               this_df['subjectID'].values[:] = subjects[i_sub]
               df_prob = pd.concat((df_prob,this_df),axis=0)
            else:
               errors_prob.append(sub)
                
        new_df_det = pd.DataFrame() ; new_df_prob = pd.DataFrame()
        for n_col,col in enumerate(df_det.columns.values.flatten()):
            if n_col>=3:
                column_data = np.array([[elem.split(',')[0][1:],elem.split(',')[-1][:-1]] for elem in df[col].values])  
                new_df[col] = pd.DataFrame(column_data[:,0], index=df.index)     
                new_df['std_'+col] = pd.DataFrame(column_data[:,1], index=df.index)  
            else:  
                new_df[col] = df[col] 
    
        new_df['nodeID'] = new_df['tractID'].values[:]
        for lign_id, lign in enumerate(df['tractID'].values):
            if np.any([el in ("Cbm","cbm","cerebellum","Cerebellum") for el in lign.split("-")]):
                new_df['tractID'].iloc[lign_id] = "cbm"
            elif np.any([el in ("White","wm","Chiasm","UnsegmentedWhiteMatter","CC","WM","Stem") for el in lign.split("-")])\
                 or "CC" in lign.split("_"):
                new_df['tractID'].iloc[lign_id] = "wm"
            elif np.any([el in ("ctx","Cortex") for el in lign.split("-")]):
                new_df['tractID'].iloc[lign_id] = "ctx"
            elif np.any([el in ("CSF","VentralDC","vessel","choroid","ventricle","Ventricle","Vent") for el in lign.split("-")]):
                new_df['tractID'].iloc[lign_id] = "others"
            else:
                new_df['tractID'].iloc[lign_id] = "subctx"
                
        new_df.to_csv("/auto/home/users/d/r/drimez/metrics_new.csv", line_terminator="\n", sep=",", index=False)
    
    if len(errors)>=1:
        print("Subjects not found: "+str(errors))
        
    return new_df
"""
def select_groups(X, select, label_seq, columns=True, all=True):
    index = 1 if columns else 0
    mask = [] ; found = False
    for lab in label_seq: 
        for bund in select: 
            if np.any(np.array(list(lab))==bund[0]) or all:
                found = True
                mask.append(1) 
        if not found:
            mask.append(0)
        found = False     
    mask = np.array(mask)==1  
    if all:
        return mask
    else:
        X_new = X.T[mask].T if columns else X[mask]
        return X_new
    
from sklearn.metrics import multilabel_confusion_matrix

def confusion_matrix_scorer(y_true,y_pred,k=1,*kwargs): 
    try:
        cm = multilabel_confusion_matrix(y_true, y_pred)[k]
        return cm[1,1] /  np.sum(cm[1])
    except Exception:
        return 0

def plot_results(final_results,filename,ensemble_or_not): 

    if isinstance(final_results,str):
        final_results_path = final_results
        final_results = {} ; final_results_ = None
        with open(final_results_path,"r") as reader:
            final_results_ = reader.readlines()
            reader.close()
        perfs_name = [eeeee.replace("\n", "") for eeeee in final_results_[0].split(",")[1:]] 
        for line in final_results_[1:]:
            params, perfs = line.split("},")
            try:
                results_ = {metric_name:float(a_metric) for metric_name, a_metric in zip(perfs_name,perfs.split(","))}
                if (not np.any([res for _,res in results_.items()])==0) \
                    and results_["test_balanced_accuracy"]>=0.35 and \
                   (not results_["train_accuracy"]-results_["test_accuracy"]>=0.2):
                    final_results[params+"}"] = results_
            except Exception:
                pass

    plotting_folder_ = "" if ensemble_or_not is None else "_bagging"
    plotting_folder = "/auto/home/users/d/r/drimez/Classify" + plotting_folder_ + "/"
    plotting_folder = "Classify" + plotting_folder_ + "/"
    
    def get_cv_split_performance(results, pipeline, performance_metric="test_accuracy"):
        args = {keyval.split(":")[0].replace("'", ""):keyval.split(":")[1] for keyval in pipeline[1:-1].split(", ")}
        metrics = dict( **{ "metric": results[performance_metric],
                            "metric_name": performance_metric, 
                            "model": pipeline}, **args )
        return pd.DataFrame([metrics]) #, [*args.keys()]
     
    x_axis = "C" 
    hues = ["scaler","select"]
    filenames_split = filename.split("_")
    if "RandForest" in filenames_split:
        hues.append("max_depth")
        x_axis = "n_estimators"
    elif "ElasticNet" in filenames_split:
        hues.append("l1_ratio")
    elif "SVM" in filenames_split:
        hues.append("kernel")
        
    if ensemble_or_not=="bagging":
        hues.append("n_estimators") 
     
    legend_s = None ; legend_k = None ; legend_c = None
    figs = [] ; axes = [] ; legends = [] ; n_figs = len(hues)
    for nn_figs in range(n_figs):
        fig, axe = plt.subplots(2,2,figsize=(24,9),sharex=True)
        figs.append(fig) ; axes.append(axe.flatten()) ; legends.append(None)
         
    for im, metric in enumerate(["train_accuracy","test_accuracy","test_corrected_accuracy",
                                 "test_balanced_accuracy"]):
        df = pd.concat([ get_cv_split_performance(results=_res, pipeline=_pip, performance_metric=metric)  
                         for _pip,_res in final_results.items() if _res is not None], ignore_index=True)  
        # ax_scaler = df.boxplot(by='scaler',ax=ax_scaler)    
        data_ind = df["metric_name"]
        data_ind = data_ind.values == metric
        data = df.iloc[data_ind]
        for il, (ax, hue) in enumerate(zip([_[im] for _ in axes],hues)):
            ax = sns.boxplot(x=x_axis,y="metric",hue=str(hue), data=data, palette="muted",ax=ax)
            legends[il] = copy.copy(ax.get_legend())._set_loc(2) 
            ax = sns.scatterplot(x=x_axis, y="metric", hue=hue, data=data, palette="muted", ax=ax,size=2)  
            ax.set_title(metric) 
            ax.get_legend().remove()
            
    for fig, hue in zip(figs,hues):
        fig.savefig(plotting_folder + filename + "_%s.png"%hue)
    # fig_scaler.legend(labels=np.unique(df["scaler"].values).tolist(), loc = 2)
    # fig_scaler.add_artist(legend_s) 
    # fig_scaler.savefig(plotting_folder + filename + "_scaler.png")
    # fig_kernel.legend(labels=np.unique(df["kernel"].values).tolist(), loc = 2)
    # fig_kernel.add_artist(legend_k).get_legend()._set_loc(2)
    # fig_kernel.savefig(plotting_folder + filename + "_kernel.png") 
    # fig_c.legend(labels=np.unique(df["select"].values).tolist(), loc = 2)
    # fig_c.add_artist(legend_c).get_legend()._set_loc(2)
    # fig_c.savefig(plotting_folder + filename + "_select.png")
    
def plot_results_SGD(final_results,filename,ensemble_or_not): 

    plotting_folder = "" if ensemble_or_not is None else "_bagging"
    
    def get_cv_split_performance(results, pipeline, performance_metric="test_accuracy"):
        metrics = { "metric": results[performance_metric],
                    "metric_name": performance_metric, 
                    "model": pipeline, 
                    "scaler":pipeline.split(",")[0].split(":")[-1],
                    "l1_ratio":pipeline.split(",")[-1].split(":")[-2].split("}")[0],
                    "alpha":pipeline.split(",")[1].split(":")[-1],
                    "select":pipeline.split(",")[-1].split(":")[-1].split("}")[0]}   
        return pd.DataFrame(metrics)
    
    
    fig_scaler, axes_scaler = plt.subplots(2,2,figsize=(18,9),sharex=True)
    fig_kernel, axes_kernel = plt.subplots(2,2,figsize=(18,9),sharex=True)
    fig_c, axes_c = plt.subplots(2,2,figsize=(18,12),sharex=True)
    for metric, ax_scaler, ax_kernel, ax_c in zip(["test_balanced_accuracy","test_corrected_accuracy","test_recall",
                                                   "test_accuracy","test_precision","test_neg_log_loss"],
                                                    axes_scaler.flatten(),axes_kernel.flatten(),axes_c.flatten()):
        df = pd.concat([ get_cv_split_performance(results=_res, pipeline=_pip, performance_metric=metric)  
                         for _pip,_res in final_results.items() if _res is not None], ignore_index=True)  
        # ax_scaler = df.boxplot(by='scaler',ax=ax_scaler)   
        data_ind = df["metric_name"]
        data_ind = data_ind.values == metric
        data = df.iloc[data_ind]
        ax_scaler = sns.boxplot(x="alpha",y="metric",hue="scaler", data=data, palette="muted",ax=ax_scaler)
        ax_scaler = sns.swarmplot(x="alpha", y="metric", hue="scaler", data=data, palette="muted", ax=ax_scaler,size=2)
        ax_scaler.set_title(metric) 
        ax_scaler.get_legend().remove()
        ax_kernel = sns.boxplot(x="alpha",y="metric",hue="l1_ratio",data=data, palette="muted",ax=ax_kernel)
        ax_kernel = sns.swarmplot(x="alpha", y="metric", hue="l1_ratio", data=data, palette="muted",ax=ax_kernel,size=2)
        ax_kernel.set_title(metric)
        ax_kernel.get_legend().remove()
        ax_c = sns.boxplot(x="alpha",y="metric",hue="select",data=data,ax=ax_c)
        ax_c = sns.swarmplot(x="alpha", y="metric", hue="select", data=data, palette="muted",ax=ax_c,size=2)
        ax_c.set_title(metric)
        ax_c.get_legend().remove()
    fig_scaler.legend(labels=np.unique(df["scaler"].values).tolist(), loc = 2, bbox_to_anchor = (1,1))
    fig_scaler.savefig("/auto/home/users/d/r/drimez/Classify"+plotting_folder+"/" + filename + "_scaler.png")
    fig_kernel.legend(labels=np.unique(df["kernel"].values).tolist(), loc = 2, bbox_to_anchor = (1,1))
    fig_kernel.savefig("/auto/home/users/d/r/drimez/Classify"+plotting_folder+"/" + filename + "_kernel.png") 
    fig_c.legend(labels=np.unique(df["select"].values).tolist(), loc = 2, bbox_to_anchor = (1,1))
    fig_c.savefig("/auto/home/users/d/r/drimez/Classify"+plotting_folder+"/" + filename + "_select.png")
     
def plot_results_sgl(final_results,filename,ensemble_or_not): 

    plotting_folder = "" if ensemble_or_not is None else "_bagging"
    
    def get_cv_split_performance(results, pipeline, performance_metric="test_accuracy"):
        metrics = { "metric": results[performance_metric],
                    "metric_name": performance_metric, 
                    "model": pipeline, 
                    "scaler":pipeline.split(",")[0].split(":")[-1],
                    "eps":pipeline.split(",")[-2].split(":")[-1] , 
                    "select":pipeline.split(",")[-1].split(":")[-1].split("}")[0]   }
        return pd.DataFrame(metrics)
    
    
    fig_scaler, axes_scaler = plt.subplots(2,2,figsize=(18,9),sharex=True)
    fig_eps, axes_eps = plt.subplots(2,2,figsize=(18,9),sharex=True) 
    for metric, ax_scaler, ax_eps in zip(["train_accuracy", "train_neg_log_loss",
                                                   "test_accuracy", "test_neg_log_loss"],
                                                    axes_scaler.flatten(),axes_eps.flatten()):
        df = pd.concat([ get_cv_split_performance(results=_res, pipeline=_pip, performance_metric=metric)  
                         for _pip,_res in final_results.items() if _res is not None])  
        # ax_scaler = df.boxplot(by='scaler',ax=ax_scaler)  
        data_ind = df["metric_name"]
        data_ind = data_ind.values == metric
        data = df.iloc[data_ind]
        ax_scaler = sns.boxplot(x="eps",y="metric",hue="scaler",data=data, palette="muted",ax=ax_scaler)
        ax_scaler = sns.swarmplot(x="eps", y="metric", hue="scaler", data=data, color=".25",ax=ax_scaler,size=2)
        ax_scaler.set_title(metric)
        ax_scaler.get_legend().remove()
        ax_eps = sns.boxplot(x='eps',y="metric",hue="select",data=data,ax=ax_eps)
        ax_eps = sns.swarmplot(x="eps", y="metric", hue="select", data=data, color=".25",ax=ax_eps,size=2)
        ax_eps.set_title(metric) 
        ax_eps.get_legend().remove()
    fig_scaler.legend(labels=np.unique(df["scaler"].values).tolist(), loc = 2, bbox_to_anchor = (1,1))
    fig_scaler.savefig("/auto/home/users/d/r/drimez/Classify"+plotting_folder+"/" + filename + "_scaler.png")
    fig_eps.legend(labels=np.unique(df["select"].values).tolist(), loc = 2, bbox_to_anchor = (1,1))
    fig_eps.savefig("/auto/home/users/d/r/drimez/Classify"+plotting_folder+"/" + filename + "_select.png") 
    
def plot_results_forest(final_results,filename,ensemble_or_not): 

    plotting_folder = "" if ensemble_or_not is None else "_bagging"
    
    def get_cv_split_performance(results, pipeline, performance_metric="test_accuracy"):
        metrics = { "metric": results[performance_metric],
                    "metric_name": performance_metric, 
                    "model": pipeline, 
                    "scaler":pipeline.split(",")[0].split(":")[-1],
                    "n_estimators":pipeline.split(",")[-1].split(":")[-1].split("}")[0], 
                    "select":pipeline.split(",")[-1].split(":")[-1].split("}")[0]}   
        return pd.DataFrame(metrics)
    
    # final_results = {}
    # for pckl_path in os.scandir(final_results_path):
    #     if pckl_path.path.split("_")[0]==final_results_path:
    #         final_results = dict(final_results,**decompress_pickle()) 
    
    fig_scaler, axes_scaler = plt.subplots(2,2,figsize=(18,12),sharex=True)
    fig_n_estimators, axes_n_estimators = plt.subplots(2,2,figsize=(18,12),sharex=True) 
    for metric, ax_scaler, ax_n_estimators in zip(["train_accuracy", "train_neg_log_loss",
                                                   "test_accuracy", "test_neg_log_loss"],
                                                    axes_scaler.flatten(),axes_n_estimators.flatten()):
        df = pd.concat([ get_cv_split_performance(results=_res, pipeline=_pip, performance_metric=metric)  
                         for _pip,_res in final_results.items() if _res is not None])  
        # ax_scaler = df.boxplot(by='scaler',ax=ax_scaler)  
        data_ind = df["metric_name"]
        data_ind = data_ind.values == metric
        data = df.iloc[data_ind]
        ax_scaler = sns.boxplot(x="n_estimators",y="metric",hue="scaler",data=data, palette="muted",ax=ax_scaler)
        ax_scaler = sns.swarmplot(x="n_estimators", y="metric", hue="scaler", data=data, palette="muted",ax=ax_scaler,size=2)
        ax_scaler.set_title(metric)
        ax_scaler.get_legend().remove()
        ax_n_estimators = sns.boxplot(x='n_estimators',y="metric",hue="select",ax=ax_n_estimators,data=data) 
        ax_n_estimators = sns.swarmplot(x="n_estimators", y="metric", hue="select", data=data, palette="muted",ax=ax_n_estimators,size=2)
        ax_n_estimators.set_title(metric) 
        ax_n_estimators.get_legend().remove()
    fig_scaler.legend(labels=np.unique(df["scaler"].values).tolist(), loc = 2, bbox_to_anchor = (1,1))
    fig_scaler.savefig("/auto/home/users/d/r/drimez/Classify"+plotting_folder+"/" + filename + "_scaler.png")
    fig_n_estimators.legend(labels=np.unique(df["select"].values).tolist(), loc = 2, bbox_to_anchor = (1,1))
    fig_n_estimators.savefig("/auto/home/users/d/r/drimez/Classify"+plotting_folder+"/" + filename + "_select.png") 

def plot_results_Logistic(final_results,filename,ensemble_or_not): 
  
    if isinstance(final_results,str):
        final_results_path = final_results
        final_results = {} ; final_results_ = None
        with open(final_results_path,"r") as reader:
            final_results_ = reader.readlines()
            reader.close()
        perfs_name = [eeeee.replace("\n", "") for eeeee in final_results_[0].split(",")[1:]] 
        for line in final_results_[1:]:
            params, perfs = line.split("},")
            try:
                results_ = {metric_name:float(a_metric) for metric_name, a_metric in zip(perfs_name,perfs.split(","))}
                if (not np.any([res for _,res in results_.items()])==0) \
                    and results_["test_balanced_accuracy"]>=0.35 and \
                   (not results_["train_accuracy"]-results_["test_accuracy"]>=0.2):
                    final_results[params+"}"] = results_
            except Exception:
                pass

    plotting_folder_ = "" if ensemble_or_not is None else "_bagging"
    plotting_folder = "/auto/home/users/d/r/drimez/Classify" + plotting_folder_ + "/"
    plotting_folder = "Classify" + plotting_folder_ + "/"
    
    def get_cv_split_performance(results, pipeline, performance_metric="test_accuracy"):
        args = {keyval.split(":")[0].replace("'", ""):keyval.split(":")[1] for keyval in pipeline[1:-1].split(", ")}
        metrics = dict( **{ "metric": results[performance_metric],
                            "metric_name": performance_metric, 
                            "model": pipeline}, **args )
        return pd.DataFrame([metrics]) #, [*args.keys()]
     
    legend_s = None ; legend_k = None ; legend_c = None  
    fig_scaler, axes_scaler = plt.subplots(2,2,figsize=(18,12),sharex=True)
    fig_n_estimators, axes_n_estimators = plt.subplots(2,2,figsize=(18,12),sharex=True) 
    for metric, ax_scaler, ax_n_estimators in zip(["train_accuracy","test_accuracy",
                                                   "test_corrected_accuracy","test_balanced_accuracy"],
                                                    axes_scaler.flatten(),axes_n_estimators.flatten()):
        df = pd.concat([ get_cv_split_performance(results=_res, pipeline=_pip, performance_metric=metric)  
                         for _pip,_res in final_results.items() if _res is not None])  
        # ax_scaler = df.boxplot(by='scaler',ax=ax_scaler)  
        data_ind = df["metric_name"]
        data_ind = data_ind.values == metric
        data = df.iloc[data_ind]
        ax_scaler = sns.boxplot(x="C",y="metric",hue="scaler",data=data, palette="muted",ax=ax_scaler)
        legend_s = copy.copy(ax_scaler.get_legend())._set_loc(2)
        ax_scaler = sns.swarmplot(x="C", y="metric", hue="scaler", data=data, palette="muted",ax=ax_scaler,size=4)
        ax_scaler.set_title(metric)
        ax_scaler.get_legend().remove() 
        ax_n_estimators = sns.boxplot(x='C',y="metric",hue="select",ax=ax_n_estimators,data=data) 
        legend_k = copy.copy(ax_n_estimators.get_legend())._set_loc(2)
        ax_n_estimators = sns.swarmplot(x="C", y="metric", hue="select", data=data, palette="muted",ax=ax_n_estimators,size=4)
        ax_n_estimators.set_title(metric)
        ax_n_estimators.get_legend().remove()
        ax_n_estimators.set_title(metric)   
    # fig_scaler.legend(labels=np.unique(df["scaler"].values).tolist(), loc = 2)
    # fig_scaler.add_artist(legend_s) 
    fig_scaler.savefig(plotting_folder + filename + "_scaler.png")
    # fig_kernel.legend(labels=np.unique(df["kernel"].values).tolist(), loc = 2)
    # fig_kernel.add_artist(legend_k).get_legend()._set_loc(2)
    fig_n_estimators.savefig(plotting_folder + filename + "_select.png")  
      
"""     
def generate_features(X,X_11,X_1,subjects=pd.read_csv("rename.txt",header=None).values.T[1]): 
    
    X1 = pd.DataFrame() ; Xpca = pd.DataFrame()
    X1_columns = [] ; groups = [] ; group_names = [] ; groupspca = [] ; group_namespca = [] 
    done = False ; X11_columns = [] ; groups_idx = [] ; groups_idx_pca = [] ; classes = {}
    for inid, nid in enumerate(subjects):  
         
        if nid.split('_')[0]!='U':
            if nid.split('_')[0] in [*classes.keys()]:   
                classes[nid.split('_')[0]].append(inid) 
            else:   
                classes[nid.split('_')[0]] = [inid]
                
            bundles = np.unique(X['tractID'].values[X.index.values[X['subjectID'].values==nid]])
            X1_columns = [] ; X11_columns = []
            for ilign, lign in enumerate(X.index.values[X['subjectID'].values==nid]):
                t = X['tractID'].values[lign]
                nni = X['nodeID'].values.flatten()[lign] 
                groups.append([]) ; groupspca.append([])
                for n_col, col in enumerate(X_1.columns.values.flatten()):
                    X1_columns.append(str((t,nni,col)))  
                    if not tuple((t,nni,col)) in group_names:
                        group_names.append(tuple((t,nni,col)))
                        groups[ilign].append(n_col)
                for n_col, col in enumerate(X_11.columns.values.flatten()):
                    X11_columns.append(str((t,nni,col)))  
                    if not tuple((t,nni,col)) in group_namespca:
                        group_namespca.append(tuple((t,nni,col)))
                        groupspca[ilign].append(n_col)
                           
            X1_columns = np.array(X1_columns).flatten().tolist()    
             
            X1 = pd.concat([X1,pd.DataFrame(np.array(X_1.values[X['subjectID'].values==nid]).reshape((1,len(X1_columns))),
                            columns=X1_columns)], ignore_index=True) 
            Xpca = pd.concat([Xpca,pd.DataFrame(np.array(X_11.values[X['subjectID'].values==nid]).reshape((1,len(X11_columns))),
                            columns=X11_columns)], ignore_index=True) 
                 
    X1 = X1.fillna(0) 
    toremove = [] 
    Y2 = [0 for a_class in subjects if a_class.split('_')[0]!='U']
    for ia_class, a_class in enumerate([*classes.keys()]): 
        for an_istance in np.array(classes[a_class]):
            Y2[int(an_istance)] = ia_class
    for classes_ in np.unique(Y2):
        toremove.append((X1.values[Y2==classes_] != 0).sum(axis=0)<=len(X1.values[Y2==classes_])/2) 
    toremove = np.array(toremove).T
    to_remove = []
    for itr, tr in [_ for _ in enumerate(toremove)][::-1]:
        if np.any(tr): 
            if np.sum(tr)==1:
                print(str(group_names.pop(itr))+"%s is missing in class %s"%(100*(len(X1.values[Y2==np.unique(Y2)[tr]])-(X1.values[Y2==np.unique(Y2)[tr]] != 0).sum(axis=0)[itr])/len(X1.values[Y2==np.unique(Y2)[tr]]),
                                                                                np.array([*classes.keys()])[tr][0]))
            elif np.sum(tr)==len(tr):
                print(str(group_names.pop(itr))+" Empty feature")
            else:
                print(str(group_names.pop(itr))+" More than 50% missing for several classes")
            groups.pop(itr)
            to_remove.append(X1.columns.values[itr])    
    X1 = X1.drop(to_remove,axis=1)    
    Xpca = Xpca.fillna(0)   
    toremove = []
    for classes_ in np.unique(Y2):
        toremove.append((Xpca.values[Y2==classes_] != 0).sum(axis=0)<=len(Xpca.values[Y2==classes_])/2)
    toremove = np.array(toremove).T
    to_remove = []  
    for itr, tr in [_ for _ in enumerate(toremove)][::-1]:  
        if np.any(tr): 
            # if np.sum(tr)==1:
            #     print(str(group_namespca.pop(itr))+"%s is missing in class %s"%(100*(len(Xpca.values[Y2==np.unique(Y2)[tr]])-(Xpca.values[Y2==np.unique(Y2)[tr]] != 0).sum(axis=0)[itr])/len(Xpca.values[Y2==np.unique(Y2)[tr]]),
            #                                                                     np.array([*classes.keys()])[tr][0]))
            # elif np.sum(tr)==len(tr):
            #     print(str(group_namespca.pop(itr))+" Empty feature")
            # else:
            #     print(str(group_namespca.pop(itr))+" More than 50% missing for several classes") 
            groupspca.pop(itr)
            to_remove.append(Xpca.columns.values[itr])  
    Xpca = Xpca.drop(to_remove,axis=1)  
            
    # for inid, nid in enumerate(['H_0','H_1','H_2','H_3','H_4','V_100','V_101','V_102','V_103']):        
    groups_idx.append([[] for _ in X1.columns])
    groups_idx_pca.append([[] for _ in Xpca.columns] )
    for nn_col, xt in enumerate(X1):  
        lmh = np.arange(len(bundles))[bundles==xt.split(',')[0][2:-1]] 
        groups_idx[0][lmh[0]].append(nn_col)  
                           
    for nn_col, xt in enumerate(Xpca):
        lmh = np.arange(len(bundles))[bundles==xt.split(',')[0][2:-1]] 
        groups_idx_pca[0][lmh[0]].append(nn_col)     
        
    n_samples, n_feats = X1.shape 
    feats_name = list(X1.columns)  
    Y1 = np.arange(len(X1.values))
    for a_class in [*classes.keys()]:
        for ia_class in classes[a_class]:
            if a_class=="H":
                Y1[ia_class] = 0
            elif a_class=="C":
                Y1[ia_class] = 1
            else:
                Y1[ia_class] = 2
    
    return X1, Xpca, Y1, groups, group_names, groupspca, group_namespca, groups_idx[0], groups_idx_pca[0]  
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def my_group_pca(XX,y_true,max_components=4,n_groups=180,p=patient_list,mask=None,pca_type=None):
         
    X = copy.copy(XX)
    n_patients=len(X.values)
    grp_names = np.unique([_.split(",")[0][2:-1].split(".")[0] for _ in X.columns.values])
    side_ = np.array([_.split(",")[0][2:-1].split(".")[1] if len(_.split(",")[0][2:-1].split("."))==2 else "" for _ in X.columns.values])
    groups_names = [_.split(",")[0][2:-1].split(".")[0] for _ in X.columns.values]   
    if (not (pca_type.replace("scale_","") in ("bilat_pca","pca_without_age"))) or len(pca_type.split("unilat"))==2:
        grp_names = np.unique([_.split(",")[0][2:-1] for _ in X.columns.values])
        groups_names = [_.split(",")[0][2:-1] for _ in X.columns.values]  
    n_groups = len(grp_names)  
    
    if "scale" in pca_type.split("_"): 
        X = pd.DataFrame(StandardScaler().fit_transform(X),columns=X.columns.values)
    
    pca_models_ = [] 
    components_ = [None for grp in range(n_groups)]
    explained_variance_ = [None for grp in range(n_groups)]
    explained_variance_ratio_ = [None for grp in range(n_groups)]
    singular_values_ = [None for grp in range(n_groups)]
    mean_ = [None for grp in range(n_groups)]
    n_components_ = [None for grp in range(n_groups)]
    noise_variance_ = [None for grp in range(n_groups)]
    features_out = [[] for pp in range(n_patients)]
    groups_out_ = []
    """ 
    bins = len(y_true)/np.unique(y_true,return_counts=True)[1] ; weights = []
    for nnnn, nnn_class in enumerate(np.unique(y_true)):
        weights = np.append(weights,[bins[nnnn]/len(bins) for _ in y_true if _==nnn_class])
    weights = weights.reshape((70,1))
    weights = np.ones((70,1))
    """
    ad = [True if str(_).split(",")[-1][2:-2]=="AD" else False for _ in X.columns.values]
    md = [True if str(_).split(",")[-1][2:-2]=="MD" else False for _ in X.columns.values]
    rd = [True if str(_).split(",")[-1][2:-2]=="RD" else False for _ in X.columns.values]
    fa = [True if str(_).split(",")[-1][2:-2]=="FA" else False for _ in X.columns.values]
    
    feature_start_idx = 0 ; idx = -1
    # if not (mask is None):
    #     grp_names = np.unique([_.split(",")[1] for _ in np.array(X.columns.values)[mask==True]])
    #     groups_names = [_.split(",")[1] for _ in np.array(X.columns.values)[mask==True]]
    for idx_ in np.arange(n_groups):
        
        grp = np.zeros((len(groups_names)))  
        for ig, l in enumerate(groups_names):  
            if l==grp_names[idx_]:  
                grp[ig] = 1  
            else:
                grp[ig] = 0  
         
        # grp = np.tile(grp==1,(9,1)) 
        grp = (grp==1).flatten() 
        grp_fa = np.logical_and(grp,fa)
        grp_md = np.logical_and(grp,md)
        grp_ad = np.logical_and(grp,ad)
        grp_rd = np.logical_and(grp,rd)
        if pca_type.replace("scale_","") in ("pca_without_age","bilat_pca") and not np.any(np.logical_and(side_=="",grp)):
            
            grp_fa_right = np.logical_and(grp_fa,side_=="right")
            grp_md_right = np.logical_and(grp_md,side_=="right")
            grp_ad_right = np.logical_and(grp_ad,side_=="right")
            grp_rd_right = np.logical_and(grp_rd,side_=="right")   
            grp_fa_left = np.logical_and(grp_fa,side_=="left")
            grp_md_left = np.logical_and(grp_md,side_=="left")
            grp_ad_left = np.logical_and(grp_ad,side_=="left")
            grp_rd_left = np.logical_and(grp_rd,side_=="left")   
        
        if np.sum(grp)>0:
            idx += 1
            
            group_x = None
            if (pca_type in ("bilat_pca","pca_without_age")) and not np.any(np.logical_and(side_=="",grp)):
                group_x = np.concatenate((np.array(np.concatenate([list(np.array([X.values[iii][grp_] for iii in range(n_patients)]).flatten()) for grp_ in [grp_ad_left,grp_ad_right]],axis=0)).reshape((69*np.sum(grp_ad),1)),
                                          np.array(np.concatenate([list(np.array([X.values[iii][grp_] for iii in range(n_patients)]).flatten()) for grp_ in [grp_md_left,grp_md_right]],axis=0)).reshape((69*np.sum(grp_md),1)),
                                          np.array(np.concatenate([list(np.array([X.values[iii][grp_] for iii in range(n_patients)]).flatten()) for grp_ in [grp_rd_left,grp_rd_right]],axis=0)).reshape((69*np.sum(grp_rd),1)),
                                          np.array(np.concatenate([list(np.array([X.values[iii][grp_] for iii in range(n_patients)]).flatten()) for grp_ in [grp_fa_left,grp_fa_right]],axis=0)).reshape((69*np.sum(grp_fa),1))),axis=1)
            else:
                group_x = np.concatenate((np.reshape([X.values[iii][grp_ad] for iii in range(n_patients)],(-1,1)),
                                          np.reshape([X.values[iii][grp_md] for iii in range(n_patients)],(-1,1)),
                                          np.reshape([X.values[iii][grp_rd] for iii in range(n_patients)],(-1,1)),
                                          np.reshape([X.values[iii][grp_fa] for iii in range(n_patients)],(-1,1))),axis=1)
                                       
            group_x = np.array(group_x) 
            pca_models_.append( PCA( n_components=min(max_components,len(group_x[0])),
                                        copy=True, whiten=False ) ) 
            group_x_std = None 
            group_x_std = StandardScaler().fit_transform(group_x)  # _transform
            pca_models_[idx].fit(group_x_std)
            old_components = pca_models_[idx].components_
                 
            if pca_type.replace("scale_","") in ("bilat_pca","pca_without_age"):
                this_features_out = pca_models_[idx].transform(group_x)
                pca_comp_sign = np.array([np.all(old_components[0]<0),old_components[1][-1]<0,False,False])
                if np.any(pca_comp_sign):
                    old_components[pca_comp_sign] = -old_components[pca_comp_sign]
                    this_features_out[:,:2] = -this_features_out[:,:2] 
                this_features_out = np.reshape(this_features_out[:,:2],(69,2*len(this_features_out)//69)) # 2 for each side
            else: 
                this_features_out = pca_models_[idx].transform(group_x)
                pca_comp_sign = np.array([np.all(old_components[0]<0),old_components[1][-1]<0,False,False])
                if np.any(pca_comp_sign):
                    old_components[pca_comp_sign] = -old_components[pca_comp_sign]
                    this_features_out[:,:2] = -this_features_out[:,:2]
                this_features_out = np.reshape(this_features_out[:,:2],(69,2*len(this_features_out)//69))
                 
            features_out = np.concatenate( (features_out,
                                            this_features_out), axis=1 )
            components_[idx] = old_components[:2]
            explained_variance_[idx] = pca_models_[idx].explained_variance_[:2]
            explained_variance_ratio_[idx] = pca_models_[idx].explained_variance_ratio_[:2]
            singular_values_[idx] = pca_models_[idx].singular_values_[:2]
            mean_[idx] = pca_models_[idx].mean_[:2]
            n_components_[idx] = 2
            noise_variance_[idx] = pca_models_[idx].noise_variance_ 
            groups_out_.append(
                np.arange(
                    feature_start_idx,
                    feature_start_idx + pca_models_[idx].n_components_,
                )
            )
            feature_start_idx += pca_models_[idx].n_components_
        else:
            print("Empty group: "+str(grp_names[idx_]))
    
    n_features_out_ = np.sum([len(grp) for grp in groups_out_])
    def generate_feature_names(pca_models,group_names=grp_names,all_features=np.array(groups_names)[fa],pca_type=pca_type,side_=side_[fa]):
        feature_names_out_ = []  
        for idx, (grp, pca_model) in enumerate(zip(groups_out_, pca_models)):
            if group_names is None:
                group_name = "group" + str(idx).zfill(
                    int(np.log10(len(groups_out_)) + 1)
                )
            else:
                group_name = _stringify_sequence(group_names[idx])  
            if not (pca_type.replace("scale_","") in ("bilat_pca","pca_without_age")):
                seg_in_group = [True if group_name==_ else False for _ in all_features] 
                feature_type = "feature" if pca_model is None else "pc"
                for seg_in in range(np.sum(seg_in_group)):
                    feature_names_out_ += [
                        "_".join([group_name, str(seg_in), feature_type + str(n)]) for n in range(2)
                    ]
            else:
                seg_in_group = [True if group_name==_ else False for _ in all_features]  
                for side in np.unique(np.array(side_)[seg_in_group]):
                    if side == "": # callosum tracts 
                        feature_type = "feature" if pca_model is None else "pc"
                        for seg_in in range(np.sum(seg_in_group)):
                            feature_names_out_ += [
                                "_".join([group_name, str(seg_in), feature_type + str(n)]) for n in range(2)
                            ]
                    else:
                        seg_in_group = [True if (group_name==_) and (side_[i_]==side) else False for i_, _ in enumerate(all_features)] 
                        feature_type = "feature" if pca_model is None else "pc"
                        for seg_in in range(np.sum(seg_in_group)):
                            feature_names_out_ += [
                                "_".join([group_name+"_"+str(side), str(seg_in), feature_type + str(n)]) for n in range(2)
                            ]
        return feature_names_out_
        
    pca_features_names = generate_feature_names(pca_models_)   
               
    return components_, pca_features_names , explained_variance_, features_out, explained_variance_ratio_, \
           singular_values_, mean_, n_components_, noise_variance_, pca_models_, groups_out_ 

# import shap
# shap.initjs() 
from alibi.explainers import KernelShap
def class_labels(classifier, instance, class_names=None):
    """
    Creates a set of legend labels based on the decision
    scores of a classifier and, optionally, the class names.
    """

    decision_scores = classifier.decision_function(instance)

    if not class_names:
        class_names = [f'Class {i}' for i in range(decision_scores.shape[1])]

    for i, score in enumerate(np.nditer(decision_scores)):
        class_names[i] = class_names[i] + ' ({})'.format(round(score.item(),3))

    return class_names
 
 
import cv2
import time
import json
import base64
# import requests
"""
def send_image(img):
    #Convert image to sendable format and store in JSON 
    # _, encimg = cv2.imencode(".png ", img)
    # encimg = img.canvas.tostring_rgb()
    # img_str = encimg.tostring()
    # img_byte = base64.b64encode(img_str)#.decode("utf-8")
    # image = open(img, 'rb') #open binary file in read mode 
    # img_byte = base64.encodebytes(image.read())
    # img_json = img_byte.encode('utf-8')
    # image.close() 
    with open(img, mode='rb') as file:
        img = file.read()
    img_json = base64.encodebytes(img).decode('utf-8')
    return img_json
    
from flask import Flask, request, Response
app = Flask(__name__)

def save_image():
    #Data conversion process
    data = request.data.decode('utf-8')
    data_json = json.loads(data)
    image = data_json['image']
    image_dec = base64.b64decode(image)
    data_np = np.fromstring(image_dec, dtype='uint8')
    decimg = cv2.imdecode(data_np, 1)
"""
def plot_to_notebook(img,img_path,section_name,train_results): 
    
    print("Plotting to notebook")
    means = str({"\nmean_"+key:(np.nanmean(val),np.nanstd(val)) for key, val in train_results.items()})
    
    empty_notebook = { "cells": [],
                       "metadata": {},
                       "nbformat": 4,
                       "nbformat_minor": 5
                      }  
    img_path = img_path.split(".")[0] + "_" + str("_".join([str(vvv) if len(str(vvv).split())<=1 else str(vvv).split()[1]
                                                            for vvv in section_name.values()])) + ".ipynb"
    if not os.path.exists("/".join(img_path.split("/")[:-1])):
        os.makedirs("/".join(img_path.split("/")[:-1]))
        
    if os.path.exists(img_path):
        with open(img_path,"r") as _reader:    
            empty_notebook = json.loads(_reader.read())
            _reader.close() 
            
    temp_img = img_path.split(".")[0]+"_temp.png"  
    img.savefig(temp_img)      
             
    empty_notebook["cells"].append({"cell_type":"code","metadata":{},"source":["# %s\n"%str(section_name) + 
                                                                               str(means) + "\n"],
                                    "execution_count":1,"outputs":[{"data": {"image/png":send_image(temp_img)},
                                                                    "metadata": {"needs_background": "light"},
                                                                    "output_type": "display_data"}]})
    os.remove(temp_img)
              
    with open(img_path,"w") as _writer:    
        json.dump(empty_notebook,_writer)
        _writer.close()
         
def plot_coef(coeff,true_coeff,mp,pv,feat_names,coeff_path,model_name,params,train_results,refresh=True): 

    if False and (not os.path.exists(coeff_path) or refresh):  
        print("Plotting feature importances")
        if coeff is None:
            coeff = true_coeff
        if len(coeff.shape)>len(true_coeff.shape):
            coeff_ = [coeff[:,_,:] for _ in range(len(coeff[0]))] 
            dfs =  []
            most_ = np.argsort(abs(coeff.mean(axis=(0,1))))[::-1] 
            for ic_, coeff in enumerate(coeff_):  
                dfs.append(pd.DataFrame(coeff.T[most_].T.reshape((coeff.size,1)),columns=["coeff"]))
                dfs[-1]["feature"] = np.array([feat_names[most_] for _ in range(len(coeff))]).flatten()
                dfs[-1]["class"] = ic_
            coeff = pd.concat(dfs,ignore_index=True)      
            fig, ax = plt.subplots(1,1,figsize=(24,4*len(coeff_[0][0])//20))   
            ax = sns.barplot(x="coeff",y="feature",hue="class",data=coeff,ax=ax,orient="h")
            # ax = coef_to_plot.plot(kind = "barh",ax=ax, fontsize=8)
            try:
                ax.get_legend().remove()
            except Exception:
                pass
            ax.boxplot(true_coeff[:,most_],vert=False)
            ax.set_yticks(np.arange(len(feat_names[most_])))
            ax.set_yticklabels(feat_names[most_]) 
            ax.set_title("Feature importance using %s Model"%model_name)
            plot_to_notebook(fig,coeff_path,params,train_results)
        else:
            most_ = np.argsort(abs(coeff).mean(0))[::-1] 
            to_plot = most_  
              
            fig, ax = plt.subplots(1,1,figsize=(24,4*len(coeff[0][0])//20))  
            # coef_to_plot = pd.DataFrame(coeff[this_plot],columns=feat_names[this_plot]) 
            ax = sns.barplot(data=coeff[:,this_plot],ax=ax)
            # ax = coef_to_plot.plot(kind = "barh",ax=ax, fontsize=8)
            try:
                ax.get_legend().remove()
            except Exception:
                pass
            ax.boxplot(true_coeff[:,this_plot],vert=False)
            ax.set_yticks(np.arange(len(feat_names[this_plot])))
            ax.set_yticklabels(feat_names[this_plot])
            # if iax%2!=0:
            #     ax.yaxis.tick_right()
            ax.set_title("Feature importance using %s Model"%model_name) 
            plot_to_notebook(fig,coeff_path,params,train_results)
    else: 
        data = np.array([np.nan_to_num(np.nanmean(true_coeff,axis=0),posinf=0, neginf=0),
                         np.nan_to_num(np.nanstd(true_coeff,axis=0),posinf=0, neginf=0)])
        
        data = np.nan_to_num(data,posinf=0, neginf=0)  # /data[0].sum()
        columns = ["true_mean","true_std"]
        if not (coeff is None):
            coeff_ = np.nan_to_num(np.nanmean(np.array([coeff[:,_,:] for _ in range(len(coeff[0]))]),axis=1),posinf=0, neginf=0)
            data = np.concatenate((coeff_,data),axis=0).T
            columns = np.append(["coeff_%s"%iop for iop in range(len(coeff_))],columns)
        else:
            coeff_ = np.nan_to_num(np.nanmean(np.array([coeff[:,_,:] for _ in range(len(coeff[0]))]),axis=1),posinf=0, neginf=0)
            data = np.concatenate((coeff_,data),axis=0).T
            columns = np.append(["coeff_%s"%iop for iop in range(len(coeff_))],columns)
        df = pd.DataFrame(data=data,columns=columns,index=feat_names).sort_values(by="true_mean",ascending=False)
        # if os.path.exists(coeff_path):
        #     old_df = pd.read_csv(coeff_path)
        #     df = pd.concat((df,old_df),axis=0,ignore_index=True)
        df["mean_perf"] = np.nan_to_num(np.nanmean(mp))
        df["mean_pval"] = np.nan_to_num(np.nanmean(pv))
        if not os.path.exists("/".join(coeff_path.split("/")[:-1])):
            os.makedirs("/".join(coeff_path.split("/")[:-1]))
        df.to_csv(coeff_path)
            
 
import shutup
def get_importance(ensemble_estimator,pipeline,X,y,train,test,attr__=None,feature_names_in_=None,cv=None,shap=False,figname=None):
    
    print("Computing feature importances")
    warnings.filterwarnings("ignore")
    shutup.please()
    def my_getattr(estimator__,attr__=attr__,feature_names_in_=feature_names_in_): 
        # print("============================="+str(estimator__.__class__.__name__)+"====================================")
        if estimator__.__class__.__name__ in ("RandomForestClassifier","My_RandomForestClassifier") \
           and (attr__ is None) and not (estimator__.__class__.__name__ == "LogisticRegressionCV"):
           
           all_importances = get_importance(estimator__,attr__=attr__,feature_names_in_=feature_names_in_) 
           return [{feat_name__:feat_importance for feat_name__,feat_importance in zip(feature_names_in_[an_imp!=0],an_imp[an_imp!=0])}
                   for an_imp in all_importances]
        elif estimator__.__class__.__name__ in ("BaggingClassifier","My_BaggingClassifier") \
           and (attr__ is None) and not (estimator__.__class__.__name__ == "LogisticRegressionCV"):
           
           all_importances = get_importance(estimator__,attr__=attr__,feature_names_in_=feature_names_in_) 
           return [{feat_name__:feat_importance for feat_name__,feat_importance in zip(feature_names_in_[an_imp!=0],an_imp[an_imp!=0])}
                   for an_imp in all_importances]
        elif estimator__.__class__.__name__ == "DecisionTreeClassifier" \
           and (attr__ is None) and not (estimator__.__class__.__name__ == "LogisticRegressionCV"):
           
            all_importances = estimator__.feature_importances_
            feature_names_in__ = feature_names_in_ 
            return {feat_name__:feat_importance for feat_name__,feat_importance in zip(feature_names_in__,all_importances)}  
        elif attr__ is None: 
            return [{feat_name__:feat_importance for feat_name__,feat_importance in zip(feature_names_in_,a_coeff)}
                    for a_coeff in estimator__.coef_]
        elif attr__ == "l1_ratio_":
            return {feat_name__:feat_importance for feat_name__,feat_importance in zip(feature_names_in_,estimator__.l1_ratio_)}  
        else:
            print("I don't know this attribute :(  : " + str(attr__))
            return False
            
    # print(ensemble_estimator.__class__.__name__)
    if ensemble_estimator.__class__.__name__ in ("RandomForestClassifier","My_RandomForestClassifier"): 
        all_importances = ensemble_estimator.feature_importances_
    elif ensemble_estimator.__class__.__name__ in ("BaggingClassifier","My_BaggingClassifier"):
        all_importances = Parallel(n_jobs=ensemble_estimator.n_jobs)(
                                    delayed(my_getattr)(tree,attr__=attr__,feature_names_in_=feature_names_in_)
                                    for tree in ensemble_estimator.estimators_ )
    else:
        all_importances = my_getattr(ensemble_estimator,attr__=attr__,feature_names_in_=feature_names_in_)
     
    if not np.any(all_importances!=0):
        return np.zeros(ensemble_estimator.n_features_in_, dtype=np.float64)

    all_features = ensemble_estimator.feature_names_in_ if feature_names_in_ is None else feature_names_in_
    if not isinstance(all_importances[0],dict):
        all_importances_ = []
        for an_importance__ in all_importances:
            this_tree_importances_ = []
            for an_importance in an_importance__:
                mean_importances = {feat_name__:[] for feat_name__ in all_features} 
                for feat__name, feat_importance in an_importance.items():
                    mean_importances[feat__name].append(feat_importance)
                    
                key_array_ = mean_importances.keys()
                for feat_name__ in key_array_:
                    mean_importances[feat_name__] = np.mean(mean_importances[feat_name__])
                
                this_tree_importances_.append(np.array([mean_importance for _,mean_importance in mean_importances.items()],
                                                dtype=np.float64))
                this_tree_importances_[-1] /= this_tree_importances_[-1].sum() 
            all_importances_.append(this_tree_importances_)
        all_importances_ = np.mean(all_importances_,axis=0) 
    else: 
        all_importances_ = []
        for an_importance in all_importances:
            mean_importances = {feat_name__:[] for feat_name__ in all_features} 
            for feat__name, feat_importance in an_importance.items():
                mean_importances[feat__name].append(feat_importance)
                
            key_array_ = mean_importances.keys()
            for feat_name__ in key_array_:
                mean_importances[feat_name__] = np.mean(mean_importances[feat_name__])
            
            all_importances_.append(np.array([mean_importance for _,mean_importance in mean_importances.items()],
                                            dtype=np.float64))
            all_importances_[-1] /= all_importances_[-1].sum() 
    
    y = y[test]
    bins = len(y)/np.unique(y,return_counts=True)[1] ; weights = []
    for nnnn, nnn_class in enumerate(np.unique(y)):
        weights = np.append(weights,[bins[nnnn]/len(bins) for _ in y if _==nnn_class])
         
    true_importances = permutation_importance( pipeline, X[test], y, scoring="balanced_accuracy",
                                               n_jobs=-1, n_repeats=10) #
    true_importances = true_importances.importances.T
      
    return all_importances_, true_importances  
  
def generate_new_groups(old_groups__, select__): 
    feature_counter = 0 ; 
    selected_groups__ = [ [] for gggroup__ in old_groups__ ]  
    for n_feature_, a_feature_ in enumerate(select__): 
        for na_group_, a_group_ in enumerate(old_groups__):
            if a_feature_ in a_group_:  
                selected_groups__[na_group_].append(n_feature_)
                feature_counter += 1      
                
    return [np.array(__) for __ in selected_groups__]
  
  
def compare_models(target="vertigo"): 
    models_df = {"base":{}, "bag":{}, "gpca":{}, "bag_gpca":{}} ; features = None
    with os.scandir("/auto/home/users/d/r/drimez/") as folder_iterator:
        for folder_entry in folder_iterator:
            if "Classify" in folder_entry.name.split("_"):
                with os.scandir(folder_entry.path) as fo_iterator:
                    for fo_entry in fo_iterator:
                        if target in fo_entry.name.split("_"):
                            with os.scandir(fo_entry.path) as iterator:
                                for entry in iterator:
                                    if "results" in entry.name.split("_"):
                                        with os.scandir(fo_entry.path) as subiterator:
                                            for entry in subiterator:
                                                feature = pd.read_csv(entry.path)
                                                if "Unnamed: 0" in feature.columns.values.flatten():
                                                    feature.index = feature["Unnamed: 0"]
                                                feature = feature.drop("Unnamed: 0", axis=1)
                                                if features is None:
                                                    features = {"importances":np.array([feature["true_mean"].values]), 
                                                                "pval":feature["mean_pval"].values[0], "name":['.'.join(fo_entry.name.split('.')[:-1])]}
                                                else:
                                                    features["importances"] = np.concatenate((features["importances"],np.array([feature["true_mean"].values])),axis=0)
                                                    features["pval"] = np.append(features["pval"],feature["mean_pval"].values[0])
                                                    
                            with os.scandir(fo_entry.path) as iterator:
                                for entry in iterator:
                                    if entry.name.split('.')[-1] == "txt" and ("reliable" in entry.name.split("_")):  
                                        this_model = '.'.join(fo_entry.name.split('.')[:-1]).split("_")[1:] 
                                        df = pd.read_csv(entry.path) 
                                        params = df["fitted params"].values
                                        params_string = ""
                                        for par_name, params_val in params:
                                            params_string += str(par_name) + "=" + str(params_val) + "_"
                                        if ("gpca" in this_model) and ("bagging" in this_model):
                                            models_df["bag_gpca"]["_".join(this_model)] = df
                                        elif "bagging" in this_model:
                                            models_df["bag"]["_".join(this_model)] = df
                                        elif "gpca" in this_model:
                                            models_df["bag"]["_".join(this_model)] = df
                                        else:
                                            models_df["base"]["_".join(this_model)] = df 
                   
    def get_cv_split_performance(results, pipeline,ttype=None, bg=None):
        args = {keyval.split(":")[0].replace("'", ""):keyval.split(":")[1] for keyval in pipeline[1:-1].split(", ")}
        metrics = dict( **{ "metric": results[performance_metric],
                            "metric_name": performance_metric, 
                            "model": pipeline}, **args )
        return pd.DataFrame([metrics])  
    
    def concat_models(model_dict,bg=None):
        all_df = pd.DataFrame()
        for model, final_results in model_dict.items():
            df = pd.concat([ get_cv_split_performance(results=_res, pipeline=_pip, ttype=model, bg=bg)  
                             for _pip,_res in final_results.items() ])  
            all_df = pd.concat([all_df,df])
        return all_df
    
    df = pd.DataFrame()
    for imod, models in enumerate([*models_df.keys()]):
        df = pd.concat([df,concat_models(models_df[models],bg=models)])  
            
    #models_df = pd.DataFrame.from_dict(models_df)
    fig, axs = plt.subplots(2,2,figsize=(24,16))
    axes = axs.flatten() 
    for metric, ax in zip(["train_accuracy", "train_balanced_accuracy", "test_accuracy", "test_balanced_accuracy"], axes):
        ax = sns.boxplot(x='type',y=metric,hue="bg",ax=ax,data=df)  
        ax.set_title(metric)
        ax.get_legend().remove()
    fig.legend(labels=[*models_df.keys()], loc = 2, bbox_to_anchor = (1,1))
    fig.savefig("/auto/home/users/d/r/drimez/Classify_resumes/comp_"+target+".png")

import shap
import itertools
def shap_explanation(X00,y0,test_cross_val,estimator,figname,fitted_params): 
    test_all_explanations = np.zeros_like(X00) 
    instances_counts = np.zeros_like(y0)  
    for train, test in test_cross_val.split(X00,y0):            
        X_train = X00[train] ; X_test = X00[test]
        ensemble_estimator = copy.deepcopy(estimator)  
        if ensemble_estimator is None:
            ensemble_estimator = copy.copy(base_model).fit(X_train)
        pred_fcn = ensemble_estimator.decision_function 
        explainer = shap.explainers.Permutation(pred_fcn, X_train, # distributed_opts={'n_cpus': 6},
                                                feature_names=feature_names_in_) 
        bins = len(y0[train])/np.unique(y0[train],return_counts=True)[1] ; weights = []
        for nnnn, nnn_class in enumerate(np.unique(y0[train])):
            weights = np.append(weights,[bins[nnnn]/len(bins) for _ in y0[train] if _==nnn_class]) 
        test_explanations = explainer.shap_values(X_test) 
        test_all_explanations[:,test,:] += test_explanations[:]
        instances_counts[test] += 1     
         
    test_all_explanations /= instances_counts
         
    clustering = shap.utils.hclust(X00, y0)
    explanation = shape.Explanation(test_all_explanations, data=X00, feature_names=feature_names_in_, clustering=clustering)
    fig = plt.figure()
    plt.suptitle("All", fontsize=12)
    shap.plots.beeswarm(explanation, max_display=len(X[0]), clustering=clustering, show=False,
                        feature_names=feature_names_in_, legend_location='lower right')
    plot_to_notebook(fig,figname,fitted_params,train_results) 
    
    class_values = list(np.unique(y)) + list([_ for _ in itertools.combinations(np.unique(y),2)])
    for class_val in class_values:
        idx = y0[np.logical_or(y0==class_val[0],y0==class_val[1])] if len(class_val)>=1 else y0[y0==class_val]
        explanation = shap.Explanation(all_explanations[idx], data=X00[idx,...], feature_names=feature_names_in_, clustering=clustering)
        fig = plt.figure() 
        plt.suptitle(str(class_val), fontsize=12)
        shap.plots.beeswarm(explanation, max_display=len(X00[0]), clustering=clustering, show=False,
                            feature_names=feature_names_in_, legend_location='lower right')
        plot_to_notebook(fig,figname,fitted_params,train_results) 

from sklearn.metrics import make_scorer, get_scorer, top_k_accuracy_score
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score
     
def balanced_metric_wrapper(metric):   
    if isinstance(metric,str):
        metric = get_scorer(metric)._score_func 
        def anonym_metric(y_true,y_pred,metric=metric,*kwargs):  
            bins = len(y_true)/np.unique(y_true,return_counts=True)[1] ; weights = []
            for nnnn, nnn_class in enumerate(np.unique(y_true)):
                weights = np.append(weights,[bins[nnnn]/len(bins) for _ in y_true if _==nnn_class])
            return -metric(y_true=np.array(y_true).astype(int),y_pred=y_pred,labels=np.array(y_true).astype(int),
                           sample_weight=weights/np.mean(weights),*kwargs)  
        return make_scorer(anonym_metric,greater_is_better=False,needs_proba=True)
    else:
        def anonym_metric(y_true,y_pred,metric=metric,*kwargs): 
            bins = len(y_true)/np.unique(y_true,return_counts=True)[1] ; weights = []
            for nnnn, nnn_class in enumerate(np.unique(y_true)):
                weights = np.append(weights,[bins[nnnn]/len(bins) for _ in y_true if _==nnn_class])
            return metric(y_true=y_true,y_pred=y_pred,sample_weight=weights/np.mean(weights),*kwargs)
        return make_scorer(anonym_metric)
    
def balanced_top_k_accuracy(y_true,y_pred,*kwargs):
 
    result = top_k_accuracy_score(y_true=y_true,y_score=y_pred,labels=np.unique(y_true),*kwargs)

    bins = len(y_true)/np.unique(y_true,return_counts=True)[1] ; weights = []
    for nnnn, nnn_class in enumerate(np.unique(y_true)):
        weights = np.append(weights,[bins[nnnn]/len(bins) for _ in y_true if _==nnn_class]) 
    return top_k_accuracy_score(y_true=y_true,y_score=y_pred,labels=np.unique(y_true),
                                sample_weight=weights/np.mean(weights),*kwargs) #/weights.sum()
    
def my_top_k_accuracy(y_true,y_pred,k=2,*kwargs): 
       
    bins = len(y_true)/np.unique(y_true,return_counts=True)[1] ; weights = []
    for nnnn, nnn_class in enumerate(np.unique(y_true)):
        weights = np.append(weights,[bins[nnnn]/len(bins) for _ in y_true if _==nnn_class])
        
    y_score = y_pred   
    y_score_n_classes = y_score.shape[1] if y_score.ndim == 2 else 2 
    labels = np.unique(y_true)
    classes = np.unique(labels)
    n_labels = len(labels)
    n_classes = len(classes) 
    sample_weight = weights/weights.mean()
  
    y_true_encoded = np.searchsorted(classes, y_true) 

    hits = hitshits = None
    if n_classes == 2:
        if k == 1:
            threshold = 0.5 if y_score.min() >= 0 and y_score.max() <= 1 else 0
            y_pred = (y_score > threshold).astype(np.int64)
            hits = y_pred == y_true_encoded
        else:
            hits = np.ones_like(y_score, dtype=np.bool_)
            
        return np.average(hits, weights=sample_weight) 
        
    elif n_classes > 2:
        sorted_pred = np.argsort(y_score, axis=1, kind="mergesort")[:, ::-1]
        sorted_prob = []  
        for i_, lign___ in enumerate(sorted_pred): 
            sorted_prob.append([])
            sorted_prob[-1] = y_score[i_,lign___]
        sorted_prob = np.array(sorted_prob)
        # hitshits = []  
        hitshits = np.logical_or( y_true == sorted_pred[:, 0].T.flatten(), 
                                                np.logical_and(np.logical_and(sorted_prob[:, 1].T>=(sorted_prob[:,0].T+sorted_prob[:,2].T)/2, # same as 2-3 >= 1-2
                                                                               sorted_prob[:, 2].T<=0.2
                                                                               ).flatten(),
                                                               y_true== sorted_pred[:, 1].T.flatten())
                                                )
        # for class_y in np.unique(y_true): 
        #     mask = class_y==y_true
        #     hitshits.append(np.mean( first_or_valid_second[mask] ) )

    return np.average(hitshits,weights=sample_weight.flatten()) 
        
def unclassable(y_true,y_pred,k=2,*kwargs): 
         
    y_score = y_pred   
    y_score_n_classes = y_score.shape[1] if y_score.ndim == 2 else 2 
    labels=np.unique(y_true)
    classes = np.unique(labels)
    n_labels = len(labels)
    n_classes = len(classes) 
    sample_weight = np.ones(n_classes)
  
    y_true_encoded = np.searchsorted(classes, y_true) 

    hits = None
    if n_classes == 2:
        if k == 1:
            threshold = 0.5 if y_score.min() >= 0 and y_score.max() <= 1 else 0
            y_pred = (y_score > threshold).astype(np.int64)
            hits = y_pred == y_true_encoded
        else:
            hits = np.ones_like(y_score, dtype=np.bool_)
    elif n_classes > 2:
        sorted_pred = np.argsort(y_score, axis=1, kind="mergesort")[:, ::-1] 
        hits = []
        for class_y in np.unique(y_true):
            hits.append(np.mean(np.logical_and( (class_y == y_true).flatten(),
                                   np.all(y_score.T>=0.3,axis=0) )) )

    return np.nanmean(hits) 
                 
def hesitate(y_true,y_pred,k=2,class__=0,*kwargs): 
         
    bins = len(y_true)/np.unique(y_true,return_counts=True)[1] ; weights = []
    for nnnn, nnn_class in enumerate(np.unique(y_true)):
        weights = np.append(weights,[bins[nnnn]/len(bins) for _ in y_true if _==nnn_class])
        
    classes_ = [0,1,2,0,1]
    weights[y_true==classes_[class__+2]] = 0
    y_score = y_pred   
    y_score_n_classes = y_score.shape[1] if y_score.ndim == 2 else 2 
    labels = np.unique(y_true)
    classes = np.unique(labels)
    n_labels = len(labels)
    n_classes = len(classes) 
    sample_weight = weights/weights[weights!=0].mean()
  
    y_true_encoded = np.searchsorted(classes, y_true) 

    hits = hitshits = None
    if n_classes == 2:
        if k == 1:
            threshold = 0.5 if y_score.min() >= 0 and y_score.max() <= 1 else 0
            y_pred = (y_score > threshold).astype(np.int64)
            hits = y_pred == y_true_encoded
        else:
            hits = np.ones_like(y_score, dtype=np.bool_)
            
        return np.average(hits, weights=sample_weight) 
        
    elif n_classes > 2:
        sorted_pred = np.argsort(y_score, axis=1, kind="mergesort")[:, ::-1]
        sorted_prob = []  
        for i_, lign___ in enumerate(sorted_pred): 
            sorted_prob.append([])
            sorted_prob[-1] = y_score[i_,lign___]
        sorted_prob = np.array(sorted_prob)
        hitshits = np.logical_and(np.logical_or(np.logical_and(np.logical_and(sorted_prob[:, 1].T>=(sorted_prob[:,0].T+sorted_prob[:,2].T)/2, # same as 2-3 >= 1-2
                                                                               sorted_prob[:, 2].T<=0.2
                                                                               ).flatten(),
                                                               y_true == sorted_pred[:, 0].T.flatten()  ),  
                                                np.logical_and(np.logical_and(sorted_prob[:, 1].T>=(sorted_prob[:,0].T+sorted_prob[:,2].T)/2, # same as 2-3 >= 1-2
                                                                               sorted_prob[:, 2].T<=0.2
                                                                               ).flatten(),
                                                               y_true== sorted_pred[:, 1].T.flatten())
                                                              ),
                                  np.logical_or(y_true==classes_[class__], y_true==classes_[class__+1]) )

        return np.average(hitshits,weights=sample_weight.flatten())  
         
    
from sklearn.model_selection import GridSearchCV    
class My_grid: 
    def __init__(self,model,param_grid,cv,n_best=None,scoring=None):
        self.param_grid = param_grid
        self.model = model 
        self.scoring = scoring
        self.cv = cv
        if n_best is None:
            num_ = len(traverse(param_grid))
            self.n_best = min(num_,max(100,num_//2))
        else:
            self.n_best = n_best
        
    def fit(self,X,y):
        grid = GridSearchCV(self.model,self.param_grid,cv=self.cv,scoring=self.scoring,
                            n_jobs=-1,refit=False,error_score=0)
        grid.fit(X,y)   
        efficient = grid.cv_results_['mean_test_accurancy'] 
        efficient = efficient[efficient>=0.4]                  # only use classifiers with better performances than random  
        rank = np.argsort(efficient)                           # convert ranks to indexes   
        results_ = {"params":{}}
        for gkey, gval in grid.cv_results_.items():
            if gkey.split('_')[0]=="param":
                results_["params"]["_".join(gkey.split('_')[1:])] = gval[rank]
            elif gkey.split('_')[0]=="mean":
                results_[gkey] = gval[rank]
            elif gkey.split('_')[0]=="std":
                results_[gkey] = gval[rank]
        return results_            
            
from My_bagging import My_BaggingClassifier
from My_forest import My_RandomForestClassifier 
# from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model._stochastic_gradient import BaseSGDClassifier
from afqinsight import make_afq_classifier_pipeline, cross_validate_checkpoint 
from sklearn.model_selection._validation import _aggregate_score_dicts, _fit_and_score
from scipy.stats import mannwhitneyu
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif as MI_class
 
# compare_models() 
      
### Enable joblib use 
import logging
import random

def reorient(data,type_,path="/auto/home/users/d/r/drimez/orders"):

    patient_list = np.array(['C_0', 'C_1', 'C_11', 'C_12', 'C_2', 'C_3', 'C_5', 'C_6',
                             'C_7', 'C_8', 'C_9', 'H_0', 'H_1', 'H_2', 'H_3', 'H_5', 'H_6',
                             'V_0', 'V_1', 'V_10', 'V_11', 'V_13', 'V_14', 'V_15', 'V_16',
                             'V_17', 'V_18', 'V_19', 'V_2', 'V_20', 'V_21', 'V_22', 'V_23',
                             'V_24', 'V_25', 'V_26', 'V_27', 'V_28', 'V_29', 'V_3', 'V_30',
                             'V_32', 'V_33', 'V_34', 'V_35', 'V_36', 'V_37', 'V_38', 'V_39',
                             'V_4', 'V_40', 'V_41', 'V_42', 'V_43', 'V_44', 'V_45', 'V_46',
                             'V_47', 'V_48', 'V_49', 'V_5', 'V_50', 'V_51', 'V_52', 'V_53',
                             'V_6', 'V_7', 'V_8', 'V_9' ])
    
    path = path + "/" if type_=="prob" else path + "_det/"
    grp_names = np.unique(["_"+_.split(",")[0][2:-1].split(".")[0] for _ in data.columns.values])
    for bundle in grp_names:
       if os.path.exists(path+bundle+".txt"):
       
           orient = pd.read_csv(path+bundle+".txt") 
           data_values = data.values
           
           columns_in = np.array([True if "_"+_.split(",")[0][2:-1].split(".")[0] == bundle else False for _ in data.columns.values])
           order = [int(_.split(",")[1][1:]) for _ in data.columns.values if "_"+_.split(",")[0][2:-1].split(".")[0] == bundle]
           straight = np.argsort(order)
           inv_ = straight[::-1]  
           for orl in orient.values:
               if orl[-1]==-1 and orl[0]!="C_4":    
                   data_values[orl[0]==patient_list,columns_in][straight] = data.values[orl[0]==patient_list,columns_in][inv_]
                   
           data = pd.DataFrame(data_values,columns=data.columns)
    
    return data


def load_tract_data(x_path,pca_path,type_=None,pca_type=None,age=None):

    if not (pca_type.replace("scale_","") in ("bilat_pca","pca_without_age")):
        pca_type = ""

    X1 = pd.read_csv(x_path).drop("Unnamed: 0",axis=1).drop(6,axis=0)
    Xpca = pd.read_csv(pca_path).drop("Unnamed: 0",axis=1).drop(6,axis=0) 
    X1 = reorient(X1.reindex(sorted(X1.columns), axis=1),type_)
    Xpca = reorient(Xpca.reindex(sorted(Xpca.columns), axis=1),type_)
    
    morphom = pd.DataFrame(np.concatenate([Xpca[feature].values.reshape((len(Xpca[feature].values),1)) for feature in Xpca.columns.values if feature.split(",")[-1][2:-2] in ("curvature (mean)", "length", "diameter_2")],axis=1),
                           columns = [feature for feature in Xpca.columns.values if feature.split(",")[-1][2:-2] in ("curvature (mean)", "length", "diameter_2")])
    Xpca = pd.DataFrame(np.concatenate([Xpca[feature].values.reshape((len(Xpca[feature].values),1)) for feature in Xpca.columns.values if not (feature.split(",")[-1][2:-2] in ("curvature (mean)", "length", "diameter_2"))],axis=1),
                           columns = [feature for feature in Xpca.columns.values if not (feature.split(",")[-1][2:-2] in ("curvature (mean)", "length", "diameter_2"))])


    if "age" in pca_type.split("_"): 
     
        if os.path.exists("/auto/home/users/d/r/drimez/data_age_corrected_"+type_+".csv") and False:
            X1 = pd.read_csv("/auto/home/users/d/r/drimez/data_age_corrected_"+type_+".csv").drop("Unnamed: 0",axis=1)
            Xpca = X1[Xpca.columns.values]
            X1 = X1.reindex(sorted(X1.columns), axis=1)
            Xpca = Xpca.reindex(sorted(Xpca.columns), axis=1) 
        else: 
            age = age.reshape(-1, 1) 
            hist, bin_edges = np.histogram(age, bins=10)
            
            weights_idx = [np.arange(len(age))[np.logical_and(age>=bin_edges[edge_i],age<bin_edges[edge_i+1]).flatten()] 
                                    for edge_i in range(len(bin_edges)-1)] 
            weights_idx[-1] = np.append(weights_idx[-1], np.arange(len(age))[(age>=bin_edges[-1]).flatten()])
            weights_idx = np.array(weights_idx)
            
            weights = []
            for age_ in np.arange(len(age)):
                weights = np.append(weights,1/hist.flatten()[np.array([np.sum(weights_idx[ww] == age_) for ww in range(len(weights_idx))]).flatten()==1]) 
           
            weights /= weights.mean() # sets the mean of the weights to 1 
            
            # correct X1 by removing the component correlated with age
            corrected_Xpca = copy.copy(X1.values)
            print("", file=open("/auto/home/users/d/r/drimez/"+type_+"_age_coef.txt","w"))
            for iop in range(len(X1.values[0])):
                gc.collect()
                non_zero = (X1.values[:,iop]!=0).flatten()
                coef_ = np.polyfit(age.flatten()[non_zero], X1.values[:,iop].flatten()[non_zero], 1, w=weights[non_zero], cov=False)[0]
                print(str(X1.columns.values[iop])+": "+str(coef_), file=open("/auto/home/users/d/r/drimez/"+type_+"_age_coef.txt","a"))
                corrected_Xpca[:,iop][non_zero] = (X1.values[:,iop] - (coef_*age).flatten() )[non_zero]
            X1 =  pd.DataFrame(corrected_Xpca, columns=X1.columns.values)  
            X1.to_csv("/auto/home/users/d/r/drimez/data_age_corrected_"+type_+".csv")  
            """
            # correct Xpca by removing the component correlated with age 
            corrected_Xpca = copy.copy(Xpca.values) ; cols = Xpca.columns.values
            for iop in range(len(Xpca.values[0])):
                if not 
                gc.collect()
                non_zero = (Xpca.values[:,iop]!=0).flatten()
                coef_ = np.polyfit(age.flatten()[non_zero], Xpca.values[:,iop].flatten()[non_zero], 1, w=weights[non_zero], cov=False)[0]  
                print(Xpca.columns.values[iop], coef_)
                corrected_Xpca[:,iop][non_zero] = (Xpca.values[:,iop] - (coef_*age).flatten() )[non_zero]
            """
            Xpca =  X1[Xpca.columns.values]
        

    patient_list = [ 'C_0', 'C_1', 'C_11', 'C_12', 'C_2', 'C_3', 'C_5', 'C_6',
                     'C_7', 'C_8', 'C_9', 'H_0', 'H_1', 'H_2', 'H_3', 'H_5', 'H_6',
                     'V_0', 'V_1', 'V_10', 'V_11', 'V_13', 'V_14', 'V_15', 'V_16',
                     'V_17', 'V_18', 'V_19', 'V_2', 'V_20', 'V_21', 'V_22', 'V_23',
                     'V_24', 'V_25', 'V_26', 'V_27', 'V_28', 'V_29', 'V_3', 'V_30',
                     'V_32', 'V_33', 'V_34', 'V_35', 'V_36', 'V_37', 'V_38', 'V_39',
                     'V_4', 'V_40', 'V_41', 'V_42', 'V_43', 'V_44', 'V_45', 'V_46',
                     'V_47', 'V_48', 'V_49', 'V_5', 'V_50', 'V_51', 'V_52', 'V_53',
                     'V_6', 'V_7', 'V_8', 'V_9' ]
    y = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   2, 2, 2, 2 ])
    """           
    with os.scandir("/auto/home/users/d/r/drimez/orders/") as it:
        for entry in it: 
            if entry.name.split(".")[0]=="txt":
                orders = pd.read_csv(entry.path)
                order_patients = orders["Unnamed: 0"].values.flatten()
                orders = orders.drop("Unnamed: 0", axis=1)
                for op in order_patients:
                    if orders["order"].values[order_patients==op]==-1:
                        try:
                            grp = [True if str(_).split(",")[0][2:-2]==entry.name.split(".txt")[0][1:] else False for _ in X1.columns.values.flatten()] 
                            out_group = [False if str(_).split(",")[0][2:-2]==entry.name.split(".txt")[0][1:] else True for _ in X1.columns.values.flatten()] 
                            ad = [True if str(_).split(",")[-1][2:-2]=="AD" else False for _ in X1.columns.values]
                            md = [True if str(_).split(",")[-1][2:-2]=="MD" else False for _ in X1.columns.values]
                            rd = [True if str(_).split(",")[-1][2:-2]=="RD" else False for _ in X1.columns.values]
                            fa = [True if str(_).split(",")[-1][2:-2]=="FA" else False for _ in X1.columns.values]
                            grp_fa = np.logical_and(grp,fa) ; grp_md = np.logical_and(grp,md)
                            grp_ad = np.logical_and(grp,ad) ; grp_rd = np.logical_and(grp,rd)
                            new_x = np.concatenate((X1.values[:,out_group], X1.values[:,grp_ad][:,::-1], X1.values[:,grp_md][:,::-1], 
                                                    X1.values[:,grp_rd][:,::-1], X1.values[:,grp_fa][:,::-1]), axis=1)
                            X1 = pd.DataFrame(new_x, columns=X1.columns.values)
                            grp = [True if str(_).split(",")[0][2:-2]==entry.name.split(".txt")[0][1:] else False for _ in Xpca.columns.values.flatten()] 
                            out_group = [False if str(_).split(",")[0][2:-2]==entry.name.split(".txt")[0][1:] else True for _ in Xpca.columns.values.flatten()] 
                            ad = [True if str(_).split(",")[-1][2:-2]=="AD" else False for _ in X1.columns.values]
                            md = [True if str(_).split(",")[-1][2:-2]=="MD" else False for _ in X1.columns.values]
                            rd = [True if str(_).split(",")[-1][2:-2]=="RD" else False for _ in X1.columns.values]
                            fa = [True if str(_).split(",")[-1][2:-2]=="FA" else False for _ in X1.columns.values]
                            grp_fa = np.logical_and(grp,fa) ; grp_md = np.logical_and(grp,md)
                            grp_ad = np.logical_and(grp,ad) ; grp_rd = np.logical_and(grp,rd)
                            new_x = np.concatenate((Xpca.values[:,out_group], Xpca.values[:,grp_ad], Xpca.values[:,grp_md], Xpca.values[:,grp_rd], Xpca.values[:,grp_fa]), axis=1)
                            Xpca = pd.DataFrame(new_x, columns=Xpca.columns.values)
                        except Exception as err:
                            print(err)
                            pass
                    
    """      
    # groups_ = list(np.array(groups_[0][yy]).flatten() for yy in range(len(groups_[0]))) 
    groups_names = [] ; corresp = {}
    for icol,col in enumerate(X1.columns.values):
        if not (col.split(",")[0] in groups_names):
            groups_names.append(col.split(",")[0])
            corresp[col.split(",")[0]] = [icol] 
        else:
            corresp[col.split(",")[0]].append(icol) 
            
    groups_ = [_ for key, _ in corresp.items()]    
    groups___ = list(np.array(ggg).flatten() for ggg in groups_ if len(ggg)>=1)#[lll] )
    
    groups_names = [] ; corresp = {}
    for icol,col in enumerate(Xpca.columns.values):
        if not (col.split(",")[0] in groups_names):
            groups_names.append(col.split(",")[0])
            corresp[col.split(",")[0]] = [icol] 
        else:
            corresp[col.split(",")[0]].append(icol) 
            
    groups_ = [_ for key, _ in corresp.items()]    
    groups_pca_ = list(np.array(ggg).flatten() for ggg in groups_ if len(ggg)>=1)#[lll] )
          
    gpca_results = my_group_pca(Xpca,y,max_components=4, pca_type=pca_type)#,mask=np.array(wm)) 
    components_ = gpca_results[0]
    pca_features_names = gpca_results[1]
    explained_variance_ = gpca_results[4] 
     
    pcaX = pd.DataFrame(gpca_results[3],columns=pca_features_names)
    pcaX_ = pcaX ; comp_names = np.array(pca_features_names)
    comp_names = np.unique([_.split(",")[0][2:-1] for _ in Xpca.columns.values]).tolist() 
    if pca_type.replace("scale_","") in ("bilat_pca","pca_without_age"): 
        comp_names = np.unique([_.split(",")[0][2:-1].replace(".","_") for _ in Xpca.columns.values]).tolist() 
    print("",file=open("pca_comps_"+type_+"_"+pca_type+".txt","w"))
    metrics = np.array(["AD", "MD", "RD", "FA"])
    for comp_indx, comp in enumerate(components_):
        comp_name = comp_names[0]
        comp_names.pop(0)
        if "right" in comp_name.split("_") or "left" in comp_name.split("_"):
            comp_names.pop(0)
        for ivect, vect in enumerate(comp):
            abs_contributions = abs(vect) 
            order = np.argsort(abs_contributions)[::-1]
            contributions = vect 
            var = explained_variance_[comp_indx][ivect]
            pca_comps = "" 
            if ("right" in comp_name.split("_") or "left" in comp_name.split("_")) and ivect==1: 
                for cont, metric in zip(contributions[order],metrics[order]):
                    pca_comps += str(np.round(cont*100,1)) + " " + metric + "  "
                print(comp_name+" (%s): "%np.round(var*100,1) +pca_comps,file=open("pca_comps_"+type_+"_"+pca_type+".txt","a"))
                comp_name = comp_name.replace("left","right")
                for ivect, vect in enumerate(comp):
                    abs_contributions = abs(vect) 
                    order = np.argsort(abs_contributions)[::-1]
                    contributions = vect 
                    var = explained_variance_[comp_indx][ivect]
                    pca_comps = "" 
                    for cont, metric in zip(contributions[order],metrics[order]):
                        pca_comps += str(np.round(cont*100,1)) + " " + metric + "  "
                    print(comp_name+" (%s): "%np.round(var*100,1) +pca_comps,file=open("pca_comps_"+type_+"_"+pca_type+".txt","a"))
            else:
                for cont, metric in zip(contributions[order],metrics[order]):
                    pca_comps += str(np.round(cont*100,1)) + " " + metric + "  "
                print(comp_name+" (%s): "%np.round(var*100,1) +pca_comps,file=open("pca_comps_"+type_+"_"+pca_type+".txt","a")) 
         
    groups_names = [] ; corresp = {}
    for icol,col in enumerate(pcaX.columns.values):
        if not ("_".join(col.split("_")[:-2]) in groups_names):
            groups_names.append("_".join(col.split("_")[:-2]))
            corresp["_".join(col.split("_")[:-2])] = [icol] 
        else:
            corresp["_".join(col.split("_")[:-2])].append(icol) 
          
    groups_ = [_ for key, _ in corresp.items()]    
    groups_group_pca_ = list(np.array(ggg).flatten() for ggg in groups_ if len(ggg)>=1)
     
    y = y.astype(np.int32) 
    refresh = False 
    
    pcaX.to_csv("/auto/home/users/d/r/drimez/data_after_"+pca_type+"_"+type_+".csv")
    
    return X1, Xpca, pcaX, morphom, groups___, groups_pca_, groups_group_pca_, y

scoring_functions = {"accuracy":get_scorer("accuracy"),"corrected_accuracy":balanced_metric_wrapper(get_scorer("accuracy")._score_func),
                     "balanced_accuracy":get_scorer("balanced_accuracy"),"precision":get_scorer("precision_micro"),
                     "geo":balanced_metric_wrapper(geometric_mean_score),"recall":get_scorer("recall_micro"),
                     "log_loss":balanced_metric_wrapper("neg_log_loss"),"top_2_accuracy":make_scorer(balanced_top_k_accuracy,needs_proba=True),
                     "unclassable":make_scorer(unclassable,needs_proba=True),"adjusted_top_2_accuracy":make_scorer(my_top_k_accuracy,needs_proba=True),
                     "sain_vs_comp":make_scorer(hesitate,needs_proba=True,class__=0),"comp_vs_vert":make_scorer(hesitate,needs_proba=True,class__=1),
                     "sain_vs_vert":make_scorer(hesitate,needs_proba=True,class__=2),"sain":make_scorer(confusion_matrix_scorer,needs_proba=False,k=0),
                     "comp":make_scorer(confusion_matrix_scorer,needs_proba=False,k=1),"vert":make_scorer(confusion_matrix_scorer,needs_proba=False,k=2)}

def mutual_info_classif(X,y,*kwargs):
    return MI_class(X,y,random_state=10000,*kwargs)
    
from sklearn.multiclass import OneVsRestClassifier

def single_model_comparison(ensemble_or_not, state=None, model=None, y=None, X1=None, Xpca=None, pcaX=None,groups_=None,
                            checkpoint_path=None, checkpoint_path_cv_=None,p_names=None,gpca_or_not=0, groups_pca_=None, scaler=None):
    warnings.filterwarnings("ignore")
    
    if state is not None:
        random.setstate(int(state))
        np.random.set_state(int(state))

    if checkpoint_path_cv_ is None:
        checkpoint_path = "/auto/home/users/d/r/drimez/Classify_bagging/checkpoints" if ensemble_or_not=="bagging" \
                          else  "/auto/home/users/d/r/drimez/Classify/checkpoints"  
         
        checkpoint_path_cv_ = "/auto/home/users/d/r/drimez/Classify_checkpoints_bagging" if ensemble_or_not=="bagging" \
                               else  "/auto/home/users/d/r/drimez/Classify_checkpoints"
                               
        checkpoint_path = "C:/Users/rimez/OneDrive/Documents/TFE/MRI/codes/Classify_bagging/checkpoints" if ensemble_or_not=="bagging" \
                          else  "C:/Users/rimez/OneDrive/Documents/TFE/MRI/codes/Classify/checkpoints"  
         
        checkpoint_path_cv_ = "C:/Users/rimez/OneDrive/Documents/TFE/MRI/codes/Classify_checkpoints_bagging" if ensemble_or_not=="bagging" \
                               else  "C:/Users/rimez/OneDrive/Documents/TFE/MRI/codes/Classify_checkpoints"  
    
    ensemble_kwargs = {"oob_score":[True],"n_jobs":[-1], "bootstrap_features":[False], # "class_weight":["balanced_subsample"], # no replacement drawing for features
                       "n_estimators":[60, 30, 10]} if ensemble_or_not=="bagging" else [None]  # "max_features":np.arange(0.4,1,0.2).tolist(), 
                       
    ensemble_estimator = My_BaggingClassifier if ensemble_or_not=="bagging" else None                      
                       
    default_params  =  {"imputer_kwargs":None,  # Use median imputation
                        "use_cv_estimator":False,  # Determine the best hyperparameters for the model in this given pipeline 
                        "verbose":0,  # Be quiet!  
                        "pipeline_verbosity":False,  # No really, be quiet!    
                        "l1_ratio":0.5,  # Explore the entire range of ``l1_ratio``  
                        "scale_l2_by":'group_length', 
                        "alpha":0.5,  
                        "fit_intercept":True,  
                        "max_iter":200} 
    
    n_class = len(np.unique(y))
    n_splits = min(np.unique(y,return_counts=True)[1].min(),5)
    test_cross_val = RepeatedStratifiedKFold(n_splits=int(n_splits), # leave one out for the least populated class
                                             n_repeats=int(np.rint(50/n_splits)), random_state=50) 
    params_cross_val = StratifiedKFold(n_splits=int(n_splits-1)) # leave one out for the least populated class 
    #
    # [,
    #               {"scaler":["standard", "minmax", "maxabs", "robust"],"feature_transformer": [GroupPCA],
    #               "feature_transformer_kwargs": [transformer_kwargs_pca], "groups":[groups_pca],"eps":[0.001,0.01,0.1]}] 
    
    final_results_svm = {} ; fitted_params_svm = [] 
    param_grid_svm = [{"scaler":[scaler],"feature_transformer": [False], 
                       "ensemble_meta_estimator_kwargs":ensemble_kwargs, "ensemble_meta_estimator":[ensemble_estimator],  # watch out for cross-training if adaboost 
                       "feature_transformer_kwargs": [None], "select":[f_classif,mutual_info_classif]}]
    estimator_kwargs_svm = [{"C":[0.01,0.1,1,10,100,1000],"kernel":["linear","rbf"],"class_weight":["balanced"],"break_ties":[True,False]}, 
                            {"C":[0.01,0.1,1,10,100,1000],"kernel":['poly'],"degree":[2,3],"class_weight":["balanced"],"coef0":[0,0.5,1,2],"break_ties":[True,False]}, 
                            {"C":[0.01,0.1,1,10,100,1000],"kernel":["sigmoid"],"class_weight":["balanced"],"coef0":[0,0.5,1,2],"break_ties":[True,False]}]
    model_ = model ; model=None
    def single_run_svm( X00, gpca_or_not, param_grid=param_grid_svm, estimator_kwargs=estimator_kwargs_svm, y=y, state=1000, ensemble_or_not=ensemble_or_not, ensemble_estimator=copy.copy(ensemble_estimator), n_class=n_class, n_splits=n_splits,
                        final_results=final_results_svm, fitted_params=fitted_params_svm,  test_cross_val=copy.copy(test_cross_val), params_cross_val=copy.copy(params_cross_val),checkpoint_path_cv=checkpoint_path_cv_,p_names=p_names): 
        X00 = X00.dropna(axis=1)
        try:
            svmcoef_ = {} ; at_least_one = False
            import shutup
            refresh = False
            shutup.please(); y0 = copy.copy(y)
            param_grid[0]["n_feats"] = np.arange(1,2,0.4).tolist() # if gpca_or_not==0 else (np.arange(2,5)/2).tolist()
            counter__ = 0  ; results = {"params":[],"results":[]}
            train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                 **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
            ext_name = "" if gpca_or_not==0 else "_gpca" 
            bad_params = [""] ; good_params = [""] ; final_results = {}
            if os.path.exists(checkpoint_path+"/unreliable_svm.txt"):
                try:
                    bad_params = pd.read_csv(checkpoint_path+"/unreliable_svm.txt", sep="}", skiprows=0,
                                             names=["fitted params","performances"]).values[1:,0].flatten()
                except Exception: 
                    pass
            else:
                print(str("fitted params")+",train_accuracy,test_accuracy,test_corrected_accuracy,test_balanced_accuracy",
                      file=open(checkpoint_path+"/unreliable_svm.txt","a"))
            if os.path.exists(checkpoint_path+"/reliable_svm.txt"):
                try:
                    good_params = pd.read_csv(checkpoint_path+"/reliable_svm.txt", sep="},", skiprows=0,
                                         names=["fitted params","perfs"]).values[1:,0].flatten() 
                except Exception: 
                    pass
            else:
                print(str("fitted params")+",train_accuracy,test_accuracy,test_corrected_accuracy,test_balanced_accuracy",
                      file=open(checkpoint_path+"/reliable_svm.txt","a"))
            for grid in param_grid:
                for params in traverse(grid):
                    for est_kwargs in traverse(estimator_kwargs): 
                        gc.collect()
                        fitted_params = dict({'scaler':params['scaler'], 
                                              "select":str(params['select']).split()[1]+"_"+str(params['n_feats'])}, 
                                              **{estk:estv for estk, estv in est_kwargs.items() 
                                                          if not estk in ("class_weight")}) 
                        if ensemble_or_not == "bagging":
                            fitted_params = dict(**fitted_params,**{"n_estimators":params["ensemble_meta_estimator_kwargs"]["n_estimators"]})
                        if gpca_or_not == 1:
                            fitted_params = dict(**fitted_params,**{"group_pca":"yes"}) 
                        else:
                            fitted_params = dict(**fitted_params,**{"group_pca":"no"})
                        if ((not str(fitted_params)[:-1] in bad_params) and (not str(fitted_params)[:-1] in good_params)) or refresh:
                            new_params = {key___:val___ for key___, val___ in params.items() if not (key___ in ("select","n_feats",
                                                                        "ensemble_meta_estimator_kwargs","ensemble_meta_estimator"))}
                            base_model = make_afq_classifier_pipeline(**new_params,**default_params)   
                            scaler = copy.copy(base_model["scale"])
                            base_model.set_params(**{bmkey:bmval for bmkey,bmval in base_model.get_params().items() 
                                                     if not bmkey in ("scale","impute")})    
                            ##### nested-cross-validation === estimates performancies of best pipeline
                            if not ensemble_or_not == "bagging":
                                new_params = {"estimate":SVC(random_state=state,probability=True,**est_kwargs)}   
                                base_model.set_params(**new_params)   
                            k = int(len(X00.iloc[0])/params["n_feats"])
                            cv_results = None ; checked_ = False
                            estimators = [] ; preprocessors = []
                            _path_cv = checkpoint_path_cv_ + "SVM" ; test_counts = 0
                            for train, test in copy.deepcopy(test_cross_val).split(X00.values,y0):
                                test_counts += 1 ; model = None
                                X0 = copy.copy(X00.iloc[train]) ; y = y0[train]  
                                if ensemble_or_not == "bagging":
                                    new_params = {"estimate":ensemble_estimator(SVC(random_state=state,probability=True,**est_kwargs),
                                                        random_state=state, y=y, **params["ensemble_meta_estimator_kwargs"])} 
                                    model = copy.copy(base_model) 
                                    model.set_params(**new_params)
                                else:
                                    model = copy.copy(base_model) 
                                scaler = scaler.fit(X0,y) 
                                X = scaler.transform(X0) 
                                selected_features = scaler.get_feature_names_out()
                                X = pd.DataFrame(X,columns=selected_features) 
                                aage = X["age"] if "age" in X.columns.values else None
                                selector = SelectKBest(params["select"],  k=k)
                                X = selector.fit_transform(copy.copy(X),y)
                                selected_features = selector.get_feature_names_out() 
                                if (not (aage is None)) and (not "age" in selected_features):
                                    selected_features = np.append(selected_features[:-1],aage)
                                X_test = selector.transform(scaler.transform(X00.values))
                                preprocessors.append([scaler,selector])
                                try: 
                                    cv_results = cross_validate_checkpoint( model, X, y=y, groups=None, scoring=scoring_functions, cv=params_cross_val, n_jobs=-1, 
                                                                            verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=False, 
                                                                            return_estimator=False, error_score='raise', workdir=checkpoint_path_cv, 
                                                                            checkpoint=False, force_refresh=refresh, serialize_cv=False) 
                                except tables.exceptions.HDF5ExtError:
                                    cv_results = cross_validate_checkpoint( model, X, y=y, groups=None, scoring=scoring_functions, cv=params_cross_val, n_jobs=-1, 
                                                                            verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=False, 
                                                                            return_estimator=False, error_score='raise', workdir=checkpoint_path_cv, 
                                                                            checkpoint=True, force_refresh=True, serialize_cv=False) 
                                except Exception as err:
                                    # print(err,file=open('/home/users/d/r/drimez/learning.out',"a"))
                                    # print(err,file=open(checkpoint_path+"err.txt","a"))
                                    try:
                                        cv_results = dict(**{"test_"+skey:[] for skey, _ in scoring_functions.items()})
                                        for train_param, test_param in copy.deepcopy(params_cross_val).split(X,y): 
                                            params_results    = _fit_and_score(estimator=model, X=X, y=y, scorer=scoring_functions, train=train_param, test=test_param,
                                                                        verbose=0, parameters=None, return_train_score=False, return_n_test_samples=False,
                                                                        return_times=False, return_estimator=True, error_score=np.nan, fit_params=None) 
                                            for tkey in [*scoring_functions.keys()]:
                                                cv_results["test_"+tkey] = np.append(cv_results["test_"+tkey],params_results["test_scores"][tkey])  
                                    except Exception as err:
                                        raise
                                     
                                if cv_results is not None:  
                                    p2 = 0 
                                    p1 = np.nan_to_num(np.nanmean(cv_results["test_accuracy"])) 
                                    if test_counts%n_splits==0 and test_counts>1: 
                                        p1 = np.nanmean(np.append(train_results["train_accuracy"][train_results["train_accuracy"]!=0],
                                                                                cv_results["test_accuracy"]))  
                                        p2 = np.nanmean(np.append(train_results["train_balanced_accuracy"][train_results["train_balanced_accuracy"]!=0],
                                                                                cv_results["test_balanced_accuracy"]))   
                                        if (p2<0.35 and p2>0 and (not ensemble_or_not == "bagging"))\
                                           or (p2<0.35 and p2>0 and (ensemble_or_not == "bagging")):
                                            train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                                                 **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
                                            # print("Model not reliable: "+str(fitted_params))  
                                            print(str(fitted_params)+","+str(p1)+","+str(0)+","+str(0)+","+str(p2),
                                                  file=open(checkpoint_path+"/unreliable_svm.txt","a"))
                                            break  
                                    elif (p1<0.4 and p1>0.05 and (not ensemble_or_not == "bagging"))\
                                         or (p1<0.25 and p1>0.05 and (ensemble_or_not == "bagging")):
                                        train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                                             **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
                                        # print("Model not reliable: "+str(fitted_params))  
                                        print(str(fitted_params)+","+str(p1)+","+str(0)+","+str(0)+","+str(p2),
                                              file=open(checkpoint_path+"/unreliable_svm.txt","a"))
                                        break 
                                    test_scores    = _fit_and_score(estimator=model, X=X_test, y=y0, scorer=scoring_functions, train=train, test=test,
                                                                    verbose=0, parameters=None, return_train_score=False, return_n_test_samples=False,
                                                                    return_times=False, return_estimator=True, error_score=np.nan, fit_params=None)
                                    print(str(fitted_params)+","+str(p1)+","+str(p2)+","+str(p_names[test]),
                                          file=open(checkpoint_path+"/patients.txt","a")) 
                                    estimators.append(test_scores["estimator"]) 
                                    for tkey in [*scoring_functions.keys()]:
                                        if np.nan_to_num(np.nanmean(cv_results["test_"+tkey]))!=0: 
                                            train_results["train_"+tkey] = np.append(train_results["train_"+tkey],cv_results["test_"+tkey]) 
                                        if test_scores["test_scores"][tkey]!=0: 
                                            train_results["test_"+tkey] = np.append(train_results["test_"+tkey],test_scores["test_scores"][tkey])  
                                    test_scores = None
                                        
                                else:
                                    estimators.append(None)   
                                  
                            if len(train_results["train_accuracy"])>1:
                                p1 = np.nan_to_num(np.nanmean(train_results["train_accuracy"]))
                                p2 = np.nan_to_num(np.nanmean(train_results["test_accuracy"])) 
                                p3 = np.nan_to_num(np.nanmean(train_results["test_corrected_accuracy"]))  
                                p4 = np.nan_to_num(np.nanmean(train_results["test_balanced_accuracy"])) 
                                p5 = np.nan_to_num(np.nanmean(train_results["test_balanced_accuracy"])) 
                                feat_names_perm = X00.columns.values.flatten() 
                                if (p2>=0.5 and (not p1>=0.12 +p2) and p5>=0.45 and len(np.unique(y))==3)\
                                    or (p2>=0.5 and (not p1>=0.12 +p2) and p5>=0.55 and len(np.unique(y))==2): 
                                    try:
                                        if est_kwargs['kernel'] == "linear":
                                            results["params"].append(fitted_params)
                                            results["results"].append(train_results)
                                            all_imp = None ; true_imp = None 
                                            final_results[str(fitted_params)] = 0 ; shutup.please()
                                            for train, test in copy.deepcopy(test_cross_val).split(X00.values,y0): 
                                                estimator = None ; X = None ; preprocessor = preprocessors[0] 
                                                X = preprocessor[0].transform(X00)   
                                                selected_features = preprocessor[0].get_feature_names_out()
                                                X = pd.DataFrame(X,columns=selected_features) 
                                                aage = X["age"] if "age" in X.columns.values else None 
                                                X = preprocessor[1].transform(X)  
                                                feat_names = preprocessor[1].get_feature_names_out() 
                                                if (not (aage is None)) and (not "age" in selected_features):
                                                    selected_features = np.append(selected_features[:-1],aage) 
                                                # X = X.values
                                                if not (estimators[0] is None):
                                                    estimator = estimators[0] 
                                                else: 
                                                    estimator = copy.copy(base_model).fit(X[train],y0[train]) 
                                                    
                                                estimator = Pipeline(steps=[("scale",preprocessor[0]),
                                                                            ("select",preprocessor[1]),
                                                                            ("estimate",estimator["estimate"])]) 
                                                this_all_imp, this_true_imp = get_importance(estimator["estimate"],estimator,X00.values,y0,
                                                                                              train,test,feature_names_in_=feat_names)
                                                   
                                                new_this_true_imp = this_true_imp        
                                                this_all_imp = np.array([_.reshape((1,len(_))) for _ in this_all_imp])                                           
                                                new_this_all_imp =  np.zeros((len(this_all_imp),len(this_all_imp[0]),len(feat_names_perm)))                                       
                                                for a_feat_idx, a_feature in enumerate(feat_names_perm):
                                                    if a_feature in feat_names_perm:
                                                        new_this_all_imp[...,a_feat_idx] = this_all_imp[...,a_feature==feat_names][...,0]
                                                
                                                if true_imp is None:
                                                    true_imp = new_this_true_imp
                                                else:
                                                    true_imp = np.concatenate((true_imp,new_this_true_imp),axis=0)
                                                if all_imp is None:
                                                    all_imp = new_this_all_imp
                                                else:
                                                    all_imp = np.concatenate((all_imp,new_this_all_imp),axis=0)
                                                    
                                                all_imp = np.concatenate((all_imp,new_this_all_imp),axis=0)
                                                estimators.pop(0) ; preprocessors.pop(0)
                                            # print(str(fitted_params)+","+str({"mean_"+key:np.nanmean(val) for key, val in train_results.items()})) 
                                            params_string = ""
                                            for par_name, params_val in fitted_params.items():
                                                params_string += str(par_name) + "=" + str(params_val) + "_"
                                                
                                            estimator = Pipeline(steps=[("scale",base_model["scale"]),
                                                                        ("select",SelectKBest(params["select"],  k=k)),
                                                                        ("estimate",copy.copy(base_model))])  
                                            _, mp, pv = permutation_test_score(estimator, X00, y0, n_jobs=-1, cv=RepeatedStratifiedKFold(n_splits=int(n_splits),  
                                                                                                                                         n_repeats=2, random_state=500),
                                                                                scoring="balanced_accuracy", n_permutations=399) 
                                            
                                            if pv<=0.15 and False:
                                                shap_explanation(X00,y0,copy.deepcopy(test_cross_val),estimator,
                                                                 checkpoint_path+"SHAPresults_SVM_"+ext_name+"/%s.txt"%params_string,
                                                                 fitted_params)
                                            plot_coef(all_imp,true_imp,mp,pv,feat_names_perm,checkpoint_path+"results_SVM_"+ext_name+"/%s.txt"%params_string,
                                                      "Support Vector classifier",fitted_params,train_results)
                                            print(str(fitted_params)+","+str(dict(**{"mean_"+key:np.nanmean(val) for key, val in train_results.items()},
                                                                              **{"std_balanced":np.nanstd(train_results["test_balanced_accuracy"]),"std":np.nanstd(train_results["test_accuracy"])})),
                                                  file=open(checkpoint_path+"/reliable_svm.txt","a")) 
                                            at_least_one = True ; final_results[str(fitted_params)] = train_results
                                        else: 
                                            all_imp = None ; true_imp = None ; shutup.please()
                                            for train, test in copy.deepcopy(test_cross_val).split(X00.values,y0): 
                                                bins = len(y0[test])/np.unique(y0[test],return_counts=True)[1] ; weights = []
                                                for nnnn, nnn_class in enumerate(np.unique(y0[test])):
                                                    weights = np.append(weights,[bins[nnnn]/len(bins) for _ in y0[test] if _==nnn_class])
                                                preprocessor = preprocessors[0]  
                                                X = preprocessor[0].transform(X00)  
                                                selected_features = preprocessor[0].get_feature_names_out()
                                                X = pd.DataFrame(X,columns=selected_features) 
                                                aage = X["age"] if "age" in X.columns.values else None 
                                                X = preprocessor[1].transform(X)  
                                                feat_names = preprocessor[1].get_feature_names_out() 
                                                if (not (aage is None)) and (not "age" in selected_features):
                                                    selected_features = np.append(selected_features[:-1],aage) 
                                                if not (estimators[0] is None):
                                                    estimator = estimators[0] 
                                                else: 
                                                    estimator = copy.copy(base_model).fit(X[train],y0[train]) 
                                                    
                                                estimator = Pipeline(steps=[("scale",preprocessor[0]),
                                                                            ("select",preprocessor[1]),
                                                                            ("estimate",estimator["estimate"])])  
                                                this_true_imp = permutation_importance( estimator, X00.values[test], y0[test], scoring="balanced_accuracy",
                                                                                            n_jobs=-1, n_repeats=10)
                                                this_true_imp = this_true_imp.importances.T
                                                new_this_true_imp = this_true_imp 
                                                this_all_imp = np.zeros_like(this_true_imp)
                                                
                                                new_this_all_imp = np.array([_.reshape((1,len(_))) for _ in this_all_imp])              
                                                
                                                if true_imp is None:
                                                    true_imp = new_this_true_imp
                                                else:
                                                    true_imp = np.concatenate((true_imp,new_this_true_imp),axis=0)
                                                if all_imp is None:
                                                    all_imp = new_this_all_imp
                                                else:
                                                    all_imp = np.concatenate((all_imp,new_this_all_imp),axis=0)
                                                    
                                                estimators.pop(0) ; preprocessors.pop(0) 
                                                
                                            estimator = Pipeline(steps=[("scale",base_model["scale"]),
                                                                        ("select",SelectKBest(params["select"],  k=k)),
                                                                        ("estimate",copy.copy(base_model))])  
                                            _, mp, pv = permutation_test_score(estimator, X00, y0, n_jobs=-1, cv=RepeatedStratifiedKFold(n_splits=int(n_splits),  
                                                                                                                                         n_repeats=2, random_state=500),
                                                                                scoring="balanced_accuracy", n_permutations=399)
                                            
                                            # print(str(fitted_params)+","+str({"mean_"+key:np.nanmean(val) for key, val in train_results.items()})) 
                                            params_string = ""
                                            for par_name, params_val in fitted_params.items():
                                                params_string += str(par_name) + "=" + str(params_val) + "_"
                                            if pv<=0.15 and False:
                                                shap_explanation(X00,y0,copy.deepcopy(test_cross_val),estimator,
                                                                 checkpoint_path+"SHAPresults_SVM_"+ext_name+"/%s.txt"%params_string,
                                                                 fitted_params)
                                             
                                            plot_coef(all_imp,true_imp,mp,pv,feat_names_perm,checkpoint_path+"results_SVM_"+ext_name+"/%s.txt"%params_string,
                                                      "Support Vector classifier",fitted_params,train_results) 
                                            print(str(fitted_params)+","+str(dict(**{"mean_"+key:np.nanmean(val) for key, val in train_results.items()},
                                                                              **{"std_balanced":np.nanstd(train_results["test_balanced_accuracy"]),"std":np.nanstd(train_results["test_accuracy"])})),
                                                  file=open(checkpoint_path+"/reliable_svm.txt","a")) 
                                            at_least_one = True ; final_results[str(fitted_params)] = train_results
                                    except Exception as err: 
                                        raise  
                                else:
                                    # print("Model not reliable: "+str(fitted_params))  
                                    print(str(fitted_params)+","+str(p1)+","+str(p2)+","+str(p3)+","+str(p5),
                                          file=open(checkpoint_path+"/unreliable_svm.txt","a")) 
                                
                            train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                                 **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
                                 
            ext_name = "" if gpca_or_not==0 else "_gpca"
            compressed_pickle(checkpoint_path+"results_SVM"+ext_name,final_results)
            if at_least_one:
                plot_results_svm(final_results,"SVM"+ext_name,ensemble_or_not)    
        except Exception as err:  
            raise
               
    final_results_rdf = {} ; fitted_params_rdf = [] 
    param_grid_rdf = [{"scaler":[scaler],"feature_transformer": [False], 
                       "ensemble_meta_estimator_kwargs":[None], "ensemble_meta_estimator":[None],   # watch out for cross-training if adaboost 
                       "feature_transformer_kwargs": [None], "select":[f_classif,mutual_info_classif], "n_feats":np.arange(1,2,0.5).tolist()}]
    estimator_kwargs_rdf = {"n_estimators":[10,20,50,100],"max_depth":np.arange(8,20,2).tolist(),"class_weight":["balanced_subsample"]}
    model=None ; refresh = True
    def single_run_rdf( X00, gpca_or_not, param_grid=param_grid_rdf, estimator_kwargs=estimator_kwargs_rdf, y=y, state=1000, ensemble_or_not=ensemble_or_not, ensemble_estimator=ensemble_estimator,n_class=n_class,
                        final_results=final_results_rdf, fitted_params=fitted_params_rdf, test_cross_val=copy.copy(test_cross_val), params_cross_val=copy.copy(params_cross_val),checkpoint_path_cv=checkpoint_path_cv_):
        try:  
            X00 = X00.dropna(axis=1)
            svmcoef_ = {} ; at_least_one = False
            import shutup
            shutup.please(); y0 = copy.copy(y)
            param_grid[0]["n_feats"] = np.arange(1,5).tolist() if gpca_or_not==0 else (np.arange(2,5)/2).tolist()
            counter__ = 0  ; results = {"params":[],"results":[]}
            train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                 **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
            ext_name = "" if gpca_or_not==0 else "_gpca" 
            bad_params = [""] ; good_params = [""] ; final_results = {}
            if os.path.exists(checkpoint_path+"/unreliable_rdf.txt"):
                try:
                    bad_params = pd.read_csv(checkpoint_path+"/unreliable_rdf.txt", sep="}", skiprows=0,
                                             names=["fitted params","performances"]).values[1:,0].flatten()
                except Exception: 
                    pass
            else:
                print(str("fitted params")+",train_accuracy,test_accuracy,test_corrected_accuracy,test_balanced_accuracy",
                      file=open(checkpoint_path+"/unreliable_rdf.txt","a"))
            if os.path.exists(checkpoint_path+"/reliable_rdf.txt"):
                try:
                    good_params = pd.read_csv(checkpoint_path+"/reliable_rdf.txt", sep="},", skiprows=0,
                                         names=["fitted params","perfs"]).values[1:,0].flatten() 
                except Exception: 
                    pass
            else:
                print(str("fitted params")+",train_accuracy,test_accuracy,test_corrected_accuracy,test_balanced_accuracy",
                      file=open(checkpoint_path+"/reliable_rdf.txt","a")) 
            for grid in param_grid: 
                for params in traverse(grid):
                    for est_kwargs in traverse(estimator_kwargs): 
                        gc.collect() 
                        fitted_params = dict({'scaler':params['scaler'], 
                                              "select":str(params['select']).split()[1]+"_"+str(params['n_feats'])}, 
                                             **{estk:estv for estk, estv in est_kwargs.items() 
                                                          if not estk in ("class_weight")}) 
                        if ensemble_or_not == "bagging":
                            fitted_params = dict(**fitted_params,**{"n_estimators":params["ensemble_meta_estimator_kwargs"]["n_estimators"]})
                        if gpca_or_not == 1:
                            fitted_params = dict(**fitted_params,**{"group_pca":"yes"}) 
                        else:
                            fitted_params = dict(**fitted_params,**{"group_pca":"no"})
                        if ((not str(fitted_params) in bad_params) and (not str(fitted_params) in good_params)) or refresh:
                            new_params = {key___:val___ for key___, val___ in params.items() if not (key___ in ("select","n_feats"))}
                            base_model = make_afq_classifier_pipeline(**new_params,**default_params)   
                            scaler = copy.copy(base_model["scale"])
                            base_model.set_params(**{bmkey:bmval for bmkey,bmval in base_model.get_params().items() 
                                                     if not bmkey in ("scale","impute")})    
                            ##### nested-cross-validation === estimates performancies of best pipeline
                            if not ensemble_or_not == "bagging":
                                new_params = {"estimate":My_RandomForestClassifier(random_state=state,n_jobs=-1,**est_kwargs) }    
                                base_model.set_params(**new_params)        
                            k = int(len(X00.iloc[0])/params["n_feats"])
                            cv_results = None ; checked_ = False
                            estimators = [] ; preprocessors = []
                            checkpoint_path_cv = checkpoint_path_cv_ + "elast" ; test_counts = 0
                            for train, test in copy.deepcopy(test_cross_val).split(X00,y0):
                                test_counts += 1
                                X0 = X00.iloc[train] ; y = y0[train]  
                                scaler = scaler.fit(X0,y) 
                                X = scaler.transform(X0) 
                                selected_features = scaler.get_feature_names_out()
                                X = pd.DataFrame(X,columns=selected_features) 
                                selector = SelectKBest(params["select"],  k=k)
                                X = selector.fit_transform(copy.copy(X),y)
                                selected_features = selector.get_feature_names_out()  
                                X_test = selector.transform(scaler.transform(X00.values))
                                model = copy.copy(base_model) ; preprocessors.append([scaler,selector])
                                try: 
                                    cv_results = cross_validate_checkpoint( model, X, y=y, groups=None, scoring=scoring_functions, cv=params_cross_val, n_jobs=1, 
                                                                            verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=False, 
                                                                            return_estimator=False, error_score='raise', workdir=checkpoint_path_cv, 
                                                                            checkpoint=True, force_refresh=refresh, serialize_cv=False) 
                                except tables.exceptions.HDF5ExtError:
                                    cv_results = cross_validate_checkpoint( model, X, y=y, groups=None, scoring=scoring_functions, cv=params_cross_val, n_jobs=1, 
                                                                            verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=False, 
                                                                            return_estimator=False, error_score='raise', workdir=checkpoint_path_cv, 
                                                                            checkpoint=True, force_refresh=True, serialize_cv=False) 
                                except Exception as err:
                                    # print(err,file=open('/home/users/d/r/drimez/learning.out',"a"))
                                    # print(err,file=open(checkpoint_path+"err.txt","a"))
                                    raise
                                     
                                if cv_results is not None:  
                                    p2 = 0 
                                    p1 = np.nan_to_num(np.nanmean(cv_results["test_accuracy"])) 
                                    if test_counts%n_splits==0 and test_counts>1: 
                                        p1 = np.nanmean(np.append(train_results["train_accuracy"][train_results["train_accuracy"]!=0],
                                                                                cv_results["test_accuracy"]))  
                                        p2 = np.nanmean(np.append(train_results["train_balanced_accuracy"][train_results["train_balanced_accuracy"]!=0],
                                                                                cv_results["test_balanced_accuracy"]))   
                                        if (p1<0.5 and p1>0 and p2<0.4 and p2>0 and (not ensemble_or_not == "bagging"))\
                                           or (p1<0.3 and p1>0 and p2<0.3 and p2>0 and (ensemble_or_not == "bagging")): 
                                            train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                                                 **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
                                            print("Model not reliable: "+str(fitted_params))  
                                            print(str(fitted_params)+","+str(p1)+","+str(0)+","+str(0)+","+str(p2),
                                                  file=open(checkpoint_path+"/unreliable_rdf.txt","a"))
                                            break  
                                    elif (p1<0.4 and p1>0.05 and (not ensemble_or_not == "bagging"))\
                                         or (p1<0.25 and p1>0.05 and (ensemble_or_not == "bagging")):
                                        train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                                             **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
                                        print("Model not reliable: "+str(fitted_params))  
                                        print(str(fitted_params)+","+str(p1)+","+str(0)+","+str(0)+","+str(p2),
                                              file=open(checkpoint_path+"/unreliable_rdf.txt","a"))
                                        break 
                                    test_scores    = _fit_and_score(estimator=model, X=X_test, y=y0, scorer=scoring_functions, train=train, test=test,
                                                                    verbose=10, parameters=None, return_train_score=False, return_n_test_samples=False,
                                                                    return_times=False, return_estimator=True, error_score=0, fit_params=None)
                                    
                                    estimators.append(test_scores["estimator"]) 
                                    for tkey in [*scoring_functions.keys()]:
                                        if np.nan_to_num(np.nanmean(cv_results["test_"+tkey]))!=0: 
                                            train_results["train_"+tkey] = np.append(train_results["train_"+tkey],cv_results["test_"+tkey]) 
                                        if test_scores["test_scores"][tkey]!=0: 
                                            train_results["test_"+tkey] = np.append(train_results["test_"+tkey],test_scores["test_scores"][tkey])  
                                    test_scores = None
                                else:
                                    estimators.append(None)  
                                  
                            if len(train_results["train_accuracy"])>1:
                                p1 = np.nan_to_num(np.nanmean(train_results["train_accuracy"]))
                                p2 = np.nan_to_num(np.nanmean(train_results["test_accuracy"])) 
                                p3 = np.nan_to_num(np.nanmean(train_results["test_corrected_accuracy"]))  
                                p4 = np.nan_to_num(np.nanmean(train_results["test_balanced_accuracy"])) 
                                p5 = np.nan_to_num(np.nanmean(train_results["test_balanced_accuracy"])) 
                                feat_names_perm = X00.columns.values.flatten() 
                                if (p2>=0.5 and (not p1>=0.12 +p2) and p5>=0.4 and len(np.unique(y))==3)\
                                    or (p2>=0.5 and (not p1>=0.12 +p2) and p5>=0.55 and len(np.unique(y))==2): 
                                    try: 
                                        results["params"].append(fitted_params)
                                        results["results"].append(train_results)
                                        all_imp = None ; true_imp = None 
                                        final_results[str(fitted_params)] = 0 ; shutup.please()
                                        for train, test in copy.deepcopy(test_cross_val).split(X00.values,y0): 
                                            estimator = None ; X = None ; preprocessor = preprocessors[0] 
                                            X = preprocessor[0].transform(X00)   
                                            selected_features = preprocessor[0].get_feature_names_out()
                                            X = pd.DataFrame(X,columns=selected_features) 
                                            aage = X["age"] if "age" in X.columns.values else None 
                                            X = preprocessor[1].transform(X)  
                                            feat_names = preprocessor[1].get_feature_names_out() 
                                            if (not (aage is None)) and (not "age" in selected_features):
                                                selected_features = np.append(selected_features[:-1],aage) 
                                            # X = X.values
                                            if not (estimators[0] is None):
                                                estimator = estimators[0] 
                                            else: 
                                                estimator = copy.copy(base_model).fit(X[train],y0[train]) 
                                                
                                            estimator = Pipeline(steps=[("scale",preprocessor[0]),
                                                                        ("select",preprocessor[1]),
                                                                        ("estimate",estimator["estimate"])]) 
                                            this_all_imp, this_true_imp = get_importance(estimator["estimate"],estimator,X00.values,y0,
                                                                                          train,test,feature_names_in_=feat_names)
                                               
                                            new_this_true_imp = this_true_imp        
                                            this_all_imp = np.array([_.reshape((1,len(_))) for _ in this_all_imp])                                           
                                            new_this_all_imp =  np.zeros((len(this_all_imp),len(this_all_imp[0]),len(feat_names_perm)))                                       
                                            for a_feat_idx, a_feature in enumerate(feat_names_perm):
                                                if a_feature in feat_names_perm:
                                                    new_this_all_imp[...,a_feat_idx] = this_all_imp[...,a_feature==feat_names][:,:,0]
                                            
                                            if true_imp is None:
                                                true_imp = new_this_true_imp
                                            else:
                                                true_imp = np.concatenate((true_imp,new_this_true_imp),axis=0)
                                            if all_imp is None:
                                                all_imp = new_this_all_imp
                                            else:
                                                all_imp = np.concatenate((all_imp,new_this_all_imp),axis=0)
                                                
                                            all_imp = np.concatenate((all_imp,new_this_all_imp),axis=0)
                                            estimators.pop(0) ; preprocessors.pop(0)
                                        # print(str(fitted_params)+","+str({"mean_"+key:np.nanmean(val) for key, val in train_results.items()})) 
                                        params_string = ""
                                        for par_name, params_val in fitted_params.items():
                                            params_string += str(par_name) + "=" + str(params_val) + "_"
                                            
                                        estimator = Pipeline(steps=[("scale",base_model["scale"]),
                                                                    ("select",SelectKBest(params["select"],  k=k)),
                                                                    ("estimate",copy.copy(base_model))])  
                                        _, mp, pv = permutation_test_score(estimator, X00, y0, n_jobs=-1, cv=RepeatedStratifiedKFold(n_splits=int(n_splits),  
                                                                                                                                     n_repeats=2, random_state=500),
                                                                            scoring="balanced_accuracy", n_permutations=399) 
                                        
                                        if pv<=0.15 and False:
                                            shap_explanation(X00,y0,copy.deepcopy(test_cross_val),estimator,
                                                             checkpoint_path+"SHAPresults_RDF_"+ext_name+"/%s.txt"%params_string,
                                                             fitted_params)
                                        plot_coef(all_imp,true_imp,mp,pv,feat_names_perm,checkpoint_path+"results_RDF_"+ext_name+"/%s.txt"%params_string,
                                                  "Random Forest",fitted_params,train_results)
                                        print(str(fitted_params)+","+str(dict(**{"mean_"+key:np.nanmean(val) for key, val in train_results.items()},
                                                                              **{"std_balanced":np.nanstd(train_results["test_balanced_accuracy"]),"std":np.nanstd(train_results["test_accuracy"])})),
                                              file=open(checkpoint_path+"/reliable_rdf.txt","a")) 
                                        at_least_one = True ; final_results[str(fitted_params)] = train_results 
                                    except Exception as err: 
                                        raise  
                                else:
                                    print("Model not reliable: "+str(fitted_params))
                                    print(str(fitted_params)+","+str(p1)+","+str(p2)+","+str(p3)+","+str(p4)+","+str(np.nanmean(train_results["test_unclassable"]))+","+str(np.nanmean(train_results["test_adjusted_top_2_accuracy"])),
                                          file=open(checkpoint_path+"/unreliable_rdf.txt","a"))  
                                    
                        train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                             **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
             
            ext_name = "" if gpca_or_not==0 else "_gpca"
            gc.collect()
            if at_least_one:
                plot_results_forest(checkpoint_path+"results_RandForest"+ext_name,"RandForest"+ext_name,ensemble_or_not)     
        except Exception as err:
            print(err)
            raise
     
    final_results_elast = {} ; fitted_params_elast = [] 
    param_grid_elast = [{"scaler":[scaler],"feature_transformer": [False], 
                         "ensemble_meta_estimator_kwargs":ensemble_kwargs, "ensemble_meta_estimator":[ensemble_estimator],   # watch out for cross-training if adaboost 
                         "feature_transformer_kwargs": [None], "select":[f_classif,mutual_info_classif], "n_feats":np.arange(1,5).tolist()}]
    estimator_kwargs_elast = {"penalty":["elasticnet"], "solver":["saga"],"C":[0.1,1,10,100,0.01], "l1_ratio":np.linspace(0,1,11).tolist(),
                              "max_iter":[2000],"class_weight":["balanced"],"tol":[0.001]}
    model=None 
    def single_run_elast( X00, gpca_or_not, param_grid=param_grid_elast, estimator_kwargs=estimator_kwargs_elast, y=y, state=1000, ensemble_or_not=ensemble_or_not, ensemble_estimator=ensemble_estimator,n_class=n_class,
                          final_results=final_results_elast, fitted_params=fitted_params_elast, test_cross_val=copy.copy(test_cross_val), params_cross_val=copy.copy(params_cross_val),checkpoint_path_cv=checkpoint_path_cv_):
        try: 
            X00 = X00.dropna(axis=1)
            svmcoef_ = {} ; at_least_one = False
            import shutup
            shutup.please(); y0 = copy.copy(y)
            param_grid[0]["n_feats"] = np.arange(1,5).tolist() if gpca_or_not==0 else (np.arange(2,5)/2).tolist()
            counter__ = 0  ; results = {"params":[],"results":[]}
            train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                 **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
            ext_name = "" if gpca_or_not==0 else "_gpca" 
            bad_params = [""] ; good_params = [""] ; final_results = {}
            if os.path.exists(checkpoint_path+"/unreliable_elast.txt"):
                try:
                    bad_params = pd.read_csv(checkpoint_path+"/unreliable_elast.txt", sep="}", skiprows=0,
                                             names=["fitted params","performances"]).values[1:,0].flatten()
                except Exception: 
                    pass
            else:
                print(str("fitted params")+",train_accuracy,test_accuracy,test_corrected_accuracy,test_balanced_accuracy",
                      file=open(checkpoint_path+"/unreliable_elast.txt","a"))
            if os.path.exists(checkpoint_path+"/reliable_elast.txt"):
                try:
                    good_params = pd.read_csv(checkpoint_path+"/reliable_elast.txt", sep="},", skiprows=0,
                                         names=["fitted params","perfs"]).values[1:,0].flatten() 
                except Exception: 
                    pass
            else:
                print(str("fitted params")+",train_accuracy,test_accuracy,test_corrected_accuracy,test_balanced_accuracy",
                      file=open(checkpoint_path+"/reliable_elast.txt","a"))
            for grid in param_grid: 
                for params in traverse(grid):
                    for est_kwargs in traverse(estimator_kwargs): 
                        gc.collect()
                        fitted_params = {'scaler':params['scaler'],'Cs':est_kwargs['C'], "l1_ratios":est_kwargs['l1_ratio'], 
                                         "select":str(params['select'])+"_"+str(params['n_feats'])}
                        fitted_params = dict({'scaler':params['scaler'], 
                                              "select":str(params['select']).split()[1]+"_"+str(params['n_feats'])}, 
                                             **{estk:estv for estk, estv in est_kwargs.items() 
                                                          if not estk in ("class_weight","max_iter","tol","solver","penalty")})
                        if ensemble_or_not == "bagging":
                            fitted_params = dict(**fitted_params,**{"n_estimators":params["ensemble_meta_estimator_kwargs"]["n_estimators"]})
                        if gpca_or_not == 1:
                            fitted_params = dict(**fitted_params,**{"group_pca":"yes"}) 
                        else:
                            fitted_params = dict(**fitted_params,**{"group_pca":"no"})
                        print(fitted_params)
                        if ((not str(fitted_params) in bad_params) and (not str(fitted_params) in good_params)) or refresh:
                            print("processing")
                            new_params = {key___:val___ for key___, val___ in params.items() if not (key___ in ("select","n_feats"))}
                            base_model = make_afq_classifier_pipeline(**new_params,**default_params)   
                            scaler = base_model["scale"]
                            base_model.set_params(**{bmkey:bmval for bmkey,bmval in base_model.get_params().items() 
                                                     if not bmkey in ("scale","impute")})    
                            new_params = {key___:val___ for key___, val___ in params.items() if not (key___ in ("select","n_feats"))}
                            base_model = make_afq_classifier_pipeline(**new_params,**default_params)   
                            new_params = {"estimate":LogisticRegression(random_state=state,n_jobs=-1,**est_kwargs) } if not ensemble_or_not == "bagging" \
                                          else {"estimate":ensemble_estimator(LogisticRegression(random_state=state,n_jobs=-1,**est_kwargs),
                                                                              random_state=state, **params["ensemble_meta_estimator_kwargs"])}
                            base_model.set_params(**new_params)      
                            k = int(len(X00.iloc[0])/params["n_feats"])
                            cv_results = None ; checked_ = False
                            estimators = [] ; preprocessors = []
                            checkpoint_path_cv = checkpoint_path_cv_ + "elast" ; test_counts = 0
                            for train, test in copy.deepcopy(test_cross_val).split(X00,y0):
                                test_counts += 1 ; gc.collect()
                                X0 = X00.iloc[train] ; y = y0[train]  
                                scaler = scaler.fit(X0,y) 
                                X = scaler.transform(X0) 
                                selected_features = scaler.get_feature_names_out()
                                X = pd.DataFrame(X,columns=selected_features) 
                                selector = SelectKBest(params["select"],  k=k)
                                X = selector.fit_transform(copy.copy(X),y)
                                selected_features = selector.get_feature_names_out()  
                                X_test = selector.transform(scaler.transform(X00.values))
                                model = copy.copy(base_model) ; preprocessors.append([scaler,selector])
                                try: 
                                    cv_results = cross_validate_checkpoint( model, X, y=y, groups=None, scoring=scoring_functions, cv=params_cross_val, n_jobs=-1, 
                                                                            verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=False, 
                                                                            return_estimator=False, error_score='raise', workdir=checkpoint_path_cv, 
                                                                            checkpoint=False, force_refresh=refresh, serialize_cv=False) 
                                except tables.exceptions.HDF5ExtError:
                                    cv_results = cross_validate_checkpoint( model, X, y=y, groups=None, scoring=scoring_functions, cv=params_cross_val, n_jobs=-1, 
                                                                            verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=False, 
                                                                            return_estimator=False, error_score='raise', workdir=checkpoint_path_cv, 
                                                                            checkpoint=False, force_refresh=True, serialize_cv=False) 
                                except Exception as err:
                                    # print(err,file=open('/home/users/d/r/drimez/learning.out',"a"))
                                    # print(err,file=open(checkpoint_path+"err.txt","a"))
                                    try:
                                        cv_results = dict(**{"test_"+skey:[] for skey, _ in scoring_functions.items()})
                                        for train_param, test_param in copy.deepcopy(params_cross_val).split(X,y): 
                                            params_results    = _fit_and_score(estimator=model, X=X, y=y, scorer=scoring_functions, train=train_param, test=test_param,
                                                                        verbose=0, parameters=None, return_train_score=False, return_n_test_samples=False,
                                                                        return_times=False, return_estimator=True, error_score="raise", fit_params=None) 
                                            for tkey in [*scoring_functions.keys()]:
                                                cv_results["test_"+tkey] = np.append(cv_results["test_"+tkey],params_results["test_scores"][tkey])  
                                    except Exception as err:
                                        raise 
                                     
                                if cv_results is not None:  
                                    p2 = 0 
                                    p1 = np.nan_to_num(np.nanmean(cv_results["test_accuracy"])) 
                                    if test_counts%n_splits==0 and test_counts>1: 
                                        p1 = np.nanmean(np.append(np.array(train_results["train_accuracy"])[np.array(train_results["train_accuracy"])!=0],
                                                                                cv_results["test_accuracy"]))  
                                        p2 = np.nanmean(np.append(np.array(train_results["train_balanced_accuracy"])[np.array(train_results["train_balanced_accuracy"])!=0],
                                                                                cv_results["test_balanced_accuracy"]))   
                                        if (p1<0.4 and p1>0 and p2<0.35 and p2>0 and (not ensemble_or_not == "bagging"))\
                                           or (p1<0.33 and p1>0 and p2<0.33 and p2>0 and (ensemble_or_not == "bagging")): 
                                            train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                                                 **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
                                            print("Model not reliable: "+str(fitted_params))  
                                            print(str(fitted_params)+","+str(p1)+","+str(0)+","+str(0)+","+str(p2),
                                                  file=open(checkpoint_path+"/unreliable_elast.txt","a"))
                                            break  
                                    elif (p1<0.35 and p1>0.05 and (not ensemble_or_not == "bagging"))\
                                         or (p1<0.25 and p1>0.05 and (ensemble_or_not == "bagging")):
                                        train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                                             **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
                                        print("Model not reliable: "+str(fitted_params))  
                                        print(str(fitted_params)+","+str(p1)+","+str(0)+","+str(0)+","+str(p2),
                                              file=open(checkpoint_path+"/unreliable_elast.txt","a"))
                                        break 
                                    test_scores    = _fit_and_score(estimator=model, X=X_test, y=y0, scorer=scoring_functions, train=train, test=test,
                                                                    verbose=10, parameters=None, return_train_score=False, return_n_test_samples=False,
                                                                    return_times=False, return_estimator=True, error_score="raise", fit_params=None)
                                    
                                    estimators.append(test_scores["estimator"]) 
                                    for tkey in [*scoring_functions.keys()]:
                                        if np.nan_to_num(np.nanmean(cv_results["test_"+tkey]))!=0: 
                                            train_results["train_"+tkey] = np.append(train_results["train_"+tkey],cv_results["test_"+tkey]) 
                                        if test_scores["test_scores"][tkey]!=0: 
                                            train_results["test_"+tkey] = np.append(train_results["test_"+tkey],test_scores["test_scores"][tkey])  
                                    test_scores = None
                                else:
                                    estimators.append(None)  
                                  
                            if len(train_results["train_accuracy"])>1:
                                p1 = np.nan_to_num(np.nanmean(train_results["train_accuracy"]))
                                p2 = np.nan_to_num(np.nanmean(train_results["test_accuracy"])) 
                                p3 = np.nan_to_num(np.nanmean(train_results["test_corrected_accuracy"]))  
                                p5 = np.nan_to_num(np.nanmean(train_results["test_balanced_accuracy"])) 
                                p4 = np.nan_to_num(np.nanmean(train_results["test_balanced_accuracy"])) 
                                feat_names_perm = X00.columns.values.flatten() 
                                if (((p2>=0.4 and (not p1>=0.12 +p2) and p5>=0.45 and len(np.unique(y))==3)\
                                    or (p2>=0.5 and (not p1>=0.12 +p2) and p5>=0.55 and len(np.unique(y))==2)) and (not ensemble_or_not == "bagging"))\
                                    or (((p2>=0.35 and (not p1>=0.1 +p2) and p5>=0.35 and len(np.unique(y))==3)\
                                         or (p2>=0.5 and (not p1>=0.12 +p2) and p5>=0.55 and len(np.unique(y))==2)) and (ensemble_or_not == "bagging")): 
                                    try: 
                                        results["params"].append(fitted_params)
                                        results["results"].append(train_results)
                                        all_imp = None ; true_imp = None 
                                        final_results[str(fitted_params)] = 0 ; shutup.please()
                                        for train, test in copy.deepcopy(test_cross_val).split(X00.values,y0): 
                                            estimator = None ; X = None ; preprocessor = preprocessors[0] 
                                            X = preprocessor[0].transform(X00)   
                                            selected_features = preprocessor[0].get_feature_names_out()
                                            X = pd.DataFrame(X,columns=selected_features) 
                                            aage = X["age"] if "age" in X.columns.values else None 
                                            X = preprocessor[1].transform(X)  
                                            feat_names = preprocessor[1].get_feature_names_out() 
                                            if (not (aage is None)) and (not "age" in selected_features):
                                                selected_features = np.append(selected_features[:-1],aage) 
                                            # X = X.values
                                            if not (estimators[0] is None):
                                                estimator = estimators[0] 
                                            else: 
                                                estimator = copy.copy(base_model).fit(X[train],y0[train]) 
                                                
                                            estimator = Pipeline(steps=[("scale",preprocessor[0]),
                                                                        ("select",preprocessor[1]),
                                                                        ("estimate",estimator["estimate"])]) 
                                            this_all_imp, this_true_imp = get_importance(estimator["estimate"],estimator,X00.values,y0,
                                                                                          train,test,feature_names_in_=feat_names)
                                               
                                            new_this_true_imp = this_true_imp        
                                            this_all_imp = np.array([_.reshape((1,len(_))) for _ in this_all_imp])                                           
                                            new_this_all_imp =  np.zeros((len(this_all_imp),len(this_all_imp[0]),len(feat_names_perm)))                                       
                                            for a_feat_idx, a_feature in enumerate(feat_names_perm):
                                                if a_feature in feat_names_perm:
                                                    new_this_all_imp[...,a_feat_idx] = this_all_imp[...,a_feature==feat_names][...,0]
                                            
                                            if true_imp is None:
                                                true_imp = new_this_true_imp
                                            else:
                                                true_imp = np.concatenate((true_imp,new_this_true_imp),axis=0)
                                            if all_imp is None:
                                                all_imp = new_this_all_imp
                                            else:
                                                all_imp = np.concatenate((all_imp,new_this_all_imp),axis=0)
                                                
                                            all_imp = np.concatenate((all_imp,new_this_all_imp),axis=0)
                                            estimators.pop(0) ; preprocessors.pop(0)
                                        # print(str(fitted_params)+","+str({"mean_"+key:np.nanmean(val) for key, val in train_results.items()})) 
                                        params_string = ""
                                        for par_name, params_val in fitted_params.items():
                                            params_string += str(par_name) + "=" + str(params_val) + "_"
                                            
                                        estimator = Pipeline(steps=[("scale",base_model["scale"]),
                                                                    ("select",SelectKBest(params["select"],  k=k)),
                                                                    ("estimate",copy.copy(base_model))])  
                                        _, mp, pv = permutation_test_score(estimator, X00, y0, n_jobs=-1, cv=RepeatedStratifiedKFold(n_splits=int(n_splits),  
                                                                                                                                     n_repeats=2, random_state=500),
                                                                            scoring="balanced_accuracy", n_permutations=399) 
                                        
                                        if pv<=0.15 and False:
                                            shap_explanation(X00,y0,copy.deepcopy(test_cross_val),estimator,
                                                             checkpoint_path+"SHAPresults_elast_"+ext_name+"/%s.txt"%params_string,
                                                             fitted_params)
                                        plot_coef(all_imp,true_imp,mp,pv,feat_names_perm,checkpoint_path+"results_elast_"+ext_name+"/%s.txt"%params_string,
                                                  "Elastic Net",fitted_params,train_results)
                                        print(str(fitted_params)+","+str(dict(**{"mean_"+key:np.nanmean(val) for key, val in train_results.items()},
                                                                              **{"std_balanced":np.nanstd(train_results["test_balanced_accuracy"]),"std":np.nanstd(train_results["test_accuracy"])})),
                                              file=open(checkpoint_path+"/reliable_elast.txt","a")) 
                                        at_least_one = True ; final_results[str(fitted_params)] = train_results 
                                    except Exception as err: 
                                        raise  
                                else:
                                    print("Model not reliable: "+str(fitted_params))
                                    print(str(fitted_params)+","+str(p1)+","+str(p2)+","+str(p3)+","+str(p4)+","+str(np.nanmean(train_results["test_unclassable"]))+","+str(np.nanmean(train_results["test_adjusted_top_2_accuracy"])),
                                          file=open(checkpoint_path+"/unreliable_elast.txt","a"))  
                                    
                        train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                             **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
            
            ext_name = "" if gpca_or_not==0 else "_gpca"
            """
            compressed_pickle(checkpoint_path+"results_ElasticNet"+ext_name,final_results) 
            plot_results_Logistic(final_results,"ElasticNet"+ext_name,ensemble_or_not)    
            """ 
        except Exception as err:
            print(err)
            raise
        
       
    default_params_  =  {"verbose":[0],  # Be quiet!     
                        "l1_ratio":np.arange(0.1,0.95,0.1).tolist(),  # Explore the entire range of ``l1_ratio`` 
                        "scale_l2_by":['group_length'], 
                        "alpha":[0.001,0.01,0.05,0.1],  
                        "fit_intercept":[True],  
                        "max_iter":[1000]} 
    param_grid_sgl = [{"scaler":[scaler],"feature_transformer": [False], 
                       "ensemble_meta_estimator_kwargs":ensemble_kwargs, "ensemble_meta_estimator":[ensemble_estimator],   # watch out for cross-training if adaboost 
                       "feature_transformer_kwargs": [None], "select":[mutual_info_classif], 
                       "n_feats":np.arange(1,4,0.5).tolist(),"tol":[0.001]},
                       {"scaler":["standard", "minmax", "maxabs", "robust"],"feature_transformer": [False], 
                       "ensemble_meta_estimator_kwargs":ensemble_kwargs, "ensemble_meta_estimator":[ensemble_estimator],   # watch out for cross-training if adaboost 
                       "feature_transformer_kwargs": [None], "select":[f_classif], 
                       "n_feats":np.arange(1.5,4,0.5).tolist(),"tol":[0.001]}]
    final_results_sgl = {} ; fitted_params_sgl = [] 
    def single_run_sgl( X00, gpca_or_not, param_grid=param_grid_sgl, y=y, test_cross_val=copy.copy(test_cross_val), params_cross_val=copy.copy(params_cross_val), checkpoint_path_cv=checkpoint_path_cv_,default_params_=default_params_,
                        ensemble_or_not=ensemble_or_not, final_results=final_results_sgl, fitted_params=fitted_params_sgl, state=1000,ensemble_estimator=ensemble_estimator,n_class=n_class, groups_=groups_, groups_pca_=groups_pca_):
                         
        groups__ = groups_ if gpca_or_not==0 else groups_pca_   
        y = np.array(y).astype(float)
            
        try: 
            X00 = X00.dropna(axis=1)
            import shutup
            shutup.please(); y0 = copy.copy(y)
            counter__ = 0  ; test_results = {"params":[],"results":[]}
            counter__ = 0  ; results = {"params":[],"results":[]}
            train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                             **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
            ext_name = "" if gpca_or_not==0 else "_gpca" 
            bad_params = [""] ; good_params = [""] ; final_results = {}
            if os.path.exists(checkpoint_path+"/unreliable_sgl.txt"):
                try:
                    bad_params = pd.read_csv(checkpoint_path+"/unreliable_sgl.txt", sep="}", skiprows=0,
                                             names=["fitted params","performances"]).values[1:,0].flatten()
                except Exception: 
                    pass
            else:
                print(str("fitted params")+",train_accuracy,test_accuracy,test_corrected_accuracy,test_balanced_accuracy",
                      file=open(checkpoint_path+"/unreliable_sgl.txt","a"))
            if os.path.exists(checkpoint_path+"/reliable_sgl.txt"):
                try:
                    good_params = pd.read_csv(checkpoint_path+"/reliable_sgl.txt", sep="},", skiprows=0,
                                         names=["fitted params","perfs"]).values[1:,0].flatten() 
                except Exception: 
                    pass
            else:
                print(str("fitted params")+",train_accuracy,test_accuracy,test_corrected_accuracy,test_balanced_accuracy",
                      file=open(checkpoint_path+"/reliable_sgl.txt","a"))
            for grid in param_grid: 
                for params in traverse(grid): 
                    for est_kwargs in traverse(default_params_):
                        gc.collect() 
                        fitted_params = {'scaler':params['scaler'],'alpha':est_kwargs['alpha'], "l1_ratio":est_kwargs['l1_ratio'], 
                                         "select":str(params['select'])+"_"+str(params['n_feats'])}
                        fitted_params = dict({'scaler':params['scaler'], 
                                              "select":str(params['select']).split()[1]+"_"+str(params['n_feats'])}, 
                                             **{estk:estv for estk, estv in est_kwargs.items() 
                                                          if not estk in ("class_weight","max_iter","tol","solver","penalty")})
                        if ensemble_or_not == "bagging":
                            fitted_params = dict(**fitted_params,**{"n_estimators":params["ensemble_meta_estimator_kwargs"]["n_estimators"]})
                        if gpca_or_not == 1:
                            fitted_params = dict(**fitted_params,**{"group_pca":"yes"}) 
                        else:
                            fitted_params = dict(**fitted_params,**{"group_pca":"no"})
                        if ((not str(fitted_params) in bad_params) and (not str(fitted_params) in good_params)) or refresh:
                            new_params = {key___:val___ for key___, val___ in params.items() if not (key___ in ("select","n_feats"))}
                            base_model = make_afq_classifier_pipeline(**new_params)#,**default_params)   
                            scaler = base_model["scale"]
                            base_model.set_params(**{bmkey:bmval for bmkey,bmval in base_model.get_params().items() 
                                                     if not bmkey in ("scale","impute")})    
                            new_params = {key___:val___ for key___, val___ in params.items() if not (key___ in ("select","n_feats"))}
                            base_model = make_afq_classifier_pipeline(**new_params)# ,**default_params)        
                            k = int(len(X00.iloc[0])/params["n_feats"])
                            cv_results = None ; checked_ = False
                            estimators = [] ; preprocessors = []
                            checkpoint_path_cv = checkpoint_path_cv_ + "elast" ; test_counts = 0
                            for train, test in copy.deepcopy(test_cross_val).split(X00,y0):
                                test_counts += 1
                                X0 = X00.iloc[train] ; y = y0[train]  
                                scaler = scaler.fit(X0,y) 
                                X = scaler.transform(X0) 
                                selected_features = scaler.get_feature_names_out()
                                X = pd.DataFrame(X,columns=selected_features) 
                                selector = SelectKBest(params["select"],  k=k)
                                X = selector.fit_transform(copy.copy(X),y)
                                selected_features = selector.get_feature_names_out()  
                                ggroup = selector.get_support(indices=True) 
                                selected_groups__ = generate_new_groups(groups__, ggroup) 
                                selected_groups__ = [selected_groups__ for _ in range(len(X))]  
                                X = pd.DataFrame(X,columns=selected_features)
                                X_test = selector.transform(scaler.transform(X00.values))
                                new_params = {key___:val___ for key___, val___ in params.items() if not (key___ in ("select","n_feats"))}
                                base_model = make_afq_classifier_pipeline(groups=selected_groups__,**new_params) #,**default_params)   
                                new_params = {"estimate":OneVsRestClassifier(groupyr.LogisticSGL(groups=selected_groups__,**est_kwargs)) } if not ensemble_or_not == "bagging" \
                                              else {"estimate":OneVsRestClassifier(ensemble_estimator(groupyr.LogisticSGL(groups=selected_groups__,**est_kwargs),
                                                                                  groups=selected_groups__, random_state=state, **params["ensemble_meta_estimator_kwargs"]))} 
                                base_model.set_params(**new_params)    
                                model = copy.copy(base_model) ; preprocessors.append([scaler,selector]) 
                                try:
                                    cv_results = cross_validate_checkpoint( model, X, y=y, groups=selected_groups__, scoring=scoring_functions, cv=copy.deepcopy(params_cross_val), n_jobs=-1, 
                                                                            verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=False, 
                                                                            return_estimator=False, error_score='raise', workdir=checkpoint_path_cv, 
                                                                            checkpoint=True, force_refresh=refresh, serialize_cv=False) 
                                except tables.exceptions.HDF5ExtError:
                                    cv_results = cross_validate_checkpoint( model, X, y=y, groups=selected_groups__, scoring=scoring_functions, cv=copy.deepcopy(params_cross_val), n_jobs=-1, 
                                                                            verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=False, 
                                                                            return_estimator=False, error_score='raise', workdir=checkpoint_path_cv, 
                                                                            checkpoint=True, force_refresh=True, serialize_cv=False) 
                                except Exception as err: 
                                    # print(err,file=open('/home/users/d/r/drimez/learning.out',"a"))
                                    try:
                                        cv_results = dict(**{"test_"+skey:[] for skey, _ in scoring_functions.items()})
                                        for train_param, test_param in copy.deepcopy(params_cross_val).split(X,y): 
                                            params_results    = _fit_and_score(estimator=model, X=X, y=y, scorer=scoring_functions, train=train_param, test=test_param,
                                                                        verbose=0, parameters=None, return_train_score=False, return_n_test_samples=False,
                                                                        return_times=False, return_estimator=True, error_score="raise", fit_params=None) 
                                            for tkey in [*scoring_functions.keys()]:
                                                cv_results["test_"+tkey] = np.append(cv_results["test_"+tkey],params_results["test_scores"][tkey])  
                                    except Exception as err:
                                        raise  
                                
                                if cv_results is not None:   
                                    p2 = 0 
                                    p1 = np.nan_to_num(np.nanmean(cv_results["test_accuracy"])) 
                                    if test_counts%n_splits==0 and test_counts>1: 
                                        p1 = np.nanmean(np.append(train_results["train_accuracy"][train_results["train_accuracy"]!=0],
                                                                                cv_results["test_accuracy"]))  
                                        p2 = np.nanmean(np.append(train_results["train_balanced_accuracy"][train_results["train_balanced_accuracy"]!=0],
                                                                                cv_results["test_balanced_accuracy"]))   
                                        if (p1<0.5 and p1>0 and p2<0.4 and p2>0 and (not ensemble_or_not == "bagging"))\
                                           or (p1<0.3 and p1>0 and p2<0.3 and p2>0 and (ensemble_or_not == "bagging")): 
                                            train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                                                 **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
                                            print("Model not reliable: "+str(fitted_params))  
                                            print(str(fitted_params)+","+str(p1)+","+str(0)+","+str(0)+","+str(p2),
                                                  file=open(checkpoint_path+"/unreliable_sgl.txt","a"))
                                            break  
                                    elif (p1<0.4 and p1>0.05 and (not ensemble_or_not == "bagging"))\
                                         or (p1<0.25 and p1>0.05 and (ensemble_or_not == "bagging")):
                                        train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                                             **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
                                        print("Model not reliable: "+str(fitted_params))  
                                        print(str(fitted_params)+","+str(p1)+","+str(0)+","+str(0)+","+str(p2),
                                              file=open(checkpoint_path+"/unreliable_sgl.txt","a"))
                                        break 
                                    test_scores    = _fit_and_score(estimator=model, X=X_test, y=y0, scorer=scoring_functions, train=train, test=test,
                                                                    verbose=10, parameters=None, return_train_score=False, return_n_test_samples=False,
                                                                    return_times=False, return_estimator=True, error_score=0, fit_params=None)
                                    
                                    estimators.append(test_scores["estimator"]) 
                                    for tkey in [*scoring_functions.keys()]:
                                        if np.nan_to_num(np.nanmean(cv_results["test_"+tkey]))!=0: 
                                            train_results["train_"+tkey] = np.append(train_results["train_"+tkey],cv_results["test_"+tkey]) 
                                        if test_scores["test_scores"][tkey]!=0: 
                                            train_results["test_"+tkey] = np.append(train_results["test_"+tkey],test_scores["test_scores"][tkey])   
                                    test_scores = None
                                else:
                                    estimators.append(None)   
                              
                            if len(train_results["train_accuracy"])>1:
                                p1 = np.nan_to_num(np.nanmean(train_results["train_accuracy"]))
                                p2 = np.nan_to_num(np.nanmean(train_results["test_accuracy"])) 
                                p3 = np.nan_to_num(np.nanmean(train_results["test_corrected_accuracy"]))  
                                p5 = np.nan_to_num(np.nanmean(train_results["test_balanced_accuracy"])) 
                                p4 = np.nan_to_num(np.nanmean(train_results["test_balanced_accuracy"])) 
                                feat_names_perm = X00.columns.values.flatten() 
                                if (p2>=0.5 and (not p1>=0.12 +p2) and p5>=0.45 and len(np.unique(y))==3)\
                                    or (p2>=0.5 and (not p1>=0.12 +p2) and p5>=0.55 and len(np.unique(y))==2): 
                                    try: 
                                        results["params"].append(fitted_params)
                                        results["results"].append(train_results)
                                        all_imp = None ; true_imp = None 
                                        final_results[str(fitted_params)] = 0 ; shutup.please()
                                        for train, test in copy.deepcopy(test_cross_val).split(X00.values,y0): 
                                            estimator = None ; X = None ; preprocessor = preprocessors[0] 
                                            X = preprocessor[0].transform(X00)   
                                            selected_features = preprocessor[0].get_feature_names_out()
                                            X = pd.DataFrame(X,columns=selected_features) 
                                            aage = X["age"] if "age" in X.columns.values else None 
                                            X = preprocessor[1].transform(X)  
                                            feat_names = preprocessor[1].get_feature_names_out() 
                                            ggroup = preprocessor[1].get_support(indices=True) 
                                            selected_groups__ = generate_new_groups(groups__, ggroup) 
                                            selected_groups__ = [selected_groups__ for _ in range(len(X))]  
                                            if (not (aage is None)) and (not "age" in selected_features):
                                                selected_features = np.append(selected_features[:-1],aage) 
                                            # X = X.values
                                            if not (estimators[0] is None):
                                                estimator = estimators[0] 
                                            else: 
                                                base_model = make_afq_classifier_pipeline(groups=selected_groups__) # ,**default_params)   
                                                new_params = {"estimate":OneVsRestClassifier(groupyr.LogisticSGL(groups=selected_groups__,**est_kwargs)) } if not ensemble_or_not == "bagging" \
                                                              else {"estimate":OneVsRestClassifier(ensemble_estimator(groupyr.LogisticSGL(groups=selected_groups__,**est_kwargs),
                                                                                                  groups=selected_groups__, random_state=state, **params["ensemble_meta_estimator_kwargs"]))} 
                                                base_model.set_params(**new_params)  
                                                estimator = copy.copy(base_model).fit(X[train],y0[train]) 
                                               
                                            y = y0 
                                            bins = len(y)/np.unique(y,return_counts=True)[1] ; weights = []
                                            for nnnn, nnn_class in enumerate(np.unique(y)):
                                                weights = np.append(weights,[bins[nnnn]/len(bins) for _ in y if _==nnn_class]) 
                                            weights = np.reshape(weights,(len(weights),1))
                                            X = pd.DataFrame(weights*X00.values,columns=X00.columns.values)
                                            estimator = Pipeline(steps=[("scale",preprocessor[0]),
                                                                        ("select",preprocessor[1]),
                                                                        ("estimate",estimator["estimate"])]) 
                                            this_all_imp, this_true_imp = get_importance(estimator["estimate"],estimator,X.values,y0,
                                                                                          train,test,feature_names_in_=feat_names)
                                               
                                            new_this_true_imp = this_true_imp        
                                            this_all_imp = np.array([_.reshape((1,len(_))) for _ in this_all_imp])                                           
                                            new_this_all_imp =  np.zeros((len(this_all_imp),len(this_all_imp[0]),len(feat_names_perm)))                                       
                                            for a_feat_idx, a_feature in enumerate(feat_names_perm):
                                                if a_feature in feat_names_perm:
                                                    new_this_all_imp[...,a_feat_idx] = this_all_imp[...,a_feature==feat_names][...,0] 
                                            
                                            if true_imp is None:
                                                true_imp = new_this_true_imp
                                            else:
                                                true_imp = np.concatenate((true_imp,new_this_true_imp),axis=0)
                                            if all_imp is None:
                                                all_imp = new_this_all_imp
                                            else:
                                                all_imp = np.concatenate((all_imp,new_this_all_imp),axis=0)
                                                
                                            all_imp = np.concatenate((all_imp,new_this_all_imp),axis=0)
                                            estimators.pop(0) ; preprocessors.pop(0)
                                        # print(str(fitted_params)+","+str({"mean_"+key:np.nanmean(val) for key, val in train_results.items()})) 
                                        params_string = ""
                                        for par_name, params_val in fitted_params.items():
                                            params_string += str(par_name) + "=" + str(params_val) + "_"
                                            
                                        estimator = Pipeline(steps=[("scale",base_model["scale"]),
                                                                    ("select",SelectKBest(params["select"],  k=k)),
                                                                    ("estimate",copy.copy(base_model))])  
                                        _, mp, pv = permutation_test_score(estimator, X00, y0, n_jobs=-1, cv=RepeatedStratifiedKFold(n_splits=int(n_splits),  
                                                                                                                                     n_repeats=2, random_state=500),
                                                                            scoring="balanced_accuracy", n_permutations=399) 
                                        
                                        if pv<=0.15 and False:
                                            shap_explanation(X00,y0,copy.deepcopy(test_cross_val),estimator,
                                                             checkpoint_path+"SHAPresults_sgl_"+ext_name+"/%s.txt"%params_string,
                                                             fitted_params)
                                        plot_coef(all_imp,true_imp,mp,pv,feat_names_perm,checkpoint_path+"results_sgl_"+ext_name+"/%s.txt"%params_string,
                                                  "Sparse Group Lasso",fitted_params,train_results)
                                        print(str(fitted_params)+","+str(dict(**{"mean_"+key:np.nanmean(val) for key, val in train_results.items()},
                                                                              **{"std_balanced":np.nanstd(train_results["test_balanced_accuracy"]),"std":np.nanstd(train_results["test_accuracy"])})),
                                              file=open(checkpoint_path+"/reliable_sgl.txt","a")) 
                                        at_least_one = True ; final_results[str(fitted_params)] = train_results 
                                    except Exception as err: 
                                        raise  
                                else:
                                    print("Model not reliable: "+str(fitted_params))
                                    print(str(fitted_params)+","+str(p1)+","+str(p2)+","+str(p3)+","+str(p4),
                                          file=open(checkpoint_path+"/unreliable_sgl.txt","a"))  
                                    
                        train_results = dict(**{"train_"+skey:[] for skey, _ in scoring_functions.items()}, 
                                             **{"test_"+skey:[] for skey, _ in scoring_functions.items()})
            """        
            ext_name = "" if gpca_or_not==0 else "_gpca"
            compressed_pickle(checkpoint_path+"results_SGL"+ext_name,final_results) 
            plot_results_sgl(final_results,"SGL"+ext_name,ensemble_or_not)    
            """ 
        except Exception as err:
            print(err)
            raise
             
    if not (model_ is None): 
        if model_ in ("svm","svm2","svm4","svm5","svm12") or model_.split("_")[0]=="svm": 
            single_run_svm(X1,gpca_or_not)    
        elif model_ in ("elast","elast2"):  
            single_run_elast(X1,gpca_or_not)  
        elif model_=="sgl":
            single_run_sgl(X1,gpca_or_not)
        else:
            single_run_rdf(X1,gpca_or_not)  
    else:
        Parallel(n_jobs=1,verbose=100,pre_dispatch='n_jobs',require="sharedmem")(
                                       delayed(model_type)(X,gpca_or_not) for gpca_or_not, X in enumerate([X1,Xpca]) 
                                       # for model_type in [single_run_sgl]
                                       for model_type in [single_run_svm,single_run_lasso,single_run_ridge,single_run_elast,single_run_sgl]   # single_run_svm
        )

   
# Parallel(n_jobs=1,verbose=100,pre_dispatch='2*n_jobs')( #,require="sharedmem"
#                                  delayed(single_model_comparison)(ensemble_or_not,model=None) 
#                                  for ensemble_or_not in [None,"bagging"]
# )
#%%%%%%%%% 
# compare_models() 
def multi_output(model, data="all", type_="det", scaler=None):
 
    x_path, pca_path = "/auto/home/users/d/r/drimez/data.csv", "/auto/home/users/d/r/drimez/data_all.csv"
    if type_=="prob":
        x_path = "/auto/home/users/d/r/drimez/data_prob.csv"
        pca_path = "/auto/home/users/d/r/drimez/data_all_prob.csv"
     
    to_drop = [ True,  True,  True,  True,  True,  True,  True,  True, False,
                True,  True,  True,  True,  True,  True,  True,  True, False,
                True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True, False,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True,  True,  True,  True,
               False,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True, False,  True,  True,  True,
                True,  True,  True, False]
    features_ = ["age","cVemp_R","cVemp_L","oVemp_R","oVemp_L","VHIT_ant_R","VHIT_ant_L","VHIT_lat_R",
                 "VHIT_lat_L","VHIT_post_R","VHIT_post_L","VNG_cal_R","VNG_cal_L","VNG_rot","Posturo"]
                  
    targets = pd.read_csv("/auto/home/users/d/r/drimez/targets.txt")
    demo_y = targets["Unnamed: 0"].values.flatten()
    targets = pd.DataFrame(targets.drop("Unnamed: 0",axis=1).values[np.argsort(demo_y)],index=np.sort(demo_y)).values
    
    exclude = ["H_4","V_12","C_4","C_10","V_31"] 
    mask = np.array([True if _ in patient_list and (not (_ in exclude)) else False for _ in demo_y])
    # to_drop = np.array(to_drop)[mask]
    targets = targets[mask]  
    X1, Xpca, pcaX, morphom, groups_, groups_pca_, groups_group_pca_, y = load_tract_data(x_path,pca_path,type_=type_,pca_type=data,age=targets[:,0]) 
    
    """
    if data == "pca_without_age":
        Xpca = Xpca.drop("age",axis=1)
    """
    pcaX_=pcaX
     
    y_temp = y  
    
    targets_ = np.concatenate((y_temp.reshape((len(y_temp),1)),targets[:,1:]-1),axis=-1)
    features_ = np.concatenate((["vertigo"],features_[1:]),axis=-1)
    """
    targets_ = targets_.T[1:].T
    features_ = features_[1:] 
    targets_ = targets_.T[1:].T
    features_ = features_[1:] 
    targets_ = targets_.T[1:].T
    features_ = features_[1:]
    targets_ = targets_.T[1:].T
    features_ = features_[1:]
    targets_ = targets_.T[1:].T
    features_ = features_[1:]
    targets_ = targets_.T[1:].T
    features_ = features_[1:]
    targets_ = targets_.T[1:].T
    features_ = features_[1:]
    targets_ = targets_.T[1:].T
    features_ = features_[1:]
    """
    
    def select_corr(X, target, name, pvals, age=targets[:,0]): 
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X) 
        selected_features = scaler.get_feature_names_out()
        X = pd.DataFrame(X,columns=selected_features)  
        new_X = pd.concat([pd.DataFrame(X.values,columns=X.columns.values),
                           pd.DataFrame(np.array([target,age]).T,columns=[name,"age"])],axis=1) # ["target"]
        new_corr = abs(new_X.corr())
        age_corr = pd.DataFrame([new_corr.values[-1,:-2]],columns=new_corr.columns.values[:-2])
        target_corr = pd.DataFrame([new_corr.values[-2,:-2]],columns=new_corr.columns.values[:-2]) 
        new_corr = pd.DataFrame(new_corr.values[:-2,:-2],columns=new_corr.columns.values[:-2],index=new_corr.index.values[:-2])
        # new_cols = np.array([_.split(",")[1] for _ in new_corr.columns.values])
        # new_corr = pd.DataFrame(new_corr.values,columns=new_cols) 
        # new_corr = pd.DataFrame(new_corr,columns=new_cols,index=new_cols)  
        
        above_th = abs(new_corr.values)>=0.7 ; drop = [] ; columns = X.columns.values
        for n_lign, (lign, name) in enumerate(zip(new_corr.values[1:][::-1],new_corr.columns.values[1:][::-1])):
            if above_th[n_lign,:n_lign].sum()>0 and (not n_lign in drop): 
                indxs = np.arange(n_lign+1)[above_th[n_lign,:n_lign+1]]
                dropped = np.argsort(abs(new_corr.values[n_lign][indxs]))[:-1] # keeps only the most correlated
                above_th[:,dropped] = 0
                drop = np.unique(np.append(drop,dropped)) 
                for col in np.array([columns[dropped]]).flatten():
                    try:
                        X = X.drop(col,axis=1)
                        age_corr.drop(col,axis=1)
                    except Exception as err:
                        pass
                
        # drop = new_corr.columns.values[np.array(drop).astype(int)].tolist()  
        # new_corr = new_corr.drop(drop,axis=1).drop(drop,axis=0)  
        # kept = new_corr.drop(drop,axis=1).columns.values   
        # below_th = np.array(age_corr.values.flatten()>0.5)
        # new_corr = new_corr.drop(drop,axis=1).drop(drop,axis=0)
        # drop = np.array(new_corr.columns.values[below_th])
        # kept = target_corr.drop(drop,axis=1).columns.values[:-1]  
        # kept = [_.split(",")[1] for _ in kept]
        # new_features = [True if (_.split(",")[1] in kept) else False for _ in X.columns.values]
        new_features = X.drop([_ for _ in X.columns.values if abs(age_corr[_].values)>0.5],axis=1)
        return new_features
             
 
    checkpoint_path_root = "/auto/home/users/d/r/drimez/Classify_tract" 
    def fit_a_target(target, ff,demo_y=demo_y,X1=X1,X1_=X1,Xpca_=Xpca,data=data,pcaX_=pcaX_,pcaX=pcaX_,type_=type_,pca_type=data,morphom=morphom,model=model, scaler=scaler): 
        for ensemble_or_not in [None,"bagging"]: # None,"bagging"
            for data_ in [None]: # None,"age"
                 
                f = data_+"_"+ff+ "_" + str(data) if data_=="age" else "_" + ff + "_" + str(data)
                f += "_"+model
                checkpoint_path = checkpoint_path_root + "_bagging_" + type_ + "_final/%s/"%f if ensemble_or_not=="bagging" \
                                  else  checkpoint_path_root + "_" + type_ + "_final/%s/"%f   
                if not os.path.isdir(checkpoint_path):
                    os.makedirs(checkpoint_path)
                checkpoint_path_cv_ = checkpoint_path + "/checkpoints"
                 
                if ff!="vertigo":
                    # target[target==1] = 0
                    target[target>1] = 1
                    
                # target_mask = np.logical_and(np.array(target).flatten()!=-1,to_drop).tolist()
                target_mask = (np.array(target).flatten()!=-1).tolist() 
                target_mask = np.array(target_mask)
                y = target.flatten() 
                y = np.array(y) 
                demo_y = list(demo_y) 
                demo_y = np.array(demo_y) 
                 
                p_names = np.array(demo_y)  
                    
                selector = SelectKBest(k="all")
                selector.fit(copy.copy(pcaX_),y)
                selected_features = selector.get_feature_names_out()  
                pvals = pd.DataFrame(selector.pvalues_.reshape(-1,1),index=selected_features,columns=["p_value"])
                pvals.sort_values(by="p_value").to_csv("/auto/home/users/d/r/drimez/pvalues_"+type_+"_"+pca_type.replace("all","pca").replace("select","pca")+".txt")
                gc.collect()
                print("pca before select: "+str(len(pcaX_.values[0])))  
                new_pca = select_corr(pcaX_, target, ff, pvals) 
                
                if ff=="vertigo":
                    fig, ax = plt.subplots(1,1,figsize=(30,15))
                    order = np.argsort(pcaX_.columns)
                    order_ind = np.argsort(pcaX_.index)
                    corr = pd.concat([pd.DataFrame(pcaX_.values[order_ind].T[order].T,columns=pcaX_.columns,index=pcaX_.index),
                                      pd.DataFrame(targets_,columns=features_)],axis=1)    
                    corr = corr.corr()
                    cols = corr.columns.values
                    index = corr.index.values 
                    corr = pd.DataFrame(corr.values,index=index,columns=cols)
                    ax = sns.heatmap(abs(corr),annot=False,ax=ax)
                    fig.savefig("/auto/home/users/d/r/drimez/Corrmaps/corr_pca_%s_final.png"%(str(data_)))  
                    fig, ax = plt.subplots(1,1,figsize=(30,15)) 
                    order = np.argsort(new_pca.columns)
                    order_ind = np.argsort(new_pca.index)
                    corr = pd.concat([pd.DataFrame(new_pca.values[order_ind].T[order].T,columns=new_pca.columns,index=new_pca.index),
                                      pd.DataFrame(targets_,columns=features_)],axis=1)   
                    corr = corr.corr()
                    cols = corr.columns.values
                    index = corr.index.values 
                    corr = pd.DataFrame(corr.values,index=index,columns=cols)
                    ax = sns.heatmap(abs(corr),annot=False,ax=ax)
                    fig.savefig("/auto/home/users/d/r/drimez/Corrmaps/corr_pca_select_%s_final.png"%(str(data_))) 
             
                pcaX_=new_pca
                print("pca after select: "+str(len(pcaX_.values[0]))) 
                
                gc.collect()
                selector = SelectKBest(k="all")
                selector.fit(copy.copy(X1_),y)
                selected_features = selector.get_feature_names_out()  
                pvals = pd.DataFrame(selector.pvalues_.reshape(-1,1),index=selected_features,columns=["p_value"])
                pvals.sort_values(by="p_value").to_csv("/auto/home/users/d/r/drimez/pvalues_"+type_+"_"+pca_type.replace("pca","all").replace("select","all")+".txt")
                gc.collect()
                
                print("all before select: "+str(len(X1_.values[0]))) 
                X1_ = select_corr(X1_, target, ff, pvals) 
                
                if ff=="vertigo": 
                    fig, ax = plt.subplots(1,1,figsize=(30,15)) 
                    order = np.argsort(X1.columns)
                    order_ind = np.argsort(X1.index)
                    corr = pd.concat([pd.DataFrame(X1.values[order_ind].T[order].T,columns=X1.columns,index=X1.index),
                                      pd.DataFrame(targets_,columns=features_)],axis=1)  
                    corr = corr.corr()
                    cols = corr.columns.values
                    index = corr.index.values 
                    corr = pd.DataFrame(corr.values,index=index,columns=cols)
                    ax = sns.heatmap(abs(corr),annot=False,ax=ax)
                    fig.savefig("/auto/home/users/d/r/drimez/Corrmaps/corr_all_%s_final.png"%(str(data_))) 
                    fig, ax = plt.subplots(1,1,figsize=(30,15)) 
                    order = np.argsort(X1_.columns)
                    order_ind = np.argsort(X1_.index)
                    corr = pd.concat([pd.DataFrame(X1_.values[order_ind].T[order].T,columns=X1_.columns,index=X1_.index),
                                      pd.DataFrame(targets_,columns=features_)],axis=1) 
                    corr = corr.corr()
                    cols = corr.columns.values
                    index = corr.index.values 
                    corr = pd.DataFrame(corr.values,index=index,columns=cols)
                    ax = sns.heatmap(abs(corr),annot=False,ax=ax)
                    fig.savefig("/auto/home/users/d/r/drimez/Corrmaps/corr_all_select_%s_final.png"%(str(data_)))
                 
                print("all after select: "+str(len(X1_.values[0]))) 
                
                gc.collect()
                print("restricted before select: "+str(len(Xpca_.values[0]))) 
                Xpca_ = select_corr(Xpca_, target, ff, pvals)
                
                if ff=="vertigo":
                    fig, ax = plt.subplots(1,1,figsize=(30,15)) 
                    order = np.argsort(Xpca_.columns)
                    order_ind = np.argsort(Xpca_.index)
                    corr = pd.concat([pd.DataFrame(Xpca_.values[order_ind].T[order].T,columns=Xpca_.columns,index=Xpca_.index),
                                      pd.DataFrame(targets_,columns=features_)],axis=1)
                    corr = corr.corr()
                    cols = corr.columns.values
                    index = corr.index.values 
                    corr = pd.DataFrame(corr.values,index=index,columns=cols)
                    ax = sns.heatmap(abs(corr),annot=False,ax=ax)
                    fig.savefig("/auto/home/users/d/r/drimez/Corrmaps/corr_select_%s_final.png"%(str(data_)))
                
                print("restricted after select: "+str(len(Xpca_.values[0]))) 
                
                gc.collect()
                print("Morphometry before select: "+str(len(morphom.values[0]))) 
                morphom = select_corr(morphom, target, ff, pvals)
                """
                fig, ax = plt.subplots(1,1,figsize=(30,15))
                corr = pd.concat([Xpca_,pd.DataFrame(targets_,columns=features_)],axis=1)
                corr = corr.corr()
                cols = corr.columns.values
                index = corr.index.values 
                ax = sns.heatmap(abs(corr),annot=False,ax=ax)
                fig.savefig("/auto/home/users/d/r/drimez/Corrmaps/corr_select_%s.png"%(str(data_)))
                """
                print("Morphometry after select: "+str(len(morphom.values[0]))) 
                
                if data_=="age":  
                    targets_drop_ = [True if not (iiuuu in (17,29)) else False for iiuuu in range(len(targets[:-1]))]   
                    X1_ = np.concatenate((X1.values[target_mask],targets[target_mask,0].reshape((len(y[target_mask]),1))),axis=-1) 
                    X1_ = pd.DataFrame(X1_,columns=list(X1.columns.values)+list(["age"]))
                    pcaX_ = pd.DataFrame(np.concatenate((pcaX_.values[target_mask],targets[target_mask,0].reshape((len(y[target_mask]),1))),axis=-1),
                                         columns=list(pcaX_.columns.values)+list(["age"])) 
                    Xpca_ = pd.DataFrame(np.concatenate((Xpca_.values[target_mask],targets[target_mask,0].reshape((len(y[target_mask]),1))),axis=-1),
                                         columns=list(Xpca_.columns.values)+list(["age"]))
                    morphom = pd.DataFrame(np.concatenate((morphom.values[target_mask],targets[target_mask,0].reshape((len(y[target_mask]),1))),axis=-1),
                                         columns=list(morphom.columns.values)+list(["age"])) 
                else:
                    Xpca_ = pd.DataFrame(Xpca_,columns=list(Xpca.columns.values))
                    X1_ = pd.DataFrame(X1_,columns=list(X1.columns.values))
                    pcaX_ = pd.DataFrame(pcaX_,columns=list(pcaX_.columns.values))
                    
                Xpca_ = Xpca_.iloc[target_mask] ; X1_ = X1_.iloc[target_mask] ; pcaX_ = pcaX_.iloc[target_mask]  
                 
                model_ = None if model == "all" else model
                print(np.unique(target.flatten()[:-1][target.flatten()[:-1]!=-1]))#.astype(int)))
                print(np.unique(target.flatten()[:-1][target.flatten()[:-1]!=-1].astype(int),return_counts=True)[1])
                
                if len(np.unique(target.flatten()[:-1][target.flatten()[:-1]!=-1].astype(int),return_counts=True)[1])==1:
                    print("All patients have the same result for test " + str(ff))
                else: 
                    print("Processing")
                    if data=="all": 
                        groups_names = [] ; corresp = {}
                        for icol,col in enumerate(X1_.columns.values):
                            if not (col.split(",")[0] in groups_names):
                                groups_names.append(col.split(",")[0])
                                corresp[col.split(",")[0]] = [icol] 
                            else:
                                corresp[col.split(",")[0]].append(icol) 
                                
                        groups_ = [_ for key, _ in corresp.items()]    
                        groups_ = list(np.array(ggg).flatten() for ggg in groups_ if len(ggg)>=1)  
                        
                        single_model_comparison(ensemble_or_not,model=model,y=y[target_mask].astype(int), X1=X1_, p_names=p_names, gpca_or_not=0, groups_pca_=groups_,
                                                Xpca=Xpca_, checkpoint_path=checkpoint_path,checkpoint_path_cv_=checkpoint_path_cv_, pcaX=pcaX_,groups_=groups_, scaler=scaler) 
                    elif len(data.split("pca"))==2:
                        groups_names = [] ; corresp = {}
                        for icol,col in enumerate(pcaX.columns.values):
                            if not ("_".join(col.split("_")[:-2]) in groups_names):
                                groups_names.append("_".join(col.split("_")[:-2]))
                                corresp["_".join(col.split("_")[:-2])] = [icol] 
                            else:
                                corresp["_".join(col.split("_")[:-2])].append(icol) 
                            
                        groups_ = [_ for key, _ in corresp.items()]    
                        groups_group_pca_ = list(np.array(ggg).flatten() for ggg in groups_ if len(ggg)>=1)
        
                        single_model_comparison(ensemble_or_not,model=model,y=y[target_mask].astype(int), X1=pcaX_, p_names=p_names, gpca_or_not=1,groups_=groups_group_pca_,
                                                Xpca=Xpca_, checkpoint_path=checkpoint_path,checkpoint_path_cv_=checkpoint_path_cv_, pcaX=pcaX_, groups_pca_=groups_group_pca_, scaler=scaler)  
                    elif data=="morphom":
                        groups_names = [] ; corresp = {}
                        for icol,col in enumerate(morphom.columns.values):
                            if not (col.split(",")[0] in groups_names):
                                groups_names.append(col.split(",")[0])
                                corresp[col.split(",")[0]] = [icol] 
                            else:
                                corresp[col.split(",")[0]].append(icol)
                            
                        groups_ = [_ for key, _ in corresp.items()]    
                        groups_group_pca_ = list(np.array(ggg).flatten() for ggg in groups_ if len(ggg)>=1)
        
                        single_model_comparison(ensemble_or_not,model=model,y=y[target_mask].astype(int), X1=morphom, p_names=p_names, gpca_or_not=0,groups_=groups_group_pca_,
                                                Xpca=morphom, checkpoint_path=checkpoint_path,checkpoint_path_cv_=checkpoint_path_cv_, pcaX=morphom, groups_pca_=groups_group_pca_, scaler=scaler)  
                    else:
                        groups_names = [] ; corresp = {}
                        for icol,col in enumerate(Xpca_.columns.values):
                            if not (col.split(",")[0] in groups_names):
                                groups_names.append(col.split(",")[0])
                                corresp[col.split(",")[0]] = [icol] 
                            else:
                                corresp[col.split(",")[0]].append(icol) 
                                
                        groups_ = [_ for key, _ in corresp.items()]    
                        groups_pca_ = list(np.array(ggg).flatten() for ggg in groups_ if len(ggg)>=1)
                        
                        single_model_comparison(ensemble_or_not,model=model,y=y[target_mask].astype(int), X1=Xpca_, p_names=p_names, gpca_or_not=0, groups_pca_=groups_pca_,
                                                Xpca=Xpca_, checkpoint_path=checkpoint_path,checkpoint_path_cv_=checkpoint_path_cv_, pcaX=pcaX_, groups_=groups_pca_, scaler=scaler) 
            
    Parallel(n_jobs=-1,verbose=10,pre_dispatch='n_jobs',require="sharedmem")(
                                       delayed(fit_a_target)(target, ff,data=data) for target, ff in zip(targets_.T,features_)  
        )
      
if __name__ == "__main__": 
  
    if len(sys.argv)>1: 
        args = [str(arg) for arg in sys.argv[1:]] 
        multi_output(model=args[0],data=args[1], type_=args[2], scaler=args[3])
        
    """       
    for mod_mod in ["svm"]:  
        multi_output(model=mod_mod)
    """














































