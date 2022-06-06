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

sys.setrecursionlimit(2097152)     
threading.stack_size(134217728)  
sys.settrace 
my_f_path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/"

my_f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset_new/PROJECT/"  
f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset_new/PROJECT/"  
my_f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/elikopy_subset_new2/PROJECT/"  
f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/elikopy_subset_new2/PROJECT/"  

patient_list = ['H_0','H_1','H_2','H_3','H_4','V_100','V_101','V_102','V_103']  
patient_list = ['H_0']  
patient_list = ['C_1']  
  
def compressed_pickle(title, data):
   with bz2.BZ2File(title + '.pbz2', 'w') as f: 
       cPickle.dump(data, f)
       f.close()
       
def decompress_pickle(filename):
   data = bz2.BZ2File(filename+".pbz2", 'rb')
   data = cPickle.load(data)
   return data
  
def my_to_csv(csv_path,obj_to_write,columns=None):
   
    csv_folder = "/".join(csv_path.split("/")[:-1])
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
   
    with open(csv_path,'w') as writer: 
        ended = False ; rows = None 
        if isinstance(obj_to_write,dict):
            cols = [obj_to_write[k] for k in [*obj_to_write.keys()]] 
            columns = [*obj_to_write.keys()] if columns is None else columns
            for key in columns:
                writer.write(key+",")
            writer.write("\n")
            print(np.shape(cols)) 
        else:
            if not (columns is None):
                for key in columns:
                    writer.write(str(key)+",")
                writer.write("\n")
            cols = obj_to_write.T
        
        try:
            if isinstance(cols[0],list) or isinstance(cols[0],np.ndarray):
                for lign in range(len(cols[0])):
                    for icol, col in enumerate(cols): 
                        if icol!=0:
                            writer.write(",")
                        writer.write(str(col[lign]))
                    writer.write("\n")    
                writer.close()
            else: 
                for icol, col in enumerate(cols): 
                    if icol!=0:
                        writer.write(",")
                    writer.write(str(col))   
                writer.close()
        except Exception:
            print(obj_to_write)

def dipy_tracto(p,n_iterations=15,model=None,f_path=None,density_map_files=None,fod=True,single_bdl=None,filtering=False,postrack=False,resp=None):

    if isinstance(fod,str):
        fod = fod=="True"
        
    if isinstance(resp,str):
        resp = resp=="True"
        
    if isinstance(n_iterations,str):
        n_iterations = int(n_iterations)
        
    if model=="_prob" and single_bdl is None:
        n_iterations = 1
    
    # loads dmri and white matter mask
    wm_path = f_path + "wm_masks/%s/wm.seg.nii.gz"%(p)
    T1_path = f_path + "wm_masks/%s/T1.nii.gz"%(p)
    bck_path = f_path + "subjects/%s/tracking/preproc/%s_bck_mask.nii.gz"%(p,p)
    csf_pve = f_path + "subjects/" + p + "/tracking/preproc/" + p + "_csf_pve.nii.gz"
    gm_pve = f_path + "subjects/" + p + "/tracking/preproc/" + p + "_gm_pve.nii.gz"
    wm_pve = f_path + "subjects/" + p + "/tracking/preproc/" + p + "_wm_pve.nii.gz"
    wm_dil = f_path + "subjects/" + p + "/tracking/preproc/" + p + "_wm_pve_dil.nii.gz" 
    wm_dil_th = f_path + "subjects/" + p + "/tracking/preproc/" + p + "_wm_pve_dil_th.nii.gz" 
    gm_pve_sub = f_path + "subjects/" + p + "/tracking/preproc/" + p + "_gm_pve_sub.nii.gz"
    csf_path_2 = f_path + "subjects/%s/tracking/preproc/%s_csf_mask.nii.gz"%(p,p) 
    gm_path_2 = f_path + "subjects/%s/tracking/preproc/%s_gm_mask.nii.gz"%(p,p) 
    seg_path = f_path + "subjects/%s/masks_vertige/%s_segmentation.nii.gz"%(p,p)
    dmri_root = f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc"%(p,p) 
    dire = f_path + "subjects/%s/tracking/preproc/"%p
    ref_t1 = f_path + "subjects/" + p + "/T1/" + p + "_T1_corr_projected.nii.gz" 
    bvals, bvecs = read_bvals_bvecs(dmri_root+".bval", dmri_root+".bvec")
    gtab = gradient_table(bvals, bvecs)
    dmri_preproc = f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(p,p)
      
    data, ref_affine, ref_image = load_nifti(dmri_preproc, return_img=True)
     
    registered_data_wm, _ = load_nifti(dire+"/"+p+"_wm_dil.nii.gz")  
    
    output_name = f_path + "subjects/" + p + "/tracking/" + p + "_dipy" + model   
    if not postrack:
        if not os.path.isdir(f_path+"subjects/%s/tracking/qcontrol"%p):
            os.mkdir(f_path+"subjects/%s/tracking/qcontrol"%p)
            
        response_wm_mask, _ = load_nifti(dire+"/"+p+"_wm_new.nii.gz")
        
        if resp==True and False: 
            response, ratio = response_from_mask_ssst(gtab, data, response_wm_mask)
             
            print("WM mask response: " + str(response))
            print("ratio: " + str(ratio))
            
            csd_model = ConstrainedSphericalDeconvModel(gtab, response, convergence=1000000)
            csd_fit = csd_model.fit(data, mask=registered_data_wm) 
            fod = csd_fit.odf(default_sphere)
            pmf = fod.clip(min=0) 
            save_nifti(f_path+"subjects/%s/tracking/qcontrol/csd_odfs_wm.nii.gz"%p,pmf,ref_affine)
            """
            scene = window.Scene()
            fodf_spheres = actor.odf_slicer(pmf, sphere=default_sphere, scale=0.9, norm=False, colormap='plasma') 
            scene.add(fodf_spheres) 
            print('Saving illustration as csd_odfs.png')
            window.record(scene, out_path=f_path+"subjects/%s/tracking/qcontrol/csd_odfs_wm.png"%p, size=(600, 600))
            """
            response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
            
            print("WM mask response: " + str(response))
            print("ratio: " + str(ratio)) 
            
            csd_model = ConstrainedSphericalDeconvModel(gtab, response, convergence=1000000)
            csd_fit = csd_model.fit(data, mask=registered_data_wm) 
            fod = csd_fit.odf(default_sphere)
            pmf = fod.clip(min=0)
            save_nifti(f_path+"subjects/%s/tracking/qcontrol/csd_odfs_auto.nii.gz"%p,pmf,ref_affine)
            """
            scene.clear() 
            scene = window.Scene()
            fodf_spheres = actor.odf_slicer(pmf, sphere=default_sphere, scale=0.9, norm=False, colormap='plasma') 
            scene.add(fodf_spheres) 
            print('Saving illustration as csd_odfs.png')
            window.record(scene, out_path=f_path+"subjects/%s/tracking/qcontrol/csd_odfs_auto.png"%p, size=(600, 600))
            scene.clear() 
            """
        # sys.setrecursionlimit(10000)
        response = None ; csd_model = None ; pmf = None
        if model == "" or True: 
            if not os.path.exists(f_path+"subjects/%s/tracking/FOD/%s_response%s.pbz2"%(p,p,model)) or True:
                print("Launching response estimation")
                fa_thr = 0.7 if model=="" else 0.8
                response_mask = mask_for_response_ssst(gtab, data, roi_radii=response_wm_mask.shape[0], fa_thr=0.8)
                response_mask[response_wm_mask==0] = 0
                response, ratio = response_from_mask_ssst(gtab, data, response_mask)
                print("WM mask response: " + str(response))
                print("ratio: " + str(ratio))
                compressed_pickle(f_path+"subjects/%s/tracking/FOD/%s_response%s"%(p,p,model),response)
            else:
                response = decompress_pickle(f_path+"subjects/%s/tracking/FOD/%s_response%s"%(p,p,model))
               
            if (not os.path.exists(f_path+"subjects/%s/tracking/FOD/%s_csd_model.pbz2"%(p,p))) or True:
                print("Launching CSD estimation")
                csd_model = ConstrainedSphericalDeconvModel(gtab, response, convergence=1000000)
                csd_fit = csd_model.fit(data, mask=registered_data_wm)
                if not os.path.exists(f_path+"subjects/%s/tracking/FOD/"%p):
                    os.mkdir(f_path+"subjects/%s/tracking/FOD/"%p)
                compressed_pickle(f_path+"subjects/%s/tracking/FOD/%s_csd_model%s"%(p,p,model),csd_model)
            else: 
                csd_model = decompress_pickle(f_path+"subjects/%s/tracking/FOD/%s_csd_model"%(p,p)) 
                csd_fit = csd_model.fit(data, mask=registered_data_wm)
             
            if model=="_prob" and (not os.path.exists(f_path+"subjects/%s/tracking/FOD/%s_fod%s"%(p,p,model))):
                print("Launching FOD fit") 
                fod = csd_fit.odf(default_sphere)
                pmf = fod.clip(min=0) 
                compressed_pickle(f_path+"subjects/%s/tracking/FOD/%s_fod%s"%(p,p,model),pmf) 
            elif model=="_prob":
                pmf = decompress_pickle(f_path+"subjects/%s/tracking/FOD/%s_fod%s"%(p,p,model)) 
          
        else:    
            if os.path.exists(f_path+"subjects/%s/tracking/FOD/%s_mrtrix_fod.nii.gz"%(p,p)):
                pmf, _ = load_nifti(f_path+"subjects/%s/tracking/FOD/%s_mrtrix_fod.nii.gz"%(p,p)) 
            else:
                data = "data_1" if p[0]=="H" else "data_2"
                rep = "dwi2response  tournier /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/dMRI/preproc/{p}_dmri_preproc.nii.gz /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/tracking/FOD/{p}_mrtrix_response.txt -mask /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/tracking/preproc/{p}_wm_new.nii.gz -fslgrad /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/{data}/{p}.bvec /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/{data}/{p}.bval -lmax 12 -force".format(p=p,data=data)
                fod = "dwi2fod csd /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/dMRI/preproc/{p}_dmri_preproc.nii.gz /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/tracking/FOD/{p}_mrtrix_response.txt /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/tracking/FOD/{p}_mrtrix_fod.nii.gz -lmax 12 -mask /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/tracking/preproc/{p}_wm_new.nii.gz -fslgrad /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/{data}/{p}.bvec /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/{data}/{p}.bval -directions /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/H_0/tracking/FOD/sphere.txt -force".format(p=p,data=data)
                
                cmd = rep + " ; " + fod
                cmd.split()
                process = subprocess.Popen(cmd, universal_newlines=True, shell=True, stdout=sys.stdout,
                                                       stderr=subprocess.STDOUT)
                # wait until finish
                out, error = process.communicate() 
                pmf, _ = load_nifti(f_path+"subjects/%s/tracking/FOD/%s_mrtrix_fod.nii.gz"%(p,p)) 
            
        stopping_criterion = BinaryStoppingCriterion(registered_data_wm) 
        # gfa = csd_model.gfa
        # stopping_criterion = ThresholdStoppingCriterion(gfa, .1)
        
        if resp:  
            """
            scene = window.Scene()
            fodf_spheres = actor.odf_slicer(pmf, sphere=default_sphere, scale=0.9, norm=False, colormap='plasma') 
            scene.add(fodf_spheres) 
            print('Saving illustration as csd_odfs.png')
            window.record(scene, out_path=f_path+"subjects/%s/tracking/qcontrol/csd_odfs_wmfa.png"%p, size=(600, 600))
            
            response_rec = recursive_response(  gtab, data, mask= response_wm_mask==1, 
                                                sh_order=8, peak_thr=0.01, init_fa=0.08,
                                                init_trace=0.0021, iter=8, convergence=0.001,
                                                parallel=True , num_processes=-1 )
            response_rec2 = response_rec.on_sphere(default_sphere)
            print("WM mask response: " + str(response_rec2))
            # print("ratio: " + str(response_rec[0][0]/response_rec[0][1]))
            
            csd_model_rec = ConstrainedSphericalDeconvModel(gtab, response_rec, convergence=10000)
            csd_fit_rec = csd_model_rec.fit(data, mask=response_mask) 
            fod_rec = csd_fit_rec.odf(default_sphere)
            pmf_rec = fod_rec.clip(min=0)
            save_nifti(f_path+"subjects/%s/tracking/qcontrol/csd_odfs_rec.nii.gz"%p,pmf,ref_affine)
            
            
            # scene.rm(response_actor)
            scene.clear()
            scene = window.Scene()
            fodf_spheres = actor.odf_slicer(pmf_rec, sphere=default_sphere, scale=0.9, norm=False, colormap='plasma') 
            scene.add(fodf_spheres) 
            print('Saving illustration as csd_odfs.png')
            window.record(scene, out_path=f_path+"subjects/%s/tracking/qcontrol/csd_odfs_rec.png"%p, size=(600, 600))
            """
            resp=False
            
            """
            scene.clear()
            scene = window.Scene()
            fodf_peaks = actor.peak_slicer(peak_model.peak_dirs, peak_model.peak_values,colors=None)
            scene.add(fodf_peaks) 
            print('Saving illustration as csd_peaks.png')
            window.record(scene, out_path=f_path+"subjects/%s/tracking/qcontrol/csd_peaks.png"%p, size=(600, 600))
            """
        if not single_bdl is None:
            density_map_files = [density_map_files]
        else:
            density_path = "/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/"%p
            with os.scandir(density_path) as my_it:
                for an_entry in my_it:
                    density_map_files.append(density_path + an_entry.name)
             
        def single_iter(n_iter,single_bdl=None,pmf=pmf,registered_data_wm=registered_data_wm,stopping_criterion=stopping_criterion, csd_model=csd_model,  
                        ref_image=ref_image,output_name=output_name,model=model,f_path=f_path,density_map_files=density_map_files,filtering=filtering): 
            
            peak_model = None
            if model == "_prob": 
                rel_th = np.random.rand(1)*0.1 + 0.8 # between 0.75 and 0.9
                max_angle = np.random.rand(1)*3 + 12
                peak_model = ProbabilisticDirectionGetter.from_pmf( pmf, max_angle=max_angle, sphere=default_sphere, pmf_threshold=0.2,
                                                                    relative_peak_threshold = rel_th, min_separation_angle=10 )
            elif model == "_sparse":
                sf_model = sfm.SparseFascicleModel(gtab, sphere=peaks.default_sphere, l1_ratio=0.3, alpha=0.001, response=response[0])
                sf_fit = sf_model.fit(data)
                sf_odf = sf_fit.odf(peaks.default_sphere)
                peak_model = peaks.peaks_from_model(model=sf_model, data=data, sphere=peaks.default_sphere,
                                                    relative_peak_threshold=.8, min_separation_angle=30, 
                                                    mask=registered_data_wm, parallel=True)
            elif model == "_boot":
                peak_model = BootDirectionGetter.from_data(data, csd_model, max_angle=25, sphere=small_sphere)  
            else:
                rel_th = np.random.rand(1)*0.25 + 0.65 # between 0.7 and 0.9
                peak_model = peaks.peaks_from_model(model=csd_model, data=data, sphere=peaks.default_sphere,
                                                    relative_peak_threshold=rel_th, min_separation_angle=15, 
                                                    mask=registered_data_wm, parallel=True, normalize_peaks=True,
                                                    num_processes=1)                           
            
            affine = np.eye(4)
            
            # seeds from dilated wm
            seeding_mask = copy.copy(registered_data_wm)
            seeds_count = 1 #  seeding_mask.sum()/4
            if not (single_bdl is None):
                density_map_img, _ = load_nifti(density_map_files[0])
                if len(np.shape(density_map_img))==4:
                    density_map_img = density_map_img[...,0]
                seeding_mask[density_map_img<=0.01] = 0
                seeds_count = 8 # 5*np.sum(seeding_mask>0.01)
                
            seeds = utils.random_seeds_from_mask( seeding_mask, affine, seed_count_per_voxel=True,  
                                                  seeds_count = int(seeds_count) ) #  
                                                                    
            if filtering: 
                pve_path = f_path + "subjects/" + p + "/tracking/preproc/" + p 
                pve_csf_data, _ = load_nifti(pve_path + "_csf_pve.nii.gz")
                pve_gm_data, _ = load_nifti(pve_path + "_gm_pve.nii.gz")
                pve_wm_data, _, voxel_size = load_nifti(pve_path + "_wm_pve.nii.gz", return_voxsize=True)
                voxel_size = np.average(voxel_size[1:4]) 
                stopping_criterion =  CmcStoppingCriterion.from_pve(pve_wm_data, pve_gm_data, pve_csf_data,
                                                                        step_size=0.5, average_voxel_size=voxel_size)  
                model = model + "_PFT"
                pve_path = f_path + "subjects/" + p + "/tracking/preproc/" + p 
                pve_csf_data, _ = load_nifti(pve_path + "_csf_pve.nii.gz")
                pve_gm_data, _ = load_nifti(pve_path + "_gm_pve.nii.gz")
                pve_wm_data, _, voxel_size = load_nifti(pve_path + "_wm_pve.nii.gz", return_voxsize=True)
                voxel_size = np.average(voxel_size[1:4]) 
                stopping_criterion =  CmcStoppingCriterion.from_pve(pve_wm_data, pve_gm_data, pve_csf_data,
                                                                    step_size=0.2, average_voxel_size=voxel_size)
                streamline_generator = ParticleFilteringTracking(peak_model, stopping_criterion, seeds, affine, step_size=0.2, 
                                                                 maxlen=1000, pft_back_tracking_dist=2, pft_front_tracking_dist=1, 
                                                                 particle_count=25, return_all=False)  
            else: 
                step_size = 0.5 if model=="_prob" and True else 0.5
                streamline_generator = LocalTracking(peak_model, stopping_criterion, seeds, affine=affine, step_size=step_size)
            
            streamlines = Streamlines(streamline_generator) 
             
            sft = StatefulTractogram(streamlines, ref_image, Space.RASMM)
            is_saved = save_tck(sft, output_name + "_%s.tck"%n_iter)
            
            # crop to non-dilated wm after filtering tracks that pass through bcgd and not through csf 
            crop = "tckresample " + output_name + "_%s.tck "%n_iter + output_name + "_%s.tck -step_size 0.5 -force ; "%n_iter if False and model=="_prob" else ""
            crop += "tckedit " + output_name + "_%s.tck "%n_iter + output_name + "_%s.tck -include "%n_iter + gm_path_2 + " -exclude " + csf_path_2  
            
            if n_iter%2==0:
                crop = crop + " -force ; tckedit " + output_name + "_%s.tck "%n_iter + output_name + "_%s.tck -mask "%n_iter + dire+"/"+p+"_wm_new.nii.gz -force ; " 
            else: 
                crop = crop + " -force ; tckedit " + output_name + "_%s.tck "%n_iter + output_name + "_%s.tck -mask "%n_iter + dire+"/"+p+"_wm_allnew.nii.gz -force ; " 
            crop = crop + "tckedit " + output_name + "_%s.tck "%n_iter + output_name + "_%s.tck -minlength 20 -force ; "%n_iter # removes short tracks
            """
            if n_iter%15==0:
                t1 = load_tck(output_name + "_%s.tck"%n_iter, f_path + "subjects/%s/dMRI/preproc/"%p + p + "_dmri_preproc.nii.gz")
                crop += "tcksift " + output_name + "_%s.tck "%n_iter + f_path + "subjects/" + p + "/tracking/preproc/" + p + "_fod.nii.gz " + output_name + "_%s.tck "%n_iter + "-term_number " + str(len(t1.streamlines))
            """ 
            # resamples the streamlines
            # crop = crop + "tckresample " + output_name + "_%s.tck "%n_iter + output_name + "_%s.tck -step_size  -force ; "%n_iter  
              
            bashcmd = crop.split()  
            process = subprocess.Popen(crop, universal_newlines=True, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
            # wait until finish
            out, error = process.communicate() 
            
            """
            t1 = load_tck(output_name + "_%s.tck"%n_iter, dmri_preproc)
            
            # seg_obj = Segmentation(seg_algo="AFQ",prob_threshold=0.1, return_idx=True, dany=False)
                         
            save_tck(t1,output_name + "_%s.tck"%n_iter)
            """
            return True
        
        SIFT = False  ; cpu_count = max(1,3) ; done = False   # multiprocessing.cpu_count()
        if n_iterations>=cpu_count and (single_bdl is None): # no sigle_bdle in loop
            for n_iter in range(n_iterations//cpu_count):
            
                # done = Parallel(n_jobs=min(5,cpu_count),verbose=60,pre_dispatch="n_jobs")(delayed(single_iter)(n_iter=subiter,single_bdl=None) for subiter in np.arange(n_iter*cpu_count,(n_iter+1)*5))
                for subiter in np.arange(n_iter*cpu_count,(n_iter+1)*cpu_count):
                    print(" ======  Tracking: %s / %s   ======= "%(subiter,n_iterations))
                    done = single_iter(n_iter=subiter,single_bdl=single_bdl)  
                
                if done: 
                    def check_size(output_name=output_name,SIFT=SIFT,dmri_preproc=dmri_preproc,f_path=f_path,n_iter=n_iter,n_iterations=n_iterations,model=model):
                        if  os.path.exists(output_name + ".trk"):
                            if os.path.getsize(output_name + ".trk")>=750000000 and not SIFT and n_iter<=n_iterations//2 and False:
                                SIFT = True
                                
                                t1 = load_tck(output_name + ".tck", dmri_preproc)
                                sift = "tcksift " + output_name + ".tck " + f_path + "subjects/" + p + "/tracking/preproc/" + p + "_fod.nii.gz " + output_name + ".tck " + "-term_number " + str(len(t1.streamlines)//2)
                            
                                # saved += output_name + "_%s.tck "%n_iter
                                # resamples the streamlines
                                # crop = crop + "tckresample " + output_name + "_%s.tck "%n_iter + output_name + "_%s.tck -step_size  -force ; "%n_iter  
                                  
                                bashcmd = sift.split()  
                                process = subprocess.Popen(sift, universal_newlines=True, shell=True, stdout=sys.stdout, stderr=subprocess.stderr)
                                # wait until finish
                                out, error = process.communicate() 
                            elif os.path.getsize(output_name + ".trk")>=1000000000 and model=="_prob":
                                print("File size exceeded")
                                os.system("rm " + "/".join(output_name.split("/")[:-1]) + "*.tck")
                                return True
                            elif os.path.getsize(output_name + ".trk")>=500000000 and not model=="_prob":
                                print("File size exceeded")
                                os.system("rm " + "/".join(output_name.split("/")[:-1]) + "*.tck")
                                return True
                        return False
                     
                    saved = "" ; check_ = False
                    with os.scandir(f_path + "subjects/%s/tracking/"%p) as it:
                        for entry in it: 
                            if len(entry.name.split('.'))>1:
                                if  entry.name.split('.')[-1]=="tck" \
                                    and ("prob" in entry.name.split('_') 
                                          or model == ""): 
                                    t1 = load_tck(entry.path , dmri_preproc) 
                                    trk = None 
                                    if  os.path.exists(output_name + ".trk"):
                                        t2 = load_trk(output_name + ".trk", dmri_preproc) 
                                        trk = StatefulTractogram(  list(t1.streamlines) + list(t2.streamlines),
                                                                   dmri_preproc,
                                                                   Space.VOX,
                                                                   data_per_streamline={k: (list(t1.data_per_streamline[k]) + list(t2.data_per_streamline[k]))
                                                                                            for k in t2.data_per_streamline.keys()  }) 
                                    else:
                                        trk = t1  
                                     
                                    save_trk(trk,output_name + ".trk") 
                                    os.system("rm "+ entry.path)
                                    check_ = check_size()
                                    if check_:
                                        break
                  
                    if check_:
                        break                  
                                     
        if not done: 
            if not single_bdl is None:
                if not os.path.isdir(f_path + "subjects/%s/tracking/Solo/"%p):
                    os.mkdir(f_path + "subjects/%s/tracking/Solo/"%p)
                output_name =  f_path + "subjects/%s/tracking/Solo/"%p + p + "_dipy" + model + "_" + single_bdl
                
            for n_iter in range(n_iterations):
                print(" ======  Tracking: %s / %s   ======= "%(n_iter,n_iterations))
                single_iter(n_iter=n_iter,single_bdl=single_bdl)  
                     
                def check_size(output_name=output_name,SIFT=SIFT,dmri_preproc=dmri_preproc,f_path=f_path,n_iter=n_iter,n_iterations=n_iterations,model=model):
                        if os.path.getsize(output_name + ".trk")>=750000000 and not SIFT and n_iter<=n_iterations//2 and False:
                            SIFT = True
                            
                            t1 = load_tck(output_name + "_%s.tck"%n_iter, f_path + "subjects/%s/dMRI/preproc/"%p + p + "_dmri_preproc.nii.gz")
                            sift = "tcksift " + output_name + "_%s.tck "%n_iter + f_path + "subjects/" + p + "/tracking/preproc/" + p + "_fod.nii.gz " + output_name + "_%s.tck "%n_iter + "-term_number " + str(len(t1.streamlines)//2)
                        
                            # saved += output_name + "_%s.tck "%n_iter
                            # resamples the streamlines
                            # crop = crop + "tckresample " + output_name + "_%s.tck "%n_iter + output_name + "_%s.tck -step_size  -force ; "%n_iter  
                              
                            bashcmd = sift.split()  
                            process = subprocess.Popen(sift, universal_newlines=True, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
                            # wait until finish
                            out, error = process.communicate()
                        elif os.path.getsize(output_name + ".trk")>=1000000000 and model=="_prob":
                            print("File size exceeded")
                            os.system("rm " + "/".join(output_name.split("/")[:-1]) + "*.tck")
                            return True
                        elif os.path.getsize(output_name + ".trk")>=500000000 and not model=="_prob":
                            print("File size exceeded")
                            os.system("rm " + "/".join(output_name.split("/")[:-1]) + "*.tck")
                            return True
                            
                        return False
                """
                saved = ""
                for subiter in np.arange(n_iter*5,(n_iter+1)*5):
                    saved += output_name + "_%s.tck "%subiter    
                merge = "tckedit " + saved + " " + output_name + ".tck ; rm " + saved
                bashcmd = merge.split()  
                process = subprocess.Popen(merge, universal_newlines=True, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
                # wait until finish
                out, error = process.communicate()   
                
                t1 = load_tck(output_name + ".tck", f_path + "subjects/%s/dMRI/preproc/"%p + p + "_dmri_preproc.nii.gz")  
                """
                saved = "" ; check_ = False
                with os.scandir(f_path + "subjects/%s/tracking/"%p) as it:
                    for entry in it: 
                        if len(entry.name.split('.'))>1:
                            if  entry.name.split('.')[-1]=="tck" \
                                and ("prob" in entry.name.split('_') or model == ""): 
                                t1 = load_tck(entry.path, dmri_preproc)  
                                trk = None 
                                if  os.path.exists(output_name + ".trk"):
                                    t2 = load_trk(output_name + ".trk", dmri_preproc)
                                    trk = StatefulTractogram(  list(t1.streamlines) + list(t2.streamlines),
                                                               dmri_preproc,
                                                               Space.VOX,
                                                               data_per_streamline={k: (list(t1.data_per_streamline[k]) + list(t2.data_per_streamline[k]))
                                                                                        for k in t2.data_per_streamline.keys()  }) 
                                else:
                                    trk = t1   
                                save_trk(trk,output_name + ".trk")  
                                os.system("rm "+ entry.path)
                                check_ = check_size()
                                if check_:
                                    break
                """
                if check_:
                    break
                """
                """                      
                query(model,"_"+str(n_iter))
                if os.path.exists(f_path + "subjects/" + patient + "/tracking/" + patient + "_dipy" + mod + "_cleaned%s.trk"%("_"+str(n_iter))):
                    os.system("rm "+ output_name + ".trk")
                """    
            
                                                  
            # merges results       
            with os.scandir(f_path + "subjects/%s/tracking/"%p) as it:
                for entry in it:
                    if os.path.isfile(f_path + "subjects/%s/tracking/"%p + entry.name) \
                        and entry.name.split('.')[-1]=="trk" \
                        and ("prob" in entry.name.split('_') or model == "") \
                        and "cleaned" in entry.name.split('_'):
                        
                        trk = None ; t1 = load_trk(f_path + "subjects/%s/tracking/"%p + entry.name, dmri_preproc)
                        if  os.path.exists(output_name + ".trk"):
                            t2 = load_trk(output_name + ".trk", dmri_preproc)
                            trk = StatefulTractogram(  list(t1.streamlines) + list(t2.streamlines),
                                                       dmri_preproc,
                                                       Space.VOX,
                                                       data_per_streamline={k: (list(t1.data_per_streamline[k]) + list(t2.data_per_streamline[k]))
                                                                                for k in t2.data_per_streamline.keys()  }) 
                        else:
                            trk = t1
                            
                        save_trk(trk,output_name + ".trk") 
    return True


def bootstrap(data,fun_to_boot):      
         
    data = np.nan_to_num(data) 
    if np.all(data==0):
        return 0
    
    if len(data)<=50:
        return fun_to_boot(data)
     
    def boot_iter(data, fun_to_boot, size_): 
        sample = np.random.choice(data, size=size_, replace=True) 
        return np.nan_to_num(fun_to_boot(sample))
        
    size_ = min(len(data),1000)
    
    result_array = Parallel(n_jobs=-1,pre_dispatch='n_jobs',require="sharedmem")( #,require="sharedmem"
                                 delayed(boot_iter)(data, fun_to_boot, size_) 
                                 for _ in range(size_) )
      
    biaised_result = np.nan_to_num(fun_to_boot(data))
    boot_1 = np.nan_to_num(np.nanmean(result_array))
    
    gc.collect()
    
    return 0.368*biaised_result + 0.632*boot_1
"""
    def log_like_iid_nbinom(params, n): 
        alpha, b = params
    
        if alpha <= 0 or b <= 0:
            return -np.inf
    
        return np.sum(sct.nbinom.logpmf(n, alpha, 1/(1+b)))
    
    
    def mle_iid_nbinom(n): 
    
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = scipy.optimize.minimize(
                fun=lambda params, n: -log_like_iid_nbinom(params, n),
                x0=np.array([0]),
                args=(n,),
                method='Powell',
                options={'xtol': 0.001, 'ftol': 0.001, "maxfev":n*1000000}
            )
    
        if res.success:
            return res.x
        else:
            print('Convergence failed with message: '+str(res.message))
         
    # This is gen_fun, for nonparametric, just a resampling
    # params = (alpha, beta), but is ignored
    # gen_params = (n,); this is used for nonparametric sample
    def resample_fun(params, n, size, rg):
        return rg.choice(n, size=size)
    
    bs_reps = bebi103.bootstrap.draw_bs_reps_mle(   mle_iid_nbinom,
                                                    resample_fun,
                                                    data,
                                                    gen_args=(len(data), ),
                                                    size=(min(len(data)*5,1000)),
                                                    n_jobs=2,
                                                    progress_bar=False,
    )
    
    result_array = [np.nan_to_num(fun_to_boot(sample)) for sample in bs_reps]
    
    biaised_result = np.nan_to_num(fun_to_boot(data))
    boot_1 = np.nan_to_num(np.mean(result_array))
    
    return 0.368*biaised_result + 0.632*boot_1
"""  
def fib_coherence(tg,threshold=0.01,rounds=3, p=None, f_path=None, tries = 0):
 
    try:
        try:
            streamlines = tg.streamlines
        except Exception:
            streamlines = StatefulTractogram(tg, f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(p,p), Space.RASMM).streamlines
    
        D33 = 1.0
        D44 = 0.02
        t = 1
        k = EnhancementKernel(D33, D44, t)    
        """ 
        fbc = FBCMeasures(streamlines, k, num_threads=-1)
        
        # Calculate LFBC for original fibers
        fbc_sl_orig, clrs_orig, rfbc_orig = fbc.get_points_rfbc_thresholded(0)
        
        # Apply a threshold on the RFBC to remove spurious fibers
        fbc_sl_thres, clrs_thres, rfbc_thres = fbc.get_points_rfbc_thresholded(threshold)
  
        return fbc_sl_thres, rfbc_orig, rfbc_thres
        """
        rfbc_thres, rfbc_orig = None, None
        last = 1000 ; diff = 10000 ; threshold = 0.01 
        fact = max(5*np.sqrt(300/len(streamlines)),4)
        true_len=len(streamlines) 
        for a_round in range(rounds):  
            if len(streamlines)<=true_len*0.75:
                break
            w = gaussian_weights(streamlines, return_mahalnobis=True, stat=np.mean).mean(1)
            w = w/w.sum() 
            th = np.quantile(w,1-(threshold))
            keep = w<=th
              
            w2 = gaussian_weights(streamlines, return_mahalnobis=True, stat=np.mean)
            w2 = w2.mean(1) 
            th = np.quantile(w2,1-threshold*fact) 
            keep2 = w2<=th
            streamlines = streamlines[keep2]
            
            fbc = FBCMeasures(streamlines, k, num_threads=-1, min_fiberlength=0)
            
            # Calculate LFBC for original fibers
            _ , clrs_orig, int_rfbc_orig = fbc.get_points_rfbc_thresholded(-1, verbose=True) 
            
            if a_round==0:
                rfbc_orig = np.mean(int_rfbc_orig)
            else:
                rfbc_thres = np.mean(int_rfbc_orig)
            
            w2 = gaussian_weights(streamlines, return_mahalnobis=True, stat=np.mean)
            w2 = w2.mean(1)  
            int_rfbc_orig = np.array(int_rfbc_orig*w2)  
            diff = last - np.mean(int_rfbc_orig)/np.std(int_rfbc_orig)
           
            last = np.mean(int_rfbc_orig)/np.std(int_rfbc_orig)
            plt.hist(int_rfbc_orig.flatten(),bins=100)
            th = np.quantile(int_rfbc_orig,threshold*fact)  
            # Apply a threshold on the RFBC to remove spurious fibers 
            keep_new = int_rfbc_orig>=th
            to_keep = np.zeros(len(streamlines))==1 
            streamlines = streamlines[keep_new]
            
        return streamlines, rfbc_orig, rfbc_thres
            
    except Exception as err:
        if str(err).split(" ")[0] == "Memoryview" and tries<=5: # try 5 times before giving up
            return fib_coherence(tg,threshold=threshold,rounds=rounds, p=p, f_path=f_path, tries = tries+1)
        else:
            print(err)
            return streamlines, 0, 0
   
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
    
def split_fanning(streamlines, streamlines_file, roi_atlas_path, diameter, tol=0.25, ending_regions=None, beginning_regions=None, seg_sl_assignments=None, csv_path=None, n_seg=0, seg_len=2, p=None,mod=None,bundle_name=None,f_path=None):
    """
    returns non-splitted streamlines segments numbers and the indices of corresponding streamlines assignements,
    or None if no split is detected
    """
    """
    endpoints_file = streamlines_file.split(".")[0] + "_endpoints.tck"
    endpoints_cmd = "tckresample -endpoints " + streamlines_file + " " + endpoints_file
    
    bashcmd = endpoints_cmd.split()  
    process = subprocess.Popen(endpoints_cmd, universal_newlines=True, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
    # wait until finish
    out, error = process.communicate()  
    """
    gc.collect()
    atlas, _ = load_nifti(roi_atlas_path) ; xn_count=0
    Xn = np.zeros(((atlas!=0).sum(),3)) # n points in 3D
    Yn = np.zeros(((atlas!=0).sum(),))  # their label in the atlas
    for xxx in range(len(atlas)):
        for yyy in range(len(atlas[0])):
            for zzz in range(len(atlas[0][0])):
                if atlas[xxx,yyy,zzz]!=0:
                    Xn[xn_count][:] = xxx,yyy,zzz
                    Yn[xn_count]    = atlas[xxx,yyy,zzz]
                    xn_count += 1
    knn = KNeighborsClassifier().fit(Xn, Yn)

    sl_start = [] ; sl_end = []  
    for sl_end_start in streamlines:
        sl_start.append(sl_end_start[0])
        sl_end.append(sl_end_start[-1])   
          
    endpoints = np.concatenate((sl_start,sl_end),axis=0) 
    
    Xnn = endpoints
    """
    for end_point in endpoints:
        Xn0n = np.concatenate((Xnn,np.array([end_point]).flatten()), axis=0)
    """ 
        
    predict_assign = knn.predict(Xnn) ; size = np.sum(atlas!=0) ; new_atlas = copy.copy(atlas)
    sl_size = len(streamlines) ; mean_reg_size = np.mean([ np.sum(new_atlas==label) for label in np.unique(atlas.flatten())[1:] ])
    
    start_pruning_th = 0.005 ; last_pruning_th = 0.01 ; pruning_iterations = 0 ; lab_len = len(np.unique(atlas.flatten())[1:])
    remove = [ start_pruning_th*mean_reg_size*np.sum(new_atlas==label)/(size*lab_len)>=2*(np.sum(predict_assign==label)/sl_size) 
               for label in np.unique(atlas.flatten())[1:] ]   
    while np.sum(remove)<=len(np.unique(new_atlas.flatten())[1:])//2 and pruning_iterations<50:
        remove = [ start_pruning_th*mean_reg_size*np.sum(new_atlas==label)/(size*lab_len)>=2*(np.sum(predict_assign==label)/sl_size) 
                   for label in np.unique(atlas.flatten())[1:] ]
        if np.sum(remove)>=20: 
            remove = [ last_pruning_th*mean_reg_size*np.sum(new_atlas==label)/(size*lab_len)>=2*(np.sum(predict_assign==label)/sl_size) 
                       for label in np.unique(atlas.flatten())[1:] ]
            break
        else: 
            last_pruning_th = copy.copy(start_pruning_th)
            start_pruning_th = last_pruning_th + 0.001
        pruning_iterations += 1
     
    print("Initialized atlas to %s regions"%np.sum(np.logical_not(remove))) ; pruning_iterations = 0
    while np.any(remove) and len(np.unique(new_atlas.flatten())[1:])>6 and pruning_iterations<200:
          
        for rem, lab in zip(remove,np.unique(new_atlas.flatten())[1:]):
            if rem:
                new_atlas[new_atlas==lab] = 0
                
        xn_count = 0
        Xn = np.zeros(((new_atlas!=0).sum(),3)) # n points in 3D
        Yn = np.zeros(((new_atlas!=0).sum(),))  # their label in the atlas
        for xxx in range(len(new_atlas)):
            for yyy in range(len(new_atlas[0])):
                for zzz in range(len(new_atlas[0][0])):
                    if new_atlas[xxx,yyy,zzz]!=0:
                        Xn[xn_count][:] = xxx,yyy,zzz
                        Yn[xn_count]    = new_atlas[xxx,yyy,zzz]
                        xn_count += 1
        knn = KNeighborsClassifier().fit(Xn, Yn)
        predict_assign = knn.predict(Xnn)
        size = np.sum(new_atlas!=0)
        remove = [ last_pruning_th*mean_reg_size*np.sum(new_atlas==label)/(size*lab_len)>=2*(np.sum(predict_assign==label)/sl_size) 
                   for label in np.unique(new_atlas.flatten())[1:] ]
        pruning_iterations += 1
        
    print("Pruned atlas to %s regions"%np.sum(np.logical_not(remove))) ; pruning_iterations = 0
    """
    def flipped_dist_matrix(x1,x2):
        columns_len = len(X[0])//2
        dist_ = np.zeros((len(X),len(X)))
        for ind1, x1 in enumerate(X[:-1]):
            for ind2, x2 in enumerate(X[ind1+1:]):
                forward = np.append(x1[:columns_len],x2[:columns_len]), np.append(x1[columns_len:],x2[columns_len:])
                dist_1 = np.mean(( forward - forward )**2) + np.mean((  -  )**2)
                dist_[ind1,ind2] = 
    """
    XXnn = np.array([ [xxnn] for xxnn in Xnn ])
    dist_ = distance_matrix_mdf(XXnn,XXnn)
    agg_clust = AgglomerativeClustering(n_clusters=None, affinity="precomputed", distance_threshold=dist_.mean()/2,
                                        linkage="average")
    agg_clust.fit(dist_)
    clusters = agg_clust.labels_
     
    print("Starting agglomerative clustering with %s clusters"%len(np.unique(clusters)))
    clust_pred = [] ; assignments = np.zeros((len(streamlines)*2,1))
    for cluster in np.unique(clusters):
    
        preds = knn.predict(Xnn[clusters==cluster])
        cmean = Xnn[clusters==cluster].mean(0)
        wpreds_1 = [np.sqrt((XNn-cmean)**2) for XNn in Xnn[clusters==cluster]]
        norm = np.mean(wpreds_1)
        wpreds = [norm/(1+w) for w in wpreds_1]
        wpreds = np.array(wpreds)/np.sum(wpreds)
        
        single_pred = []
        for cl_p in np.unique(preds):
            single_pred.append(wpreds[preds==cl_p].sum())
            
        clust_pred = preds[np.argmax(single_pred)]
        assignments[clusters==cluster] = clust_pred
      
    assignments = np.concatenate((assignments[:len(streamlines)],assignments[len(streamlines):]),axis=1)
    print("Ended with %s regions"%len(np.unique(assignments.flatten())))
    print(np.unique(assignments[:,0].flatten()),np.unique(assignments[:,1].flatten()))
    
    if ending_regions is None:
        ending_regions = np.unique(assignments.flatten())
    if beginning_regions is None:
        beginning_regions = np.unique(assignments.flatten())

    def combinations(possibilities):
        starts = np.unique(possibilities[:,0].flatten())
        ends = np.unique(possibilities[:,1].flatten())
        intersect = [False if _ in starts else True for _ in ends] 
        return itertools.product( starts, ends[intersect])

    print("Splitting associated branches") 
    sl_set = [[] for iiii in combinations(assignments)]  
    couples = [[None,None] for iiii in range(len(sl_set))]
    segments = [[[] for iiii in range(len(sl_set))] for nnn_seg in range(n_seg)]
    seg_assignment = [[[] for iiii in range(len(sl_set))]  for nnn_seg in range(n_seg)]
    for couple, (end_roi, start_roi) in enumerate(combinations(assignments)): 
        for assigned_sl, assignment in enumerate(assignments):   
            if (end_roi in assignment) and (start_roi in assignment):   
                split_sl = np.array(streamlines[assigned_sl]) 
                to_append_sl_set = []
                couples[couple] = (start_roi,end_roi)
                for nnn_seg in range(n_seg): 
                    to_append = split_sl[np.array(seg_sl_assignments[assigned_sl])==nnn_seg]
                    if len(to_append)>0:
                        if not (isinstance(to_append[0],list) or isinstance(to_append[0],np.ndarray)):
                            to_append = [to_append]  
                    segments[nnn_seg][couple].append(to_append)
                    to_append_sl_set.append(to_append)   
                    seg_assignment[nnn_seg][couple].append(assigned_sl) 
                sl_set[couple].append(to_append_sl_set)
          
    std_dist = [] ; distances = [] ; volumes = [] ; dist_matrices = []
    new_sl_set_non_splitted = [] ; new_sl_set_splitted = {}
    for this_seg, seg in enumerate(segments):    # segments should be this_sl splitted by segments  
        distances.append([]) ; volumes.append([])  
        for bdle_couple in itertools.combinations_with_replacement(np.arange(len(seg)),2) :  
            if len(seg[bdle_couple[0]])>0 and len(seg[bdle_couple[1]])>0:
                if np.any([len(_)>0 for _ in seg[bdle_couple[0]]]) and np.any([len(_)>0 for _ in seg[bdle_couple[1]]]): 
                
                    fgarray = [_ for _ in seg[bdle_couple[0]] if len(_)>1] # remove single-pont streamlines
                    temp_tg_ = StatefulTractogram(  fgarray,  
                                                    f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(p,p), 
                                                    Space.RASMM)  
                    new_sl_tg = _resample_tg(temp_tg_,int(seg_len*2))  
                    new_sl_1 = np.array(new_sl_tg)  
                    temp_tg_ = None  
                     
                    fgarray = [_ for _ in seg[bdle_couple[1]] if len(_)>1] # remove single-pont streamlines 
                    temp_tg_ = StatefulTractogram(  fgarray,  
                                                    f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(p,p), 
                                                    Space.RASMM)  
                    new_sl_tg = _resample_tg(temp_tg_,int(seg_len*2))
                    new_sl_2 = np.array(new_sl_tg)  
                    temp_tg_ = None ; fgarray = None
                    
                    try:
                        dist_matrix = distance_matrix_mdf(new_sl_1,new_sl_2)
                        mean_dist = dist_matrix.mean() 
                        dist_matrix = None
                        gc.collect()
                        if bdle_couple[0] != bdle_couple[1]:
                            distances[this_seg].append(mean_dist)
                        else:
                            if mean_dist == 0:
                                mean_dist = 1
                            volumes[this_seg].append(mean_dist)
                    except Exception as err:
                        print(err)
                        
        dist_matrix = np.eye(len(volumes[this_seg]))
        """
        for mat_lign, v in enumerate(volumes[this_seg]):
            dist_matrix[mat_lign,mat_lign] = v 
        """       
        mat_col = 1 ; mat_lign = 0
        for d_ in distances[this_seg]:
            dist_matrix[mat_lign,mat_col] = d_ / np.sqrt(volumes[this_seg][mat_lign]*volumes[this_seg][mat_col]) 
            mat_col += 1
            if mat_col==len(dist_matrix):
                mat_lign +=1
                mat_col = mat_lign + 1
        
        dist_matrices.append(dist_matrix)
        
        mean_distance = np.mean(distances[this_seg])
        # joblib.dump(distances,f_path + "subjects/%s/tracking/AFQ/cleaned_tracks"%p +mod+"/MAM/"+bundle_name+"_%s.txt"%this_seg) 
        split_matrix = abs(dist_matrix-np.triu(np.ones_like(dist_matrix)) )
        split = np.any(split_matrix>=tol*len(volumes)) # mean normalised dist with all braches exceeds threshold*total normalised volume (so # of branches)
        print(split_matrix,tol*len(volumes[this_seg]),split)  
        if split: 
            bdle_couple = None 
            counts = np.sum(split_matrix+split_matrix.T>=tol,axis=0)
            dist__ = np.sum(split_matrix+split_matrix.T,axis=0) 
            if (len(np.unique(counts))==2 and np.unique(counts)[0]==1) or len(counts)==2: # only one branch get away 
                if 2*volumes[this_seg][np.arange(len(seg))[np.argmax(dist__)]]<np.sum(volumes[this_seg]): # this is a minor branch
                    bdle_couple = np.arange(len(seg))[np.argmax(dist__)]  
                else: # it is the principal branch
                    if this_seg==0 or this_seg==len(segments)-1: # and or start ==> choose the smallest
                        bdle_couple = [_ for _ in np.arange(len(seg)) if _!=np.argmin(volumes[this_seg])] 
                    else:
                        bdle_couple = [_ for _ in np.arange(len(seg)) if _!=np.argmax(dist__)]   
                    
            elif this_seg==0 or this_seg==len(segments)-1:   
                bdle_couple = [_ for _ in np.arange(len(seg)) if _!=np.argmin(volumes[this_seg])] 
                
            else:  # all other cases are not covered and we split by default, keeping 1 segment at least
                bdle_couple = [_ for _ in np.arange(len(seg)) if _!=np.argmax(volumes[this_seg])] 
                
            new_sl_set_splitted[str(this_seg)] = bdle_couple  
    
    splitted_ = []
    for seg in np.arange(len(segments)):
        if str(this_seg) in [*new_sl_set_splitted.keys()]:
            if new_sl_set_splitted[str(this_seg)]==np.arange(len(segments[seg])): # all had split
                splitted_.append(True)
            else:
                splitted_.append(False)
        else:
            splitted_ = np.zeros_like(np.arange(len(segments)))==0 
            break
     
    if np.all(splitted_): # not normal to have only splits
        new_sl_set_splitted = {}
            
    joblib.dump(distances,f_path + "subjects/%s/tracking/AFQ/cleaned_tracks"%p +mod+"/MAM/"+bundle_name+"_dist.txt")
    joblib.dump(volumes,f_path + "subjects/%s/tracking/AFQ/cleaned_tracks"%p +mod+"/MAM/"+bundle_name+"_vol.txt")
    """        
    split_ind = []
    for seg_num in new_sl_set_splitted.keys():
        split_ind.append( np.arange(int(int(seg_num)*seg_len),int((int(seg_num)+1)*seg_len)) )
    """    
    unsplitted_sl = {} ; splitted_sl = {} ; ref = {}
    for couple, couple_sl in enumerate(sl_set):  # iterates over couples
        for iissl, ssl in enumerate(couple_sl):   # iterates over streamlines 
            issl = couple*100000 + iissl # no problem if no more than 100 000 streamlines per couple
            splitted_ = [] ; reat = False
            for _ in range(n_seg):
                 if len(ssl[_])>1:
                     if str(_) in [*new_sl_set_splitted.keys()]:
                         if np.any(couple==new_sl_set_splitted[str(_)]):
                             splitted_.append(True)
                         else: 
                             splitted_.append(False)
                     else: 
                         splitted_.append(False)
            if not (splitted_[0] or splitted_[-1]) and np.any(splitted_[1:-1]): # not normal to split only in the center ==> unsplit the whole streamline
                 reat=True
            else:
                while splitted_[0] and len(splitted_)>1: # skips all splits at beginning
                    splitted_ = splitted_[1:]
                while splitted_[-1] and len(splitted_)>1: # skips all splits at end
                    splitted_ = splitted_[:-1]
                if np.any(splitted_): # not normal to split somewhere in between to unsplitted segments, nor to have only splits
                    reat=True
                 
            for nnn_seg in range(n_seg):  
                if (not (str(nnn_seg) in [*new_sl_set_splitted.keys()]) and len(ssl[nnn_seg])>0) or reat: 
                   if str(issl) in [*unsplitted_sl.keys()]:  
                       if len(ssl[nnn_seg])>0:
                           min_dist_ = [np.linalg.norm(ssl[nnn_seg][0]-unsplitted_sl[str(issl)][-1]),
                                        np.linalg.norm(ssl[nnn_seg][-1]-unsplitted_sl[str(issl)][-1]),
                                        np.linalg.norm(ssl[nnn_seg][0]-unsplitted_sl[str(issl)][0]),
                                        np.linalg.norm(ssl[nnn_seg][-1]-unsplitted_sl[str(issl)][0])]  
                           min_ = np.argmin( min_dist_ )
                           if min_ == 0:
                               unsplitted_sl[str(issl)] = np.concatenate( (unsplitted_sl[str(issl)], ssl[nnn_seg]), axis=0) 
                           elif min_ == 1:
                               unsplitted_sl[str(issl)] = np.concatenate( (unsplitted_sl[str(issl)], ssl[nnn_seg][::-1]), axis=0) 
                           elif min_ == 2:
                               unsplitted_sl[str(issl)] = np.concatenate( (unsplitted_sl[str(issl)][::-1], ssl[nnn_seg] ), axis=0) 
                           else:
                               unsplitted_sl[str(issl)] = np.concatenate( (unsplitted_sl[str(issl)][::-1], ssl[nnn_seg][::-1] ), axis=0) 
                   elif len(ssl[nnn_seg])>0:  
                       unsplitted_sl[str(issl)] = ssl[nnn_seg]  
                elif len(ssl[nnn_seg])>0:   
                   to_split = new_sl_set_splitted[str(nnn_seg)]
                   if not (isinstance(to_split,list) or isinstance(to_split,np.ndarray)):
                       to_split = [to_split]
                       
                   if np.any(couple==to_split):
                       if str(couple) in [*splitted_sl.keys()]:  
                           if str(issl) in [*splitted_sl[str(couple)].keys()]: 
                               if len(ssl[nnn_seg])>0:
                                   min_ = np.argmin( [np.linalg.norm(ssl[nnn_seg][0]-splitted_sl[str(couple)][str(issl)][-1]),
                                                      np.linalg.norm(ssl[nnn_seg][-1]-splitted_sl[str(couple)][str(issl)][-1]),
                                                      np.linalg.norm(ssl[nnn_seg][0]-splitted_sl[str(couple)][str(issl)][0]),
                                                      np.linalg.norm(ssl[nnn_seg][-1]-splitted_sl[str(couple)][str(issl)][0])] )
                                   if min_ == 0:
                                       splitted_sl[str(couple)][str(issl)] = np.concatenate( (splitted_sl[str(couple)][str(issl)], ssl[nnn_seg]), axis=0)  
                                   elif min_ == 1:
                                       splitted_sl[str(couple)][str(issl)] = np.concatenate( (splitted_sl[str(couple)][str(issl)], ssl[nnn_seg][::-1]), axis=0)     
                                   elif min_ == 2:
                                       splitted_sl[str(couple)][str(issl)] = np.concatenate( (splitted_sl[str(couple)][str(issl)][::-1], ssl[nnn_seg]), axis=0) 
                                   else:
                                       splitted_sl[str(couple)][str(issl)] = np.concatenate( (splitted_sl[str(couple)][str(issl)][::-1], ssl[nnn_seg][::-1]), axis=0) 
                                       unsplitted_sl[str(issl)] = np.concatenate( (unsplitted_sl[str(issl)][::-1], ssl[nnn_seg][::-1] ), axis=0)  
                           else:    
                               splitted_sl[str(couple)][str(issl)] = ssl[nnn_seg]  
                       elif len(ssl[nnn_seg])>0:
                           splitted_sl[str(couple)] = {str(issl): ssl[nnn_seg] }
                   else:
                       if str(issl) in [*unsplitted_sl.keys()]: 
                           min_ = np.argmin( [np.linalg.norm(ssl[nnn_seg][0]-unsplitted_sl[str(issl)][-1]),
                                              np.linalg.norm(ssl[nnn_seg][-1]-unsplitted_sl[str(issl)][-1]),
                                              np.linalg.norm(ssl[nnn_seg][0]-unsplitted_sl[str(issl)][0]),
                                              np.linalg.norm(ssl[nnn_seg][-1]-unsplitted_sl[str(issl)][0])]  )  
                           if min_ == 0:
                               unsplitted_sl[str(issl)] = np.concatenate( (unsplitted_sl[str(issl)], ssl[nnn_seg]), axis=0) 
                           elif min_ == 1:
                               unsplitted_sl[str(issl)] = np.concatenate( (unsplitted_sl[str(issl)], ssl[nnn_seg][::-1]), axis=0) 
                           elif min == 2:
                               unsplitted_sl[str(issl)] = np.concatenate( (unsplitted_sl[str(issl)][::-1], ssl[nnn_seg] ), axis=0) 
                           else:
                               unsplitted_sl[str(issl)] = np.concatenate( (unsplitted_sl[str(issl)][::-1], ssl[nnn_seg][::-1] ), axis=0)
                       elif len(ssl[nnn_seg])>0:  
                           unsplitted_sl[str(issl)] = ssl[nnn_seg]  

    if new_sl_set_splitted!={}:   
        new_splitted_sl = []
        for ic_, c_ in enumerate([*splitted_sl.keys()]): # branches
            new_splitted_sl.append([])
            for sc_ in [*splitted_sl[c_].keys()]: # streamlines
                new_splitted_sl[ic_].append(splitted_sl[c_][sc_])
        splitted_sl = new_splitted_sl
        
        new_unsplitted_sl = []
        for ic_, c_ in enumerate([*unsplitted_sl.keys()]): # streamlines
            new_unsplitted_sl.append(unsplitted_sl[c_])
        unsplitted_sl = new_unsplitted_sl
         
        trk_split = unsplitted_sl ; bdle_list = None
        for spi in splitted_sl: 
            trk_split = trk_split + spi 
        if not os.path.exists(f_path + "subjects/%s/tracking/splitted_segments.trk"%(p)):
            trk_split = StatefulTractogram( trk_split,  
                                            f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(p,p), 
                                            Space.RASMM )
            trk_split.to_vox()
            save_trk(trk_split, f_path + "subjects/%s/tracking/splitted_segments.trk"%(p))
        """
        unsplitted_sl_ = [None for _ in unsplitted_sl[0]]
        for seg in unsplitted_sl:
            for isl_, sl_ in enumerate(seg):
                unsplitted_sl_[isl_] = sl_ if unsplitted_sl_[isl_] is None else np.concatenate((unsplitted_sl_[isl_],sl_),axis=0)
        
        splitted_sl_ = [[None for __ in _[0]] for _ in splitted_sl]
        """ 
        return unsplitted_sl, splitted_sl, dist_matrices 
    else: 
        return streamlines, None, dist_matrices

def curvature(streamline):
    
    curv_to_return = np.zeros_like(streamline[:,0])
    str_x = streamline[:,0]
    str_y = streamline[:,1]
    str_z = streamline[:,2]
    
    # let's extend to prepare for derivation
    str_x1 = np.concatenate(( str_x, [str_x[-1]] )) ; str_x2 = np.concatenate(( [str_x[0]], str_x ))
    str_y1 = np.concatenate(( str_y, [str_y[-1]] )) ; str_y2 = np.concatenate(( [str_y[0]], str_y ))
    str_z1 = np.concatenate(( str_z, [str_z[-1]] )) ; str_z2 = np.concatenate(( [str_z[0]], str_z )) 
      
    # gradient
    grad_x1 = np.diff(str_x1) ; grad_x2 = np.diff(str_x2)
    grad_y1 = np.diff(str_y1) ; grad_y2 = np.diff(str_y2)
    grad_z1 = np.diff(str_z1) ; grad_z2 = np.diff(str_z2)
     
    grad_x = (grad_x1 + grad_x2)/2
    grad_y = (grad_y1 + grad_y2)/2
    grad_z = (grad_z1 + grad_z2)/2
    
    # let's extend to prepare for derivation
    grad_x1 = np.concatenate(( grad_x, [grad_x[-1]] )) ; grad_x2 = np.concatenate(( [grad_x[0]], grad_x ))
    grad_y1 = np.concatenate(( grad_y, [grad_y[-1]] )) ; grad_y2 = np.concatenate(( [grad_y[0]], grad_y ))
    grad_z1 = np.concatenate(( grad_z, [grad_z[-1]] )) ; grad_z2 = np.concatenate(( [grad_z[0]], grad_z ))
     
    # 1D curvature
    curv_x1 = np.diff(grad_x1) ; curv_x2 = np.diff(grad_x2)
    curv_y1 = np.diff(grad_y1) ; curv_y2 = np.diff(grad_y2)
    curv_z1 = np.diff(grad_z1) ; curv_z2 = np.diff(grad_z2)
    
    curv_x = (curv_x1 + curv_x2)/2
    curv_y = (curv_y1 + curv_y2)/2
    curv_z = (curv_z1 + curv_z2)/2
     
    # 3D curvature estimates
    curv_to_return = np.sqrt( (curv_z*grad_y - curv_y*grad_z)**2 + (curv_x*grad_z - curv_z*grad_x)**2 + (curv_y*grad_x - curv_x*grad_y)**2 )/np.power(grad_x**2 + grad_y**2 + grad_z**2, 1.5)

    return curv_to_return*100
      
def diameter_and_dir(streamlines_in_seg, weights=None, sampling_dist=4,f_path=None,p=None):
     
    if streamlines_in_seg is None:
        return np.nan, np.nan, np.nan, np.nan
     
    slices = int(np.rint((sampling_dist-1)*2))
    # new_sl = np.array([set_number_of_points(np.array(sub_seg),nb_points=slices) for sub_seg in streamlines_in_seg])
    temp_tg = copy.copy(streamlines_in_seg)
    if isinstance(streamlines_in_seg,list): 
        streamlines_in_seg = list([ tttt for itt,tttt in enumerate(streamlines_in_seg) if len(tttt)>2]) 
        temp_tg = StatefulTractogram( streamlines_in_seg,  
                                      f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(p,p), 
                                      Space.RASMM)
    
    new_sl_tg = _resample_tg(temp_tg,slices) ; new_sl = None
    if isinstance(new_sl_tg,list) or isinstance(new_sl_tg,np.ndarray):
        new_sl = np.array(new_sl_tg)
        streamlines_in_seg = list([ tttt for itt,tttt in enumerate(streamlines_in_seg) if len(tttt)>2])
    else:
        new_sl = np.array(new_sl_tg.streamlines) 
          
    """
    new_sl_pca = _resample_tg(temp_tg,int(np.rint(1 + 2*sampling_dist)))  
    if isinstance(new_sl_pca,list) or isinstance(new_sl_pca,np.ndarray):
        new_sl_pca = np.array(new_sl_pca)[:,1::2]
    else:
        new_sl_pca = np.array(new_sl_pca.streamlines)[:,1::2] 
    """   
    if len(new_sl)>50:
        mean_sl = np.mean(new_sl,1)
        """
        centroids = np.mean(new_sl_pca,0) 
        pca_assignments = []
        for s in streamlines_in_seg: 
            pca_assignments.append([np.argmin([np.linalg.norm(si-centroid) for centroid in centroids]) for si in s]) 
         
        mean_diameter = []  ; bundle_direction = []
        for ppoint in range(new_sl_pca.shape[1]):
        
            coords = None 
            for iis, s in enumerate(streamlines_in_seg):  
                to_append = s[pca_assignments[iis]==ppoint]
                if not isinstance(to_append,list):
                    to_append = [to_append]
                for point_to_append in to_append: 
                    if coords is None:
                        coords = point_to_append
                    else:
                        coords = np.concatenate((coords,point_to_append),axis=0)
             
            try:
                pca = PCA(n_components=3)
                pca.fit_transform(np.array(coords))
                ellipsis_axes = pca.components_ 
                
                # flat segment so smallest component is axial the direction of the bundle
                bundle_direction.append(ellipsis_axes[2])
                 
                # approximate mean diameter as mean  of the probabilistic elipsoid (3 times std <=> 95% of datas in it)
                mean_diameter.append( [pca.explained_variance_[0] * 3, pca.explained_variance_[1] * 3] )
            
            except Exception as err: 
                mean_diameter.append( [np.nan, np.nan] )
                 
        mean_diameter = np.array(mean_diameter)
        mean_diameter = (np.nanmean(mean_diameter[:,0]),
                         np.nanmean(mean_diameter[:,1])) # np.array(mean_diameter).mean(0)
        bundle_direction = np.nanmean(bundle_direction)
        """
        mean_diameter2 = []
        for ppoint in range(slices): 
            mean_diameter2.append( 3*np.linalg.norm( mean_sl - new_sl[:,ppoint], axis=-1 ).std() )
        
        return 0, 0, 0, np.nanmean(mean_diameter2) 
        # return mean_diameter[0], mean_diameter[1], bundle_direction, np.nanmean(mean_diameter2) 
    else: 
        return np.nan, np.nan, np.nan, np.nan  
     
# adapted from pyAFQ API   
def get_streamlines_json(f_path=my_f_path,p=None,clean_bundles=None,path="/home/users/d/r/drimez/",sub=patient_list,bundle_dict=None):
    sls_json_fname = op.abspath(op.join(
        path, "afqb_streamlines.json"))
    if not op.exists(sls_json_fname) or True:
        subses_info = []

        brain_path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/atlases_dany/reference_brains/MNI152NLin6_res-1x1x1_T1w_descr-brain.nii.gz"
        def load_next_subject(subses_info):  
            root = my_f_path + "subjects/H_0/dMRI/preproc/"
            this_bundles_file = clean_bundles
            this_img = nib.load(root+"H_0_dmri_preproc.nii.gz")
            fbval, fbvec = my_f_path+"data_1/H_0.bval", my_f_path+"data_1/H_0.bvec"
            gtab = dpg.gradient_table(fbval,fbvec) 
            mapping = reg.syn_register_dwi( this_img, gtab,
                                            template=brain_path)[1]
            this_sft = load_tractogram( this_bundles_file,
                                        this_img,
                                        Space.RASMM)
            subses_info.append((this_sft, this_img, mapping))
            return subses_info
        
        bundle_dict = BundleDict(bundle_info=BUNDLES,resample_to=brain_path,seg_algo="AFQ") 
        bundle_id = 1
        for name, bundle in bundle_dict.items(): 
            bundle_dict[name] = {"uid":bundle_id} 
            bundle_id += 1
        
        sls_dict = {}
        subses_info = load_next_subject(subses_info)  # load first subject
        for b in bundle_dict.keys():
            if b != "whole_brain":
                for i in range(len(sub)):
                    sft, img, mapping = subses_info[i]
                    idx = np.where( sft.data_per_streamline['bundle'] == bundle_dict[b]['uid'] )[0]
                    # use the first subses that works
                    # otherwise try each successive subses
                    if len(idx) == 0:
                        # break if we run out of subses
                        if i + 1 >= len(self.valid_sub_list):
                            break
                        # load subses if not already loaded
                        if i + 1 >= len(subses_info):
                            load_next_subject()
                        continue
                    if len(idx) > 100:
                        idx = np.random.choice(
                            idx, size=100, replace=False)
                    these_sls = sft.streamlines[idx]
                    these_sls = dps.set_number_of_points(these_sls, 100)
                    tg = StatefulTractogram(  these_sls,
                                              img,
                                              Space.RASMM)
                    tg.to_rasmm()
                    delta = dts.values_from_volume( mapping.forward,
                                                    tg.streamlines, np.eye(4))
                    moved_sl = dts.Streamlines([d + s for d, s in zip(delta, tg.streamlines)])
                    moved_sl = np.asarray(moved_sl)
                    median_sl = np.median(moved_sl, axis=0)
                    sls_dict[b] = {"coreFiber": median_sl.tolist()}
                    for ii, sl_idx in enumerate(idx):
                        sls_dict[b][str(sl_idx)] = moved_sl[ii].tolist()
                    break

        with open(sls_json_fname, 'w') as fp:
            json.dump(sls_dict, fp)
            
    return sls_json_fname
  
def scnd_round(trk, det_trk, bundle_dict, dwi_affine, profile_weights="gauss",f_path=None,p=None,mod=None,split=True,seg_len_mm=2):
   
    if not (profile_weights is None
            or isinstance(profile_weights, str)
            or callable(profile_weights)
            or hasattr(profile_weights, "__len__")):
        raise TypeError(
            "profile_weights must be string, None, callable, or"
            + "a 1D or 2D array")
    if isinstance(profile_weights, str):
        profile_weights = profile_weights.lower()
    if isinstance(profile_weights, str) and\
            profile_weights != "gauss" and profile_weights != "median":
        raise TypeError(
            "if profile_weights is a string,"
            + " it must be 'gauss' or 'median'")

    keys = []
    vals = []
    for k in bundle_dict.keys():
        if k != "whole_brain":
            keys.append(bundle_dict[k]['uid'])
            vals.append(k)
    reverse_dict = dict(zip(keys, vals))

    bundle_names = []
    node_numbers = []
    profiles = []
    this_profile = [] 

    clean_path = f_path  + "subjects/%s/tracking/AFQ/cleaned_tracks"%p + mod + "/"  
    if not os.path.isdir(clean_path):
        os.mkdir(clean_path)
        os.mkdir(clean_path+"/profiles/")
        os.mkdir(clean_path+"/MAM/")
        os.mkdir(clean_path+"/FBC/")
        
    fbc = {} ; MAM = {}
    if isinstance(trk,str):
        trk = nib.streamlines.load(trk) 
        trk.to_rasmm()
     
    bundle_name = None
    for b in range(1):
        this_sl = trk 
        bundle_name = [*bundle_dict.keys()][b]
        fbc_sl_thres, rfbc_orig, rfbc_thres = fib_coherence(this_sl, p=p, f_path=f_path) 
        fbc["original"] = rfbc_orig
        fbc["final"] =  rfbc_thres # before, after
        trk2 = StatefulTractogram( fbc_sl_thres,
                                   f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(p,p),
                                   Space.RASMM ) 
        # trk2.to_vox()
        # save_trk(trk2, clean_path + bundle_name + ".trk")
        # save_tck(trk2, clean_path + bundle_name + ".tck")
        print("FBC step done")
        tracks_filename = f_path + "subjects/%s/tracking/AFQ/tracks"%p + mod + "/" + bundle_name + ".trk"
        tracks_filename = clean_path + bundle_name + ".trk"
        this_sl = trk2.streamlines
        
        if split:
        
            lengths = length(this_sl)
            mean_length = np.mean(lengths[np.logical_and(lengths>=np.quantile(lengths,0.25),
                                                         lengths<=np.quantile(lengths,0.75)) ]) 
            n_seg = int(np.rint(mean_length/seg_len_mm))
            seglen = mean_length/n_seg
              
            temp_tg = StatefulTractogram( this_sl[lengths>=np.quantile(lengths,0.5)],  
                                          f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(p,p), 
                                          Space.RASMM )
        
            new_sl_tg = _resample_tg(temp_tg,1 + 2*n_seg)  
            new_sl = np.array(new_sl_tg)  
              
            centroids = np.mean(new_sl[:,1::2],0)  
            assignments_ = []
            for s in this_sl: 
                assignments_.append([np.argmin([np.linalg.norm(si-centroid) for centroid in centroids]) for si in s])  
            new_sl = None 
            """
            if isinstance(profile_weights, str):
                if profile_weights == "gauss":
                    this_prof_weights = gaussian_weights(this_sl, n_points=n_seg) 
                elif profile_weights == "median":
                    # weights bundle to only return the mean
                    def _median_weight(bundle):
                        # get number of point then new_number = number/freq
                        fgarray = set_number_of_points(bundle, n_seg)
                        values = np.array( values_from_volume(  scalar_data,
                                                                fgarray,
                                                                dwi_affine    ) )
                        weights = np.zeros(values.shape)
                        for ii, jj in enumerate(np.argsort(values, axis=0)[len(values)//2, :]):
                            weights[jj, ii] = 1
                        return weights
                    this_prof_weights = _median_weight
            else:
                this_prof_weights = profile_weights
            """
            print("Whole bundle morphometric step")
            data = np.zeros((4,n_seg,))   
            data0 = [[ [] for nn_sl in range(len(this_sl))] for nn_seg in range(n_seg)] 
            points_in_seg = [[ [] for nn_sl in range(len(this_sl))] for nn_seg in range(n_seg)] 
            for n_sl, this_sl_single in enumerate(this_sl):
            
                curv = curvature(this_sl_single)  
                for n_p, point in enumerate(curv):
                    seg_id = assignments_[n_sl][n_p] 
                    data0[seg_id][n_sl] += np.nan_to_num(float(point)/np.sum(assignments_[n_sl]==seg_id)) # this_prof_weights[n_sl,seg_id]*
                    points_in_seg[seg_id][n_sl].append(this_sl_single[n_p])
             
            bund_dir = np.zeros((n_seg,3))  
            for nn_seg in range(n_seg):
                to_compute = points_in_seg[nn_seg]
                if nn_seg==(n_seg-1) or True:
                    to_compute = copy.copy(points_in_seg[nn_seg])
                    to_compute = [ to_compute[itt] for itt,tttt in enumerate(points_in_seg[nn_seg]) if len(tttt)>2]
                if len(to_compute)>50:
                    diamx, diamy, bdir, diam2 = diameter_and_dir(to_compute,sampling_dist=seg_len_mm,f_path=f_path,p=p) 
                    data[1,nn_seg] = diamx
                    data[2,nn_seg] = diamy
                    data[3,nn_seg] = diam2
                    bund_dir[nn_seg] = bdir 
                    
            data_0 = Parallel(n_jobs=-1,pre_dispatch='2*n_jobs',require="sharedmem")(delayed(np.nanmean)(data0[segment]) for segment in range(n_seg))                     
            data_1 = Parallel(n_jobs=-1,pre_dispatch='2*n_jobs',require="sharedmem")(delayed(np.nanstd)(data0[segment]) for segment in range(n_seg)) 
            """
            points_in_seg = [[ [] for nn_sl in range(len(this_sl))] for nn_seg in range(n_seg)] 
            for n_sl, this_sl_single in enumerate(this_sl):
            
                curv = curvature(this_sl_single)
                seglen = np.rint(len(this_sl_single)/n_seg)
                if seglen<2: # resample the tract to have at least 2 point per slice
                    single_tract_tg = [this_sl_single]
                    this_sl_single = _resample_tg(single_tract_tg, int(n_seg*2.2))
                    seglen = np.rint(len(this_sl_single)/n_seg)
                points_in_sl = []
                for n_p, point in enumerate(curv):
                    seg_id = min(int(np.rint(np.nan_to_num(n_p/seglen))),n_seg-1) 
                    data[0,seg_id] += np.nan_to_num(this_prof_weights[n_sl,seg_id]*float(point)/seglen)  
                    points_in_seg[seg_id][n_sl].append(this_sl_single[n_p])
             
            bund_dir = np.ones((n_seg,3))*(-1)  
            for nn_seg in range(n_seg): 
                to_compute = points_in_seg[nn_seg]
                if nn_seg==(n_seg-1):
                    to_compute = copy.copy(points_in_seg[nn_seg])
                    to_compute = [ to_compute[itt] for itt,tttt in enumerate(points_in_seg[nn_seg]) if len(tttt)>2]
                if len(to_compute)>50:
                    diamx, diamy, bdir, diam2 = diameter_and_dir(to_compute,sampling_dist=seg_len_mm,f_path=f_path,p=p) 
                    data[1,nn_seg] = diamx
                    data[2,nn_seg] = diamy
                    data[3,nn_seg] = diam2
                    bund_dir[nn_seg] = bdir 
            
            data[0] /= len(this_sl)
            """
            profiles.append(list(data_0))
            profiles.append(list(data_1))
            # profiles.append(list(data[0]))
            profiles.append(list(data[1]))
            profiles.append(list(data[2]))
            profiles.append(list(data[3])) 
            profiles.append(list(bund_dir))  
         
            print("Splitting into branches") 
            ending_regions = np.arange(1,42) ; beginning_regions = np.arange(1,42)
            unsplitted_sl, splitted_sl, MAM_dist = split_fanning( this_sl, '.'.join(tracks_filename.split('.')[0]) + ".tck", seg_sl_assignments=assignments_,#, ending_regions, beginning_regions, 
                                                           roi_atlas_path=f_path+"subjects/%s/tracking/preproc_atlas/%s_AAL_affreg.nii.gz"%(p,p), f_path=f_path,
                                                           diameter=min( np.mean(data[1]), np.mean(data[2]) ), seg_len=seglen, n_seg=n_seg, p=p, mod=mod, bundle_name=bundle_name)
            MAM[bundle_name] = MAM_dist
             
            if not (splitted_sl is None):
                print("Saving splits")
                this_sl_list = []
                trkname = bundle_name
                filenames = [trkname + "_%s"%o for o in range(1,1+len(splitted_sl))]
                for index, this_splitted_sl in enumerate(splitted_sl):
                    
                    print(np.shape(this_splitted_sl))
                    fbc_sl_thres, rfbc_orig, rfbc_thres = fib_coherence(this_splitted_sl, p=p, f_path=f_path, rounds=1)  
                    fbc[bundle_name + "_" + str(index+1)] = rfbc_orig, rfbc_thres # before, after 
                    
                    # fbc_sl_thres = list(fbc_sl_thres) # if not (det_trk is None) else fbc_sl_thres
                    print(np.shape(fbc_sl_thres))
                    org_index = np.arange(len(fbc_sl_thres))
                    # fbc_sl_thres = [fbc_sl_thres[clean] for clean in clean_idx if (clean in org_index)] 
                      
                    new_track = StatefulTractogram( fbc_sl_thres,  
                                                    f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(p,p), 
                                                    Space.RASMM)
                    # new_track.to_vox()
                    # save_trk(new_track, f_path + "subjects/%s/tracking/AFQ/cleaned_tracks"%p +mod+"/" + filenames[index] + ".trk") 
                    this_sl_list.append(new_track)
                    
                    new_bdl = bundle_dict[bundle_name]
                    new_bdl['uid'] = 10*new_bdl['uid'] + index
                    bundle_dict[bundle_name + "_" + str(index+1)] = new_bdl
                    
                fbc_sl_thres, rfbc_orig, rfbc_thres = fib_coherence(unsplitted_sl, p=p, f_path=f_path, rounds=1)  
                fbc[bundle_name + "_0"] = rfbc_orig, rfbc_thres # before, after 
                
                new_track = StatefulTractogram( fbc_sl_thres,  
                                                f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(p,p), 
                                                Space.RASMM)
                # new_track.to_vox()                                 
                # save_trk(new_track, f_path + "subjects/%s/tracking/AFQ/cleaned_tracks"%p +mod+"/" + trkname + "_0.trk")
                this_sl_list.append(new_track)
                my_to_csv(f_path + "subjects/%s/tracking/AFQ/cleaned_tracks"%p +mod+"/MAM/%s.txt"%(bundle_name),MAM)
                
            else: 
                print("No splits detected")
                this_sl_list = [this_sl]
                filenames = [".".join(tracks_filename.split(".")[:-1])]
           
        else:
            print("Set splitting to False, not splitting")
            this_sl_list = [this_sl]
            filenames = [".".join(tracks_filename.split(".")[:-1])]
    """
    profile_dict = dict() 
     
    features = ["curvature (mean)","curvature (std)","diameter_x","diameter_y","diameter_2","bundle_dir"] 
    for ii, scalar in enumerate(features):
        profile_dict[scalar] = profiles[ii]
    """
    my_to_csv(f_path + "subjects/%s/tracking/AFQ/cleaned_tracks"%p +mod+"/FBC/%s.txt"%(bundle_name),fbc)
    # my_to_csv(f_path + "subjects/%s/tracking/AFQ/cleaned_tracks"%p +mod+"/MAM/%s.txt"%(bundle_name),MAM)
    # my_to_csv(f_path + "subjects/%s/tracking/AFQ/cleaned_tracks"%p +mod+"/profiles/%s.txt"%(bundle_name),profile_dict,columns=features)

    return this_sl_list, filenames, bundle_dict
 
def read_txt(txt_path):
    
    output_txt = None
    with open(txt_path,"r") as reader:
        output_txt = reader.readlines()
        reader.close()
    
    keys = output_txt[0].split("\n")[0].split(",") 
    output_ = {keys[ic]:[] for ic in range(len(keys))}
    for line in output_txt[1:]:
        for ic, column in enumerate(line.split("\n")[0].split(",")):
            output_[keys[ic]].append(column)
    
    if len(output_txt)==2:
        return pd.DataFrame.from_dict([output_])
    else:
        return pd.DataFrame.from_dict(output_)

from dipy.stats.analysis import orient_by_streamline, values_from_volume 
import warnings  
# slightly adapted copy-paste from API (:
def tract_profiles(clean_bundles_folder, data_imap, scalar_dict, dwi_affine, profile_weights="gauss",f_path=None,p=None,mod=None,seg_len_mm=4,step_len=0.5,ref_path=None):
    """
    full path to a CSV file containing tract profiles

    Parameters
    ----------
    profile_weights : str, 1D array, 2D array callable, optional
        How to weight each streamline (1D) or each node (2D)
        when calculating the tract-profiles. If callable, this is a
        function that calculates weights. If None, no weighting will
        be applied. If "gauss", gaussian weights will be used.
        If "median", the median of values at each node will be used
        instead of a mean or weighted mean.
        Default: "gauss"
    """ 
    bundle_dict = data_imap["bundle_dict"]
    if not (profile_weights is None
            or isinstance(profile_weights, str)
            or callable(profile_weights)
            or hasattr(profile_weights, "__len__")):
        raise TypeError(
            "profile_weights must be string, None, callable, or"
            + "a 1D or 2D array")
    if isinstance(profile_weights, str):
        profile_weights = profile_weights.lower()
    if isinstance(profile_weights, str) and\
            profile_weights != "gauss" and profile_weights != "median":
        raise TypeError(
            "if profile_weights is a string,"
            + " it must be 'gauss' or 'median'")

    keys = [] ; vals = []
    for k in bundle_dict.keys():
        if k != "whole_brain":
            keys.append(bundle_dict[k]['uid'])
            vals.append(k)
    reverse_dict = dict(zip(keys, vals))

    bundle_names = [] ; node_numbers = []
    profiles = [] ; this_profile = []  
         
    # trk = nib.streamlines.load(clean_bundles_file) 
    def single_profile(trk_path,bundle_name,parrallel, reverse_dict=reverse_dict, step_len=step_len, seg_len_mm=seg_len_mm, ref_path=ref_path,
                        scalar_dict=scalar_dict, dwi_affine=dwi_affine, profile_weights=profile_weights,f_path=f_path,p=p,mod=mod):
        import shutup
        shutup.please()
        trk = load_trk(trk_path,ref_path)            
        bundle_names = [] ; node_numbers = [] ; profiles = []  
        this_sl = trk.streamlines    
         
        n_seg = None ; lengths = length(this_sl) ; mean_length = None
        if not isinstance(lengths,np.ndarray):
            mean_length = lengths
            lengths = np.array([lengths])
        else: 
            standard = this_sl[np.argmin(abs(lengths-np.quantile(lengths,0.625)))] # sets reference streamline to the mid-second-quartile length (long enough to be in the bundle, short enough to be "straight")
            this_sl = orient_by_streamline(this_sl, standard, n_points=100)        # reorient streamlines for correct estimation of mean bundle
            
            mean_length = np.mean(lengths[np.logical_and(lengths>=np.quantile(lengths,0.25),
                                                                 lengths<=np.quantile(lengths,0.75)) ]) if len(lengths)>100 \
                                  else np.mean(lengths)
        if os.path.exists(f_path+"/afq_ref.txt"):
            ref_df = read_txt(f_path+"/afq_ref.txt") 
            if not bundle_name in ref_df["bundle"].values: 
                n_seg = int(np.rint(mean_length/seg_len_mm))
                if (not n_seg==0):
                    print(p+","+bundle_name+","+str(n_seg),file=open(f_path+"/afq_ref.txt","a")) 
            else:
                try:
                    n_seg = int(ref_df.values[ref_df.values[:,1]==bundle_name,2]) 
                except Exception:
                    n_seg = int(ref_df.values[ref_df.values[:,1]==bundle_name,2][0])
                    print("Bundle " + bundle_name + " is " + str(len(ref_df.values[ref_df.values[:,1]==bundle_name,2])) \
                          + " times in the ref.txt file!\nUsing the first lign.")
        else:  
            n_seg = int(np.rint(mean_length/seg_len_mm))
            if (not n_seg==0):
                print("patient,bundle,n_seg",file=open(f_path+"/afq_ref.txt","a")) 
                print(p+","+bundle_name+","+str(n_seg),file=open(f_path+"/afq_ref.txt","a")) 
        
        if n_seg>0: 
            seglen = mean_length/n_seg 
                  
            print("Defining centroids for sampling")  
            new_sl = None
            centroids = None
            if len(this_sl)!=0:
                if isinstance(this_sl[0],list) or isinstance(this_sl[0],np.ndarray):
                    if not (isinstance(this_sl[0][0],list) or isinstance(this_sl[0][0],np.ndarray)): # it is not a list of streamlines but a single streamline
                        this_sl = [this_sl]
             
            stop_ = False                                
            if len(this_sl)>1:
                this_sl = [_ for _ in this_sl if len(_)>2]
            else:
                stop_ = len(this_sl)<=2
            
            if not stop_:
                if len(this_sl)>1: 
                    new_sl_tg = _resample_tg(trk,1 + 2*n_seg)  
                    new_sl = np.array(new_sl_tg)  
                    centroids = np.mean(new_sl[:,1::2],0)  # centrers of segments
                elif len(this_sl)>50:                      
                    temp_tg = StatefulTractogram.from_sft( this_sl[np.logical_and(lengths>=np.quantile(lengths,0.25),
                                                                     lengths<=np.quantile(lengths,0.75)) ], trk)  
                    new_sl_tg = _resample_tg(temp_tg,1 + 2*n_seg)  
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
                      
                print("Sampling along %s bundle"%bundle_name)
                """      
                
                """     
                print("Computing morphometry metrics")
                data = np.zeros((4,n_seg,)) 
                data0 = [[ np.nan for nn_sl in range(len(this_sl))] for nn_seg in range(n_seg)] 
                points_in_seg = [[ [] for nn_sl in range(len(this_sl))] for nn_seg in range(n_seg)] 
                for n_sl, this_sl_single in enumerate(this_sl): 
                    if len(this_sl_single)>1:
                        try:
                            curv = curvature(np.array(this_sl_single))
                            for n_p, point in enumerate(curv):
                                seg_id = assignments_[n_sl][n_p]  
                                if np.isnan(data0[seg_id][n_sl]):
                                    data0[seg_id][n_sl] = np.nan_to_num(float(point)/np.sum(assignments_[n_sl]==seg_id))
                                else:
                                    data0[seg_id][n_sl] += np.nan_to_num(float(point)/np.sum(assignments_[n_sl]==seg_id)) 
                                points_in_seg[seg_id][n_sl].append(this_sl_single[n_p])
                        except Exception:
                            print(np.array(this_sl_single).shape,file=open("/auto/home/users/d/r/drimez/afq_err.txt","w"))
                 
                bund_dir = np.zeros((n_seg,3))  
                for nn_seg in range(n_seg):
                    to_compute = points_in_seg[nn_seg]
                    if nn_seg==(n_seg-1) or True:
                        to_compute = copy.copy(points_in_seg[nn_seg])
                        to_compute = [to_compute[itt] for itt,tttt in enumerate(points_in_seg[nn_seg]) if len(tttt)>2]
                    if len(to_compute)>50:
                        diamx, diamy, bdir, diam2 = diameter_and_dir(to_compute,sampling_dist=seg_len_mm,f_path=f_path,p=p) 
                        data[1,nn_seg] = diamx
                        data[2,nn_seg] = diamy
                        data[3,nn_seg] = diam2
                        bund_dir[nn_seg] = bdir 
                        
                data_0 = Parallel(n_jobs=-1,pre_dispatch='2*n_jobs', mmap_mode="r+")(delayed(bootstrap)(data0[segment], np.nanmean) for segment in range(n_seg))                     
                data_1 = Parallel(n_jobs=-1,pre_dispatch='2*n_jobs', mmap_mode="r+")(delayed(np.nanstd)(data0[segment]) for segment in range(n_seg))  
                profiles.append(list(data_0))
                profiles.append(list(data_1))
                profiles.append(list(data[1]))
                profiles.append(list(data[2]))
                profiles.append(list(data[3])) 
                profiles.append(list(bund_dir)) 
                profiles.append(list(lengths_))  
                
                clean_path = f_path + "subjects/%s/tracking/AFQ/temp"%p + mod + "/" 
                if not os.path.exists(clean_path):
                    os.mkdir(clean_path)
                    
                tracks_filename = clean_path + bundle_name # + ".trk"
                trk2 = StatefulTractogram.from_sft(this_sl, trk) 
                # save_tck(trk2, tracks_filename + ".tck")  ;  trk2 = None 
                """
                if not os.path.isdir("/home/users/d/r/drimez/AFQ_data/samples/"):
                    os.mkdir("/home/users/d/r/drimez/AFQ_data/samples/") 
                """
                this_sl = StatefulTractogram.from_sft(this_sl,trk)
                this_sl.to_vox()
                """
                print(this_sl.streamlines[0],file=open("/home/users/d/r/drimez/sl.txt","w"))
                print(this_sl.affine,file=open("/home/users/d/r/drimez/sl.txt","a"))
                """
                empty = False 
                def one_scalar_profile(ii, scalar, scalar_file, n_seg=n_seg, assignments_=assignments_, fa_mask=None,
                                       this_sl=this_sl.streamlines, affine_=this_sl.affine, return_fa_mask=False):
                    scalar_data_ = nib.load(scalar_file)
                    scalar_data = scalar_data_.get_fdata() 
                    """
                    this_profile.append(afq_profile( scalar_data,
                                                     this_sl,
                                                     dwi_affine,
                                                     weights=this_prof_weights ))
                    """ 
                    #################               
                    # this_trk_bundle = load_trk(tracks_filename, f_path + "subjects/%s/dMRI/preproc/"%p + p + "_dmri_preproc.nii.gz")
                    # fbc_sl_thres, rfbc_orig, rfbc_thres = fib_coherence(this_trk_bundle,threshold=0.1)
                    # save_tck(fbc_sl_thres,tracks_filename.split(".")[0]+".tck")     
                    # save_trk(fbc_sl_thres,tracks_filename.split(".")[0]+".trk")  
                    # save_tck(this_trk_bundle,tracks_filename.split(".")[0]+".tck") 
                    """
                    if not os.path.isdir("/auto/home/users/d/r/drimez/AFQ_data/samples/" + p + "/"):
                        os.mkdir("/auto/home/users/d/r/drimez/AFQ_data/samples/" + p + "/")
                    sample_cmd  = "tcksample -force -debug " + tracks_filename + ".tck " + scalar_file + " /auto/home/users/d/r/drimez/AFQ_data/samples/" + p + "/" + bundle_name + "_" + scalar + ".txt" 
                    bashcmd = sample_cmd.split()  
                    process = subprocess.Popen(sample_cmd, universal_newlines=True, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
                    # wait until finish
                    out, error = process.communicate()  
                    """
                    data = np.zeros((3,n_seg,))  
                    # counter = len(this_sl) ; 
                    values = np.array([[ np.nan for nn_sl in range(len(this_sl))] for nn_seg in range(n_seg)]).astype(np.float64) ; these_streamlines = None
                    if return_fa_mask:
                        fa_mask = np.array([[ np.nan for nn_sl in range(len(this_sl))] for nn_seg in range(n_seg)])
                    """
                    try:
                        with open("/auto/home/users/d/r/drimez/AFQ_data/samples/" + p + "/" + bundle_name + "_" + scalar + ".txt", "r") as reader: 
                            these_streamlines = reader.readlines()
                            reader.close()
                    except Exception:
                        empty = True
                        break
                    """
                    these_streamlines = values_from_volume(scalar_data, list(this_sl), np.eye(4))
                    # if len(these_streamlines[1:])>len(this_sl):
                    #    these_streamlines = these_streamlines[:-1]
                    for n_sl, stream_line in enumerate(these_streamlines): # reads streamlines one by one [1:]
                         
                        # stream = [float(point) for point in stream_line.split("\n")[0].split()] 
                        stream = stream_line
                        """
                        if seglen<=(seg_len_mm/step_len) and False: # less than 1 point per segment is not normal
                            print("less than %s points for %smm segments, segment length is reduced ==> this streamline will surely not \
                                   participate to all segments"%((seg_len_mm/step_len)-2,seg_len_mm)) 
                        """
                        try:
                            this_sl_assign = np.array(assignments_[n_sl])
                            unique, counts = np.unique(this_sl_assign,return_counts=True)
                            unique = unique.astype(int)
                            for n_p, point in enumerate(stream): 
                                seg_id = this_sl_assign[n_p] 
                                if return_fa_mask:
                                    if point>=0.1:
                                        fa_mask[seg_id][n_sl] = True
                                    else:
                                        fa_mask[seg_id][n_sl] = False
                                to_use = True
                                if (not np.isnan(fa_mask[seg_id][n_sl])) and (fa_mask[seg_id][n_sl]==False):
                                    to_use=False
                                if to_use:
                                    if np.isnan(values[seg_id][n_sl]):
                                        values[seg_id][n_sl] = np.nan_to_num(point/counts[unique==int(seg_id)])
                                    else:
                                        values[seg_id][n_sl] += np.nan_to_num(point/counts[unique==int(seg_id)]) 
                        except Exception: 
                            pass
                            # data[0,seg_id] += np.nan_to_num(this_prof_weights[n_sl,seg_id]*float(point)/seglen) 
                        """  
                        mean_data = data[0].flatten() 
                        for n_p, point in enumerate(stream):
                            seg_id = min(int(np.rint(np.nan_to_num(n_p/seglen))),n_seg-1) 
                            data[1,seg_id] += (1/(counter-1))*( mean_data[seg_id] - np.nan_to_num(this_prof_weights[n_sl,seg_id]*float(point)/seglen) )**2 
                        """   
                    """    
                    data[0] = Parallel(n_jobs=-1,pre_dispatch='n_jobs',verbose=10, mmap_mode="r+")( 
                                       delayed(bootstrap)(values, np.mean, True) for segment in np.arange(n_seg) 
                                       )                      
                    data[1] = Parallel(n_jobs=-1,pre_dispatch='n_jobs',verbose=10, mmap_mode="r+")( 
                                       delayed(bootstrap)(values, np.std,True) for segment in np.arange(n_seg) 
                                       ) 
                    """
                    print("",file=open("/auto/home/users/d/r/drimez/afq.txt","w"))
                    for val in values[0]:
                        print(val,file=open("/auto/home/users/d/r/drimez/afq.txt","a"))
                    data[0] = Parallel(n_jobs=-1,pre_dispatch='2*n_jobs', mmap_mode="r+")(delayed(bootstrap)(values[segment], np.nanmean) for segment in range(n_seg))                   
                    data[1] = Parallel(n_jobs=-1,pre_dispatch='2*n_jobs', mmap_mode="r+")(delayed(np.nanstd)(values[segment]) for segment in range(n_seg)) 
                    """
                    try:
                        os.system("rm "+ "/auto/home/users/d/r/drimez/AFQ_data/samples/" + p + "/" + bundle_name + "_" + scalar + ".txt") 
                    except Exception:
                        pass
                    """
                    ########## 
                    print(data[0],file=open("/home/users/d/r/drimez/sl.txt","a"))
                    # profiles.append(list(this_profile[ii]))
                    if return_fa_mask:
                        return fa_mask, list(data[0]), list(data[1])
                    else:
                        return list(data[0]), list(data[1])
                  
                print("Sampling microstructure metrics")
                fa_mask, mean_data, std_data = one_scalar_profile(0, "FA", scalar_dict["FA"], return_fa_mask=True) 
                sampled_datas = Parallel(n_jobs=3,pre_dispatch='n_jobs', mmap_mode="r+")(delayed(one_scalar_profile)(ii, scalar, scalar_file, fa_mask=fa_mask) 
                                                            for ii, (scalar, scalar_file) in enumerate(scalar_dict.items()) if scalar!="FA")  
                profiles.append(list(mean_data))
                profiles.append(list(std_data))
                for mean_data, std_data in sampled_datas: 
                    profiles.append(list(mean_data))
                    profiles.append(list(std_data))
                 
                nodes = list(np.arange(n_seg))
                bundle_names.extend([bundle_name] * len(nodes))
                node_numbers.extend(nodes) 
                # os.system("rm " + tracks_filename + ".tck") 
                
                
                if parrallel and not empty:
                    if not os.path.exists(f_path + "subjects/%s/tracking/AFQ/temp"%p +mod+"/"):
                        os.mkdir(f_path + "subjects/%s/tracking/AFQ/temp"%p +mod+"/")
                    # joblib.dump([profiles, bundle_names, node_numbers],f_path + "subjects/%s/tracking/AFQ/temp"%p +mod+"/"+bundle_name+".txt")
                    print(profiles, bundle_names, node_numbers,file=open("/auto/home/users/d/r/drimez/afq.txt","a"))
                    return profiles, bundle_names, node_numbers
                elif empty:
                    return np.zeros((15,1)), np.array([bundle_name]), np.array([0])
            else:
                return np.zeros((15,1)), np.array([bundle_name]), np.array([0])
        else:
            return np.zeros((15,1)), np.array([bundle_name]), np.array([0])
    
    print("Launching features sampling")  
    trk_names = [] ; bundle_name_list = []
    with os.scandir(clean_bundles_folder) as folder_it:
        for entry in folder_it:
            try:
                in_afq = np.any([len(entry.name.split(_))==2 for _ in np.append(BUNDLES, CALLOSUM_BUNDLES)])
                if entry.name.split(".")[-1]=="trk":
                    trk_names.append(entry.path)
                    bdle_name = entry.name.replace(".trk","")
                    if bdle_name[0] == "_":
                        bdle_name = bdle_name[1:]
                    bundle_name_list.append(bdle_name)
            except Exception:
                if entry.name.split(".")[-1]=="trk":
                    trk_names.append(entry.path)
                    bdle_name = entry.name.replace(".trk","")
                    if bdle_name[0] == "_":
                        bdle_name = bdle_name[1:]
                    bundle_name_list.append(bdle_name)
        folder_it.close() 
        
    results = Parallel(n_jobs=-1, pre_dispatch='2*n_jobs', verbose=600, mmap_mode="r+")( 
                        delayed(single_profile)(b,bn,True) for b,bn in zip(trk_names,bundle_name_list) 
                      )  
                      
    profiles, bundle_names, node_numbers = None, None, None  
    """ 
    for b in trk_uids:
        pro, bdln, nodn = joblib.load( f_path + "subjects/%s/tracking/AFQ/temp"%p +mod+"/"+reverse_dict[b]+".txt", 
                                       mmap_mode="r+")
        os.system("rm " + f_path + "subjects/%s/tracking/AFQ/temp"%p +mod+"/"+reverse_dict[b]+".txt")
        if profiles is None:
            profiles, bundle_names, node_numbers = pro, bdln, nodn
        else: 
            profiles = np.concatenate((profiles,pro),axis=1)
            bundle_names = np.append(bundle_names,bdln) 
            node_numbers = np.append(node_numbers,nodn) 
    """ 
    for pro, bdln, nodn in results:  
        if profiles is None:
            profiles, bundle_names, node_numbers = np.array(pro), bdln, nodn
        else: 
            try:
                profiles = np.concatenate((profiles,np.array(pro)),axis=1)
                bundle_names = np.append(bundle_names,bdln) 
                node_numbers = np.append(node_numbers,nodn)   
            except Exception:
                print(pro,file=open("/auto/home/users/d/r/drimez/afq_ERR.txt","w"))

    profile_dict = dict()
    profile_dict["subjectID"] = np.full_like(bundle_names,p)
    profile_dict["tractID"] = bundle_names
    profile_dict["nodeID"] = node_numbers
     
    features = np.array([ [sd_key, "std_"+sd_key] for sd_key in scalar_dict.keys() ])
    features = np.append(["curvature (mean)","curvature (std)","diameter_x","diameter_y","diameter_2","bundle_dir","length"],features.flatten())
    for ii, scalar in enumerate(features):
        profile_dict[scalar] = profiles[ii]

    # profile_dframe = pd.DataFrame(profile_dict)
    # meta = dict(source=clean_bundles_file, parameters=get_default_args(afq_profile)) 

    return profile_dict, None
""" 
def wmql_trad(bdle_name):
    
    data_query = None
    with open("/auto/home/users/d/r/drimez/wmql_tracts.qry", "r") as reader: 
        data_query = reader.readlines() 
        reader.close()
    
    wmql_name = Natbrain2wmql[bdle_name]
    for line in data_query:
        if line.split(" ")[0].split('.')[0]==wmql_name: # we found the definition of the bundle
"""       
def def_bundles(bundles,updates=None,p=None,natbrain=True):
 
    bdles = ["ATR_L","ATR_R","CGC_L","CGC_R","CST_L","CST_R","IFO_L","IFO_R",
             "ILF_L","ILF_R","SLF_L","SLF_R","ARC_L","ARC_R","UNC_L","UNC_R"] 
    bdle_cat = "_natbrain" if natbrain else ""
     
    defaults_updates = {} # impose no common end regions in CST for ex
    for key_bname in bdles:
    
         hem = key_bname[-1:]
         bund = key_bname[:-1] 
         if hem=="L":
             target = 1 # right hemisphere labels
         else:
             target = 2 # lef hemispher labels 
             
         hemisphere = nib.load("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/atlases_dany/label/Human/Hemispheric_space-MNI152NLin6_res-1x1x1.nii.gz")
         img = hemisphere.get_fdata()
         img[img!=target] = 0
         img[img!=0] = 1
         target_roi = nib.Nifti1Image(img, header=hemisphere.header, affine=hemisphere.affine)  
         
         defaults_updates[key_bname] = { "exclude":       {"replace":False, 
                                                           "update":target_roi}   }
         """
                                         "cross_midline": {"replace":True, 
                                                           "update":True}         }
         """
    
    if not (updates is None):      
        updates = dict(defaults_updates,**updates)
    else:
        updates = defaults_updates
          
    if natbrain:
        dire = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/%s/tracking/preproc/"%p 
        density_map_files = [] 
        density_path = "/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/"%p
        with os.scandir(density_path) as my_it:
            for an_entry in my_it:
                density_map_files.append(density_path + an_entry.name)
    
        Natbrainlab_bdl = {} 
        for prob_map_path in density_map_files:
            AFQ = False
            bundle_name = prob_map_path.split("/")[-1].split(".")[0]
            if bundle_name == "Callosum_midsag" or len(bundle_name.split('_'))==1:
                pass
            elif "R"==bundle_name.split('_')[1] or "L"==bundle_name.split('_')[1]:
                bundle_name = '_'.join(bundle_name.split('_')[:2])
                AFQ = True
            elif "map"==bundle_name.split('_')[-1]:
                bundle_name = bundle_name.split('_')[1]
                AFQ = True
                
            prob_map = nib.load(prob_map_path)
            
            does_cross_midline = np.logical_and( not "Left" in bundle_name.split("_"), 
                                                 not "Right" in bundle_name.split("_") ) 
            if not AFQ:
                Natbrainlab_bdl[bundle_name] = {"cross_midline": does_cross_midline,
                                                "exclude": [],
                                                "include": [dire+"/"+p+"_wm_dil.nii.gz"],
                                                "prob_map": prob_map,
                                                "start": [None],
                                                "end": [None],
                                                "space": "subject" }
            else:
                pass
                """
                Natbrainlab_bdl[bundle_name] = bundles[bundle_name]
                """
            
        bundles = Natbrainlab_bdl
 
 
    UIDs = {"bundle":[],"uid":[]} ; uid_indx = 1
    for bundle, rois in bundles.items(): # update in updates.items(): 
    
        if bundle in [*updates.keys()]: 
            update = updates[bundle]
            for up_key, up_val in update.items():
                kind = up_val["replace"]
                if kind:
                    bundles[bundle][up_key] = up_val["update"]
                else:
                    bundles[bundle][up_key] = np.append(bundles[bundle][up_key],up_val["update"]).tolist()
                    
        UIDs["bundle"].append(bundle)
        UIDs["uid"].append(uid_indx)  
        bundles[bundle]['uid'] = uid_indx
        uid_indx += 1
        
    pd.DataFrame(UIDs).to_csv("/home/users/d/r/drimez/UIDs%s.txt"%bdle_cat,sep="\t",index=False)
    
    return bundles
   
from dipy.tracking.streamline import select_random_set_of_streamlines 

outs = None
def afq(f_path=my_f_path, patient_list="H_0", **kwargs):

    patient = patient_list
    if isinstance(patient_list,list):
        patient = patient_list[0]

    T1 = f_path + "subjects/" + patient + "/T1/" + patient + "_T1_corr_projected.nii.gz"
    MNI = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/atlases_dany/reference_brains/MNI152NLin6_res-1x1x1_T1w_descr-brain.nii.gz"
    this_path = f_path + "subjects/" + patient + "/tracking/preproc_atlas/"

    if not os.path.isdir(this_path):
        os.mkdir(this_path)
    
    if not os.path.exists(this_path + patient + "_AAL_affreg.nii.gz"):
        lut = pd.read_csv("/auto/home/users/d/r/drimez/LUT.txt",sep="  ",header=None,index_col=False,names=["id","name","r","g","b","a"])
        ctx = list([i_ for i_, _ in zip(lut.values[:,0],lut.values[:,1]) if ("ctx"==_.split('-')[0] or 'ctx'==_.split("_")[0])]) + list([3,8,42,47])  
        wmparc, affine = load_nifti(f_path + "subjects/" + patient + "/tracking/preproc/%s_reg_wmparc.nii.gz"%patient)
        
        new_wmparc = np.zeros_like(wmparc)
        for c_ in ctx:
            new_wmparc[wmparc==c_] = c_
            
        save_nifti(this_path + patient + "_AAL_affreg.nii.gz", new_wmparc, affine)
        
    p = patient
    """
    trk = load_trk("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset/PROJECT/subjects/V_47/tracking/V_47_dipy_prob.trk",
                   f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(p,p))
    Lens = length(trk.streamlines)>2
    new_trk = StatefulTractogram.from_sft(list(trk.streamlines[Lens]), trk) 
    save_trk(new_trk,"/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset/PROJECT/subjects/V_47/tracking/V_47_dipy_prob.trk")
    """
    """
    # translation + rotation
    flir_cmd = "flirt -in {T1} -ref {mni} -out {path}/{p}_T1_affreg_6.nii.gz -omat {path}/{p}_affreg_6.mat -dof 6 ; flirt -in {path}/{p}_T1_affreg_6.nii.gz -ref {mni} -out {path}/{p}_T1_affreg_9.nii.gz -omat {path}/{p}_affreg_9.mat -dof 9 ; flirt -in {path}/{p}_T1_affreg_9.nii.gz -ref {mni} -out {path}/{p}_T1_affreg_12.nii.gz -omat {path}/{p}_affreg_12.mat -dof 12 ; ".format(T1=T1,mni=MNI,path=this_path,p=patient)
    
    # concatenates transforms   
    flir_cmd += "convert_xfm -omat {path}/{p}_inv_affreg9to12.mat -concat {path}/{p}_affreg_12.mat {path}/{p}_affreg_9.mat ; convert_xfm -omat {path}/{p}_inv_affreg.mat -concat {path}/{p}_inv_affreg9to12.mat {path}/{p}_affreg_6.mat ; ".format(p=patient,path=this_path)
    
    # invert transform and apply it to AAL atlas
    flir_cmd += "convert_xfm -omat {path}/{p}_affreg.mat -inverse {path}/{p}_inv_affreg.mat ; flirt -in {mni} -ref {T1} -out {path}/{p}_AAL_affreg.nii.gz -init {path}/{p}_affreg.mat -omat {path}/{p}_affreg.mat -dof 12 -interp nearestneighbour -applyxfm".format(T1=T1,mni=MNI,path=this_path,p=patient)
    
      
    # deformable registration 
    flir_cmd += "antsRegistration --float 1 --dimensionality 3 --use-histogram-matching 1 --transform SyN[0.2,3,0] --convergence [150x100x50,1e-6,10] --shrink-factors 8x4x2 --smoothing-sigmas 4x3x2mm  --metric CC[{mni},{T1}] --initial-moving-transform {path}/{p}_affreg.tfm --output [{path}/{p}_MNI_antswarpreg,{path}/{p}_MNI_antswarpreg.nii.gz] --write-composite-transform 1 -v ; antsApplyTransforms --reference-image {T1} --input {mni} --output {path}/{p}_AAL_antswarpreg.nii.gz --transform {path}/{p}_MNI_antswarpregInverseComposite.h5 --interpolation GenericLabel".format(T1=T1,mni=MNI,path=this_path,p=patient)
    
    flir_cmd.split()  
    process = subprocess.Popen(flir_cmd, universal_newlines=True, shell=True, stdout=sys.stdout, stderr=sys.stdout) 
    outs, errs = process.communicate()     
    """     
    # sys.setrecursionlimit(1500)
    f_path = my_f_path; p=patient
    for natbrain in [False]: # True
        for mod in ["_dipy_prob","_dipy"]: # ,"",""_dipy_prob
            step = "reco"  
            # ref_image, affine = load_nifti()
            # save_nifti("ref.nii", ref_image, affine)
            # ref_image = nib.load("C:\\Users\\rimez\\OneDrive\\Documents\\TFE\\Datas\\atlases\\reference_brains\\MNI152NLin6_res-1x1x1_T1w.nii.gz")
            # fbval, fbvec = read_bvals_bvecs("H_0.bval","H_0.bvec")
            # fdata, fbval, fbvec = "H_0_dmri_preproc.nii.gz", "H_0.bval", "H_0.bvec"
             
            density_map_files = {}
            if natbrain:
                density_path = "/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/"%p  
                with os.scandir(density_path) as my_it:
                    for an_entry in my_it:
                        if not an_entry.name.split('.')[0].split('_')[-1]=="map":
                            density_map_files[an_entry.name.split('.')[0].split("_")[0]] = density_path + an_entry.name
                        else:
                            nat_name = an_entry.name.split('.')[0].split("_")
                            new_name = "" 
                            if "Left" in nat_name:
                                new_name = "_L"
                                nat_name = nat_name[:-1]
                            elif "Right" in nat_name:
                                new_name = "_R"
                                nat_name = nat_name[:-1]
                            if "Uncinate" in nat_name:
                                new_name = "UNC"+new_name
                            elif "Arcuate" in nat_name:
                                new_name = "ARC"+new_name
                            else:
                                new_name = '_'.join(nat_name)+new_name
                            
                            density_map_files[new_name] = density_path + an_entry.name
            elif False:
                density_path = "/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/"%patient
                """
                for folder in ["/auto/home/users/d/r/drimez/AFQ_data/templates/","/auto/home/users/d/r/drimez/AFQ_data/callosum_templates/"]:
                    with os.scandir(folder) as my_it:
                        for an_entry in my_it:
                            if not ( ("roi1" in an_entry.name.split("_")) or ("roi2" in an_entry.name.split("_")) ):
                                if folder=="/auto/home/users/d/r/drimez/AFQ_data/templates/" or an_entry.name=="Callosum_midsag.nii.gz":
                                    density_map_files['_'.join(an_entry.name.split('_')[:2]).split(".")[0]] = folder + an_entry.name 
                                else:
                                    density_map_files[an_entry.name.split('_')[1].split(".")[0]] = folder + an_entry.name
                """ 
                with os.scandir(density_path) as my_it:
                    for an_entry in my_it:
                        if not ( ("roi1" in an_entry.name.split("_")) or ("roi2" in an_entry.name.split("_")) ):
                            if not an_entry.name=="Callosum_midsag.nii.gz":
                                density_map_files['_'.join(an_entry.name.split('_')[:2]).split(".")[0]] = folder + an_entry.name 
                            else:
                                density_map_files[an_entry.name.split('_')[1].split(".")[0]] = folder + an_entry.name
            
            root = f_path + "subjects/%s/dMRI/preproc/"%p
            fdata = nib.load(root+p+"_dmri_preproc.nii.gz")
            img_affine = fdata.affine
            
            filter_by_endpoints = False if natbrain else True
            prob_threshold = 0.01 if mod=="_dipy" else 0 
            query = True ; refresh = True ; refresh_cleaning = True ; refresh_profile = True
            
            step = "reco" 
            """
            if os.path.exists(f_path+"subjects/%s/tracking/FOD/%s_fod_prob.pbz2"%(p,p)):
                os.system("rm "+f_path+"subjects/%s/tracking/FOD/%s_fod_prob.pbz2"%(p,p))
            """
            # seg_obj = Segmentation(seg_algo="AFQ",prob_threshold=prob_threshold, return_idx=True, dany=False, filter_by_endpoints=filter_by_endpoints)
            bundle_info = np.append(BUNDLES,CALLOSUM_BUNDLES).tolist()
             
            updates = {"UNC_L": {"split": {"replace":True, "update":False}}, 
                       "UNC_R": {"split": {"replace":True, "update":False}},
                       "ILF_L": {"split": {"replace":True, "update":False}}, 
                       "ILF_R": {"split": {"replace":True, "update":False}} }
                       
            ##### bundle extraction 
            fbval, fbvec = f_path+"data_1/%s.bval"%p, f_path+"data_1/%s.bvec"%p
            ref_image = None
            if natbrain:
                ref_image = copy.copy(fdata)
            else: 
                ref_image = nib.load("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/atlases_dany/reference_brains/MNI152NLin6_res-1x1x1_T1w.nii.gz")
        
            ref_bundles = BundleDict(bundle_info=bundle_info,resample_to=ref_image,seg_algo="AFQ") # RECO_BUNDLES_80
            ref_bundles.gen_all()
            
            ref_bundles_dict = def_bundles(ref_bundles._dict,p=p,natbrain=natbrain,updates=updates)
            base_bundles_dict = copy.copy(ref_bundles_dict)
            df = pd.DataFrame(ref_bundles_dict) 
            lines = ["space"]
            df.drop(lines,axis=0).to_csv("/home/users/d/r/drimez/bundles.txt",sep="\t") 
            min_fib = {"":None} ; query_fiber_groups = None
                 
            if step=="reco" and query: 
                    
                def load_query(single_query=None,side=None,ref_bundles_dict=None,f_path=f_path,refresh=refresh,mod=mod,p=p):
                    model = "" if mod=="_dipy" else "_prob"
                    force = True
                    if os.path.exists(f_path+"subjects/%s/tracking/query/"%p +mod[1:]+"/"):
                        if not os.listdir(f_path+"subjects/%s/tracking/query/"%p +mod[1:]+"/"): 
                            force = True
                              
                    if ((single_query is None) and ((not os.path.exists(f_path+"subjects/%s/tracking/query/"%p +mod[1:]+"/")) or refresh))\
                        or (single_query is not None) or force:  
                        
                        model = '""' if mod=="_dipy" else "_prob" 
                        seg_cmd = "python wm_query.py %s False %s %s"%(model,single_query,p)
                        seg_cmd.split()
                        process = subprocess.Popen(seg_cmd, universal_newlines=True, shell=True,
                                                   stdout=sys.stdout, stderr=sys.stdout)
                        model = "" if mod=="_dipy" else "_prob"
                        outs, errs = process.communicate() 
                    else:
                        print("Already segmented, loading results\nUse refresh=True to re-segment")
                        
                    dire = f_path + "/subjects/%s/tracking/preproc/"%p 
                    trk_path = f_path+"subjects/%s/tracking/query/"%p +mod[1:]+"/" if single_query is None \
                               else f_path+"subjects/%s/tracking/Solo/query%s/"%(p,model)
                    fiber_groups = {} ; query_counter = 99
                    with os.scandir(trk_path) as my_it:
                        for an_entry in my_it:
                            if an_entry.name.split('.')[-1] == "trk": 
                                trk = load_trk(an_entry.path, root+p+"_dmri_preproc.nii.gz") 
                                bdle_name = ".".join(an_entry.name.split(mod)[-1].split(".")[:-1]) 
                                fiber_groups[bdle_name] = {}
                                fiber_groups[bdle_name]['sl'] = trk 
                                
                                if not (ref_bundles_dict is None):
                                    if not bdle_name in [*ref_bundles_dict.keys()]:
                                        query_counter += 1 ; probmab = None
                                        if len(an_entry.name.split('cp'))==2:
                                            """
                                            if len(an_entry.name.split('.left'))==2:
                                                prob_map1, affine = load_nifti("/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/Inferior_Cerebellar_Pedunculus_Left.nii.gz"%p)
                                                prob_map2, _ = load_nifti("/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/Superior_Cerebelar_Pedunculus_Left.nii.gz"%p)
                                                prob_map3, _ = load_nifti("/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/Cortico_Ponto_Cerebellum_Right.nii.gz"%p)
                                                probmab = prob_map1+prob_map2+prob_map3
                                                probmab[probmab>=1] = 1 
                                                save_nifti("/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/Cerebellum_Left.nii.gz"%p, probmab, affine)
                                                probmab = "/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/Cerebellum_Left.nii.gz"%p 
                                            elif len(an_entry.name.split('.right'))==2:
                                                prob_map1, affine = load_nifti("/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/Inferior_Cerebellar_Pedunculus_Right.nii.gz"%p)
                                                prob_map2, _ = load_nifti("/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/Superior_Cerebelar_Pedunculus_Right.nii.gz"%p)
                                                prob_map3, _ = load_nifti("/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/Cortico_Ponto_Cerebellum_Left.nii.gz"%p) 
                                                probmab = prob_map1+prob_map2+prob_map3
                                                probmab[probmab>=1] = 1 
                                                save_nifti("/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/Cerebellum_Left.nii.gz"%p, probmab, affine)
                                                probmab = "/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/Cerebellum_Right.nii.gz"%p 
                                            """    
                                            ref_bundles_dict[bdle_name] = { "cross_midline": False,
                                                                            "exclude": [],
                                                                            "include": [dire+"/"+p+"_wm_dil.nii.gz"],
                                                                            "prob_map": probmab,
                                                                            "start": [dire+"/"+p+"_wm_dil.nii.gz"],
                                                                            "end": [dire+"/"+p+"_wm_dil.nii.gz"],
                                                                            "space": "subject",
                                                                            "split":False,
                                                                            "query":True,
                                                                            "uid": query_counter}
                                                                            
                                        else: # len(an_entry.name.split('striatum'))==2 or len(an_entry.name.split('thalamo'))==2: 
                                            ref_bundles_dict[bdle_name] = { "cross_midline": False,
                                                                            "exclude": [],
                                                                            "include": [dire+"/"+p+"_wm_dil.nii.gz"],
                                                                            "prob_map": dire+"/"+p+"_wm_dil.nii.gz",
                                                                            "start": [dire+"/"+p+"_wm_dil.nii.gz"],
                                                                            "end": [dire+"/"+p+"_wm_dil.nii.gz"],
                                                                            "space": "subject",
                                                                            "split":True,
                                                                            "query":True,
                                                                            "uid": query_counter}  
                                
                        my_it.close()   
                         
                    return fiber_groups, ref_bundles_dict
            
                query_fiber_groups, ref_bundles_dict = load_query(ref_bundles_dict=ref_bundles_dict)
            """
            if not os.path.exists(f_path+"subjects/%s/tracking/%s"%(p,p)+mod+".trk"):
                step = "profile"   
            """  
            if step=="reco": 
                ##### bundle cleaning #####
                if not os.path.isdir(f_path+"subjects/%s/tracking/AFQ/"%p):
                    os.mkdir(f_path+"subjects/%s/tracking/AFQ/"%p) 
                if not os.path.isdir(f_path+"subjects/%s/tracking/AFQ/tracks"%p +mod+"/"):
                    os.mkdir(f_path+"subjects/%s/tracking/AFQ/tracks"%p +mod+"/")
                    
                """
                if not os.path.exists(f_path+"subjects/%s/tracking/AFQ/tracks"%p +mod+"/"+"all.pbz2") or refresh:
                    print("Running Segmentation for " + mod + " tracts")
                    trk = load_trk(f_path+"subjects/%s/tracking/%s"%(p,p)+mod+".trk", root+p+"_dmri_preproc.nii.gz")
                    # trk.to_vox()
                    fiber_groups = seg_obj.segment( base_bundles_dict, trk, fdata=fdata, fbval=fbval, fbvec=fbvec, 
                                                    reg_template=ref_image, img_affine=img_affine ) # reg_template=ref_image, 
                    compressed_pickle(f_path+"subjects/%s/tracking/AFQ/tracks"%p +mod+"/"+"all",fiber_groups)  
                else:
                    print("Already segmented, unpickeling results\nUse refresh=False to re-segment")
                    fiber_groups = decompress_pickle(f_path+"subjects/%s/tracking/AFQ/tracks"%p +mod+"/"+"all")
                """  
                
                fiber_groups = {}
                cleaned_bundles = {} ; bundle_dict = None 
                if not (query_fiber_groups is None):
                    for name, fib_dict in query_fiber_groups.items(): 
                        fiber_groups[name] = fib_dict
                        fiber_groups[name]["idx"] = None
                
                def add_bundle_dict(parent,news,mod=mod,p=p,f_path=f_path): 
                    if parent is None:
                        parent = news
                    elif news is None:
                        parent = parent
                    else: 
                        for name_, new_d in news.items():
                            cleaned_tg = None  
                            if name_ in [*parent.keys()]:
                                new_ = new_d["sl"]
                                bundle = parent[name_]['sl'] 
                            else:
                                new_ = new_d["sl"]
                                bundle = None
                            try:
                                if len(new_)!=0:
                                    if isinstance(new_[0],list) or isinstance(new_[0],np.ndarray):
                                        if not (isinstance(new_[0][0],list) or isinstance(new_[0][0],np.ndarray)): # it is not a list of streamlines but a single streamline
                                            new_ = [new_]
                                        if bundle is None:
                                            parent[name_] = {"sl":list(new_)}
                                        else:
                                            parent[name_]["sl"] = list(bundle) + list(new_)
                                        if not os.path.exists(f_path + "subjects/%s/tracking/Solo/"%p + p + mod + "_" + name + "_cleaned.trk"):
                                            sft_ = StatefulTractogram( new_, root+p+"_dmri_preproc.nii.gz",Space.RASMM)
                                            # sft_.to_vox()
                                            save_trk(sft_, f_path + "subjects/%s/tracking/Solo/"%p + p + mod + "_" + name + "_cleaned.trk")
                                        else:
                                            old_ = load_trk(f_path + "subjects/%s/tracking/Solo/"%p + p + mod + "_" + name + "_cleaned.trk").streamlines
                                            sft_ = StatefulTractogram( list(old_) + list(new_),
                                                                          root+p+"_dmri_preproc.nii.gz", Space.RASMM)
                                            # sft_.to_vox()
                                            save_trk( sft_, f_path + "subjects/%s/tracking/Solo/"%p + p + mod + "_" + name + "_cleaned.trk" )
                                # save_trk(new_, f_path+"subjects/%s/tracking/AFQ/tracks"%(p)+mod+"/"+name+".trk")
                                # compressed_pickle(f_path+"subjects/%s/tracking/AFQ/tracks"%(p)+mod+"/bundle_dict_"+name,bundle_dict_)
                            except Exception as err:
                                print("Couldn't save bundle " +name)
                                print(err)  
                    return parent
                  
                def extend_bdl(name,to_extend,tol=1000,min_fib=min_fib,mod=mod,bundle_info=bundle_info,ref_image=ref_image, fdata=fdata, fbval=fbval, fbvec=fbvec,refresh=refresh,
                               ref_bundles_dict=ref_bundles_dict,base_bundles_dict=base_bundles_dict,ref_bundles=ref_bundles, img_affine=img_affine,filter_by_endpoints=filter_by_endpoints):
                    global outs
                           
                    bundle = to_extend 
                    model = '\"\"' if mod=="_dipy" else "_prob"
                    new_seg_obj = Segmentation( seg_algo="AFQ",prob_threshold=0, return_idx=True, 
                                                dany=False, filter_by_endpoints=filter_by_endpoints ) #, parallel_segmentation={"engine": "serial"}) 
                    single_fib_groups = None
                    iterations = 0 ; max_iter = 10 ; new_bundle = copy.copy(bundle) ; start_time = time.time() ; now=time.time()
                    new_bundle_path = f_path + "subjects/%s/tracking/Solo/"%p + p + "_dipy" + model + "_" + name ; trk = None
                    if os.path.exists(new_bundle_path + ".trk"): # outs and not (outs is None):     
                        trk = load_trk(new_bundle_path + ".trk", root+p+"_dmri_preproc.nii.gz")
                        single_fib = new_seg_obj.segment( base_bundles_dict, trk, fdata=fdata, fbval=fbval,  
                                                          fbvec=fbvec, reg_template=ref_image, img_affine=img_affine )
                        single_fib_groups = add_bundle_dict(single_fib_groups,single_fib) 
                                
                        if query:
                            unselected = np.arange(len(trk.streamlines)).tolist()
                            for _, selected_val in single_fib.items():
                                selected__ = selected_val["idx"]
                                for s___ in selected__:
                                    unselected.remove(s___) 
                            sft_ = StatefulTractogram(trk.streamlines[unselected],
                                                         root+p+"_dmri_preproc.nii.gz",Space.RASMM)
                            # sft_.to_vox()
                            save_trk( sft_, new_bundle_path + ".trk")
                                      
                            single_fib, ref_bundles_dict = load_query(single_query=True,side=None,ref_bundles_dict=ref_bundles_dict,f_path=f_path,refresh=refresh)
                            
                            single_fib_groups = add_bundle_dict(single_fib_groups,single_fib)  
                        
                        os.system("rm " + new_bundle_path + ".trk")  
                    elif os.path.exists(new_bundle_path + "_cleaned.trk"): # outs and not (outs is None):     
                        trk = load_trk(new_bundle_path + "_cleaned.trk", root+p+"_dmri_preproc.nii.gz") 
                        new_bundle = list(bundle) + list(trk.streamlines)
                        
                    unselected = []    
                    while len(new_bundle)<=tol+200 and iterations<=max_iter and (now-start_time)<=int(5*60) and False: 
                         
                        iterations += 1 
                        new_bundle_path = f_path + "subjects/%s/tracking/Solo/"%p + p + "_dipy" + model + "_" + name  
                        model = '\"\"' if mod=="_dipy" else "_prob"
                        outs = None
                        def my_tracto(outs=outs,name=name,p=p,model=model,f_path=f_path,ref_bundles_dict=ref_bundles_dict,density_map_files=density_map_files,base_bundles_dict=base_bundles_dict): 
                            this_density_map = None ; shoud_I_run = True
                            if name in [*density_map_files.keys()]:
                                this_density_map = [density_map_files[name]]
                            else:
                                try:
                                    this_density_map = base_bundles_dict[name]["prob_map"] 
                                    temp_img, temp_affine = load_nifti(this_density_map)
                                    temp_img = new_seg_obj.mapping.transform_inverse(temp_img)
                                    save_nifti("/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/"%p + name + ".nii.gz",temp_img,nib.load(root+p+"_dmri_preproc.nii.gz").affine)
                                    this_density_map = ["/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/"%p + name + ".nii.gz"]
                                except Exception:    
                                    shoud_I_run = False
                                    this_density_map = ["/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/None.nii.gz"%p]
                                    temp_img, temp_affine = load_nifti(root+p+"_dmri_preproc.nii.gz")
                                    temp_img[temp_img!=0] = 1
                                    save_nifti(this_density_map[0],temp_img[...,0],temp_affine) 
                            temp_img, temp_affine = load_nifti(this_density_map[0])
                            if not temp_img.shape[0]==128:
                                temp_img = new_seg_obj.mapping.transform_inverse(temp_img)
                                save_nifti(this_density_map[0],temp_img,nib.load(root+p+"_dmri_preproc.nii.gz").affine)
                            if shoud_I_run:
                                outs = dipy_tracto(p,n_iterations=2,model=model,f_path=f_path,density_map_files=this_density_map[0],fod=False,single_bdl=name,
                                                   filtering=False, postrack=False, resp=False) 
                        thread = threading.Thread(target=my_tracto)
                        thread.start() 
                        # wait here for the result to be available before continuing
                        thread.join()                   
                        if outs and not (outs is None):     
                            if name in [*ref_bundles_dict.keys()]:
                                new_bundle_path = f_path + "subjects/%s/tracking/Solo/"%p + p + "_dipy" + model + "_" + name
                                trk = load_trk(new_bundle_path + ".trk", root+p+"_dmri_preproc.nii.gz")
                                
                                single_fib = new_seg_obj.segment( base_bundles_dict, trk, fdata=fdata, fbval=fbval, 
                                                                  fbvec=fbvec, reg_template=ref_image, img_affine=img_affine ) 
                                
                                single_fib_groups = add_bundle_dict(single_fib_groups,single_fib) 
                                
                                if query:
                                    unselected_indx = np.arange(len(trk.streamlines)).tolist()
                                    for _, selected_val in single_fib.items():
                                        selected__ = selected_val["idx"]
                                        for s___ in selected__:
                                            unselected_indx.remove(s___) 
                                    
                                    unselected = list(unselected) + list(trk.streamlines[unselected]) 
                            else:
                                print("Bundle is not in bundle dict") 
                        now = time.time()  
                        
                    if len(unselected)>1:
                        sft_ = StatefulTractogram(unselected,root+p+"_dmri_preproc.nii.gz",Space.RASMM)
                        # sft_.to_vox()
                        save_trk( sft_, new_bundle_path + ".trk") 
                        single_fib, ref_bundles_dict = load_query(single_query=True,side=None,ref_bundles_dict=ref_bundles_dict,f_path=f_path,refresh=refresh) 
                        single_fib_groups = add_bundle_dict(single_fib_groups,single_fib)  
                        os.system("rm " + new_bundle_path + ".trk") 
                        trk = load_trk(new_bundle_path + "_cleaned.trk", root+p+"_dmri_preproc.nii.gz") 
                        new_bundle = list(bundle) + list(trk.streamlines)
                    
                    return single_fib_groups
                                        
                def single_bdl_cleaning(name,bundle,min_fib=min_fib,cleaned_bundles=None,mod=mod):
                
                    print("Cleaning launched for " + name)  
                    org_index = np.arange(len(bundle))
                    
                    lengths = length(bundle.streamlines)
                    standard = bundle.streamlines[np.argmin(abs(lengths-np.quantile(lengths,0.625)))] 
                    bdl_to_clean = StatefulTractogram.from_sft(orient_by_streamline(bundle.streamlines, standard, n_points=100),bundle)  
                    to_clean = None ; data_per_streamline=None 
                    
                    print("Cleaning") 
                    cleaned_bundle, clean_idx = seg_clean_bundle(bdl_to_clean, length_threshold=5, return_idx=True)   
                    cleaned_bundle.to_rasmm()
                    cleaned_indx = [clean for clean in clean_idx if (clean in org_index)] # keeps only probalistic fibers if dual filtering
                    """
                    selected = np.array(sl_ids)[clean_idx] # ids corresponding to cleaned fibers of this bundle
                    selected = selected.tolist()
                    """
                    bundle_dict = {name:ref_bundles_dict[name]}
                    if ("prob" in mod.split("_")) and len(cleaned_bundle)>100:
                        print("Launching second round of cleaning")
                        split = False
                        if "split" in [*ref_bundles_dict[name].keys()] and False:
                            split = ref_bundles_dict[name]["split"] 
                            print("Split set to false for this bundle")
                        this_sl_list, filenames, bundle_dict = scnd_round( bundle.streamlines[cleaned_indx], bundle.streamlines[cleaned_indx], bundle_dict, img_affine, 
                                                                           profile_weights="gauss", f_path=f_path, p=p, mod=mod, split=split)
                    else:
                        this_sl_list, filenames = [cleaned_bundle], [name]
                     
                    cleaned_bundles = None
                    for bundle_name, abundle in zip(filenames,this_sl_list):  
                        if not (cleaned_bundles is None): 
                            cleaned_bundles = StatefulTractogram(  list(cleaned_bundles.streamlines) + list(abundle.streamlines), root+p+"_dmri_preproc.nii.gz", Space.RASMM,
                                                                   data_per_streamline=  {k: list(cleaned_bundles.data_per_streamline[k]) + list(abundle.data_per_streamline[k])  
                                                                                         for k in cleaned_bundles.data_per_streamline.keys()  }  ) 
                        else:
                            try:
                                cleaned_bundles = StatefulTractogram( abundle.streamlines, fdata, Space.RASMM ) 
                            except Exception:
                                cleaned_bundles = StatefulTractogram( list(abundle), fdata, Space.RASMM )  
                    
                    return cleaned_bundles, bundle_dict
                """    
                extended_df = [] ; extended_df_file = None 
                extended = 0 ; extended_idx = -1
                with open("/auto/home/users/d/r/drimez/extends.txt","r") as reader:
                    extended_df_file = reader.readlines()
                    reader.close()
                for iline, line in enumerate(extended_df_file):
                    Line = line.split("\n")[0].split(",")
                    extended_df.append([Line[0],int(Line[1])]) 
                    if Line[0]==p+mod:
                        extended_idx = iline
                        extended = int(Line[1])
                if extended_idx==-1:
                    extended_df.append([p+mod,0]) 
                     
                for ifib, name in enumerate([*fiber_groups.keys()]): 
                    if ifib>=extended:
                        bundle = fiber_groups[name]["sl"]
                        # check if enough fibers in the bundle
                        if name in [*min_fib.keys()]:
                            tol = min_fib[name]
                        else:
                            tol = 200 if mod=="_dipy" else 300
                        
                        if name.split("_")[0]=="ARC":
                            name = "af_"+name.split("_")[1]
                        elif name.split("_")[0]=="UNC":
                            name = "uf_"+name.split("_")[1]
                        elif name.split("_")[0]=="IFO":
                            name = "ifof_"+name.split("_")[1]
                        query_name = "_"+name.split("_")[0].lower()+ ".".join(name.split(".")[1:-1])
                        if query_name in [*fiber_groups.keys()]:
                            tol -= len(fiber_groups[query_name]["sl"])
                            
                        if len(bundle)<=tol and name in bundle_info: 
                            print("Not enough fibers, launching specific tracking for " + name) 
                            news = extend_bdl(name,bundle,tol=tol,min_fib=min_fib,mod=mod)
                            fiber_groups = add_bundle_dict(fiber_groups,news)  
                        extended_df[extended_idx][1] += 1
                        with open("/auto/home/users/d/r/drimez/extends.txt","w") as writer:
                            for line in extended_df:
                                writer.write(str(line[0])+","+str(line[1])+"\n")
                            writer.close() 
                """      
                def single_iter_cleaning(name, fib_dict,mod=mod,refresh_cleaning=refresh_cleaning,f_path=f_path,root=root,p=p):
                    cleaned_tg = None ; bundle_dict_ = None
                    if (not os.path.exists(f_path+"subjects/%s/tracking/AFQ/tracks"%(p)+mod+"/"+name+".trk")) or refresh_cleaning:   
                        # segmented_bdle = list(fib_dict["idx"]) + list(pd.read_csv(f_path+"subjects/%s/tracking/query/"%(p)+mod[1:]+"/"+ p + mod+"_"+name+".csv"))  
                        print(name + "," + str(len(fib_dict["sl"].streamlines)),
                              file=open(f_path+"subjects/%s/tracking/AFQ/n_sl%s.txt"%(p,mod),"a"))
                        new_fib_sl = select_random_set_of_streamlines(fib_dict["sl"].streamlines, 3000)
                        new_fib_sl = StatefulTractogram.from_sft(list(new_fib_sl),fib_dict["sl"]) 
                        org_space = new_fib_sl.space 
                        cleaned_tg, bundle_dict_ = single_bdl_cleaning(name,new_fib_sl)   
                        cleaned_tg = StatefulTractogram.from_sft(list(cleaned_tg.streamlines),new_fib_sl)  
                        # cleaned_tg.to_vox()
                        # print(cleaned_tg.streamlines[0],file=open("/home/users/d/r/drimez/sl.txt","a"))
                        try:  
                            save_trk(cleaned_tg, f_path+"subjects/%s/tracking/AFQ/tracks"%(p)+mod+"/"+name+".trk")
                            # compressed_pickle(f_path+"subjects/%s/tracking/AFQ/tracks"%(p)+mod+"/bundle_dict_"+name,bundle_dict_)
                        except Exception as err:
                            print("Couldn't save bundle " + name)
                            print(err)
                    else:
                        """
                        cleaned_tg = load_trk(f_path+"subjects/%s/tracking/AFQ/tracks"%p +mod+"/"+name+".trk", 
                                              root+p+"_dmri_preproc.nii.gz") 
                        if os.path.exists(f_path+"subjects/%s/tracking/AFQ/tracks"%p +mod+"/bundle_dict_"+name+".pbz2"):
                            bundle_dict_ = decompress_pickle(f_path+"subjects/%s/tracking/AFQ/tracks"%p +mod+"/bundle_dict_"+name) 
                        else:
                            bundle_dict_ = {name:ref_bundles_dict[name]}
                            compressed_pickle(f_path+"subjects/%s/tracking/AFQ/tracks"%(p)+mod+"/bundle_dict_"+name,bundle_dict_)
                        """
                        print("Cleaned file exists \nUse refresh_cleaning = True to re-compute")
                    
                    return None # cleaned_tg, bundle_dict_
                
                print([*fiber_groups.keys()])  
                if os.path.exists(f_path+"subjects/%s/tracking/AFQ/n_sl%s.txt"%(p,mod)):
                    os.system("rm " + f_path+"subjects/%s/tracking/AFQ/n_sl%s.txt"%(p,mod))  
                cleaned_results = Parallel(n_jobs=-1,verbose=500,pre_dispatch='2*n_jobs',
                                           require="sharedmem")( delayed(single_iter_cleaning)(name,fib_dict) 
                                                                   for name, fib_dict in fiber_groups.items() ) 
                """
                bundle_uids = [value__["uid"] for key__, value__ in ref_bundles_dict.items() ]     
                cleaned_tracktogram = None ; bundle_dict = None ; data_per_streamline_uid = []
                for ik, (cleaned_tg, bundle_dict_) in zip(bundle_uids,cleaned_results):
                    if bundle_dict is None:
                        bundle_dict = bundle_dict_
                    else:
                        bundle_dict = dict(**bundle_dict,**bundle_dict_)
                        
                    if cleaned_tracktogram is None:
                        cleaned_tracktogram = cleaned_tg
                        data_per_streamline_uid = (np.ones(len(cleaned_tg.streamlines))*ik).astype(int) 
                    elif not (cleaned_tg is None): 
                        data_per_streamline = {k: list(cleaned_tracktogram.data_per_streamline[k]) + list(cleaned_tg.data_per_streamline[k])
                                                    for k in cleaned_tracktogram.data_per_streamline.keys()}
                        data_per_streamline_uid = np.append( data_per_streamline_uid, 
                                                             (np.ones(len(cleaned_tg.streamlines))*ik) ).astype(int)               
                        cleaned_tracktogram = StatefulTractogram( list(cleaned_tracktogram.streamlines) + list(cleaned_tg.streamlines), root+p+"_dmri_preproc.nii.gz", 
                                                                  Space.RASMM, data_per_streamline=data_per_streamline  ) 
                  """                                                
                ##### visualization
                # cleaned_tracktogram.data_per_streamline['bundle'] = list(data_per_streamline_uid.astype(int))
                """
                affine = ref_image.affine 
                try:
                    
                    save_trk(  cleaned_tracktogram,
                               f_path+"subjects/%s/tracking/AFQ/%s"%(p,p)+mod+".trk"  )
                    
                    pass
                except Exception as err:
                    print(err) 
                """
                # os.system("rm " + f_path+"subjects/%s/tracking/%s"%(p,p)+mod+".trk")
            
            step = "profile"
            if step=="profile":
                skip = False
                if os.path.exists(f_path+"subjects/%s/tracking/AFQ/%s"%(p,p)+mod+".csv") and not refresh_profile:
                    import pathlib
                    import datetime
                    modif_time = datetime.datetime.fromtimestamp(pathlib.Path(f_path+"subjects/%s/tracking/AFQ/"%p + p + mod + ".csv").stat().st_mtime)
                    if not (str(modif_time).split(" ")[0].split("-")[1]=="04" and int(str(modif_time).split(" ")[0].split("-")[2])<10):
                        skip = True
                
                if not skip:
                    ##### bundle profiles
                    _, img_affine = load_nifti(root+p+"_dmri_preproc.nii.gz") 
                    # trk = load_trk(f_path+"subjects/H_0/tracking/H_0_dipy.trk", "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/atlases_dany/reference_brains/MNI152NLin6_res-1x1x1_T1w.nii.gz")
                    dti_path = f_path + "/subjects/%s/dMRI/microstructure/dti/%s_"%(p,p)
                    fa = path=dti_path + "FA.nii.gz" 
                    md = path=dti_path + "MD.nii.gz" 
                    ad = path=dti_path + "AD.nii.gz" 
                    rd = path=dti_path + "RD.nii.gz" 
                    """
                    odi_path = f_path + "/subjects/%s/dMRI/microstructure/noddi/%s_noddi_"%(p,p)
                    fbundle = odi_path + "fbundle.nii.gz" 
                    fintra = odi_path + "fintra.nii.gz" 
                    fextra = odi_path + "fextra.nii.gz" 
                    fiso = odi_path + "fiso.nii.gz" 
                    odi = odi_path + "odi.nii.gz" 
                    """
                    scalar_dict = {"FA":fa,"MD":md,"AD":ad,"RD":rd} # ,"fbundle":fbundle,"fintra":fintra,"fextra":fextra,"fiso":fiso,"odi":odi}
                    # cleaned_bundles = st.tgram_to_bundles(trk, ref_bundles, ref_image)
                    
                    profile_dict, meta = tract_profiles( f_path+"subjects/%s/tracking/AFQ/tracks"%p +mod+"/", {"bundle_dict":ref_bundles_dict}, 
                                                         scalar_dict, img_affine, profile_weights="gauss",f_path=f_path,p=p,mod=mod, ref_path=root+p+"_dmri_preproc.nii.gz")
                    
                    my_to_csv(f_path+"subjects/%s/tracking/AFQ/%s"%(p,p)+mod+"_new.csv",profile_dict) 
                    """
                    os.system("rm " + f_path+"subjects/%s/tracking/query/"%p +mod[1:]+"/")
                    os.system("rm " + f_path+"subjects/%s/tracking/%s"%(p,p)+mod+".trk")
                    """
                    
                """
                df = pd.DataFrame(data=[p for m in range(len(profile_dframe.index.values))],columns=["subjectID"])            
                pd.concat([df,profile_dframe]).to_csv(f_path+"subjects/%s/tracking/AFQ/%s"%(p,p)+mod+".csv", line_terminator="\n", sep=",", index=False)               
                """  
            """        
            if step=="plot":
                indiv_profile = profile_dframe.values ; sft = load_trk(f_path+"subjects/H_0/tracking/AFQ/H_0_dipy"+mod+".trk") 
                single_bundle_viz(indiv_profile, sft, bundle, scalar_name, affine=img_affine, bundle_dict=None, figure=None, include_profile=True)               
            """          
if __name__ == "__main__": 
  
    if len(sys.argv)>1: 
        args = [str(arg) for arg in sys.argv[1:]] 
        afq(f_path=my_f_path,patient_list=str(args[0]))         
    else:     
        afq(f_path=my_f_path,patient="H_0") 
                   
                   
                   
                   
                   
                   
    