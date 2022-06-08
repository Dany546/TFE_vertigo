"""
tensor2metric -adc -fa -ad -rd -cl -cp -cs -value -vector tensor -mask image -debug
population_template -type "rigid_affine" -voxel_size jsp,jsp,jsp input_dir template # input = all input images
tcksample [ options ]  tracks image output_values
values_from_volume(data, streamlines, affine) to sample along trackts
"""
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
from cmtk import *
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import subprocess
import copy as cp 
import pickle, json
import bz2 
import _pickle as cPickle
import sys
import os 
import gc

f_path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/"
f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset_new/PROJECT/"  
f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/elikopy_subset_new2/PROJECT/" 
# f_path = "C:\\Users\\rimez\\OneDrive\\Documents\\TFE\\Datas\\Trash\\subset\\ELIKOPY_subset\\PROJECT\\" 
patient_list = ['H_0','H_1','H_2','H_3','H_4','V_100','V_101','V_102','V_103'] 
patient_list = ['U_57'] # 0 11 1 64 20 40 57 58
 
patient_list = ["U_5","U_7","U_8","U_9"] # "U_4", "U_18",
patient_list = ["H_0","H_1","H_2","H_3","H_4"]
patient_list = ["C_1"]
 

steps = ['fod','track','sift','whole']   
steps = ['reg_5TT','dipy']  
steps = ["preproc","5TT","reg_5TT","rois"]  # ,"reg_prob"]  
steps = ["reg_prob","rois"]  
steps = ["preproc","5TT","reg_5TT","dipy"]     
 
# steps = ["preproc"]
# steps = ["rois"] 
n_jobs = -1

wm_mask = "yes" ; reg_step = "none" ; n_streams = 50000 
algos = ["iFOD2",       # default, prob with trilin interp of FOD in SH basis
         "SD_STREAM",   # det with SD of trilin interp of FOD
         "Tensor_Det",  # det with dmri: trilin interp of tensor eigenvector
         "Tensor_Prob"] # prob with dmri: bootstrap + trilin interp of tensor eigenvector
refresh = False       
density_map_files = []
density_path = "/auto/home/users/d/r/drimez/AFQ_data/Natbrainlab/"
with os.scandir(density_path) as my_it:
    for an_entry in my_it:
        density_map_files.append(density_path + an_entry.name)

tracking_algo = "dipy_prob" ; model="_sparse" ; filtering = False ; postrack=False ; reg = False  ; resp=True

def compressed_pickle(title, data):
   with bz2.BZ2File(title + '.pbz2', 'w') as f: 
       cPickle.dump(data, f)
       f.close()
       
def decompress_pickle(filename):
   data = bz2.BZ2File(filename+".pbz2", 'rb')
   data = cPickle.load(data)
   return data
 
def recursor(path,patient,iterator):  
    # if iterator is os.scandir("dir"), entry will be the list of contents in "dir" directory 
    for entry in iterator:  
        new_path = os.path.join(path,entry.name) 
        if entry.is_file(): 
            if entry.name.split('.')[-1] == "tck":
                # it's a tck file so let's convert it  
                # new_trk_path = os.path.join(path,".".join(entry.name.split('.')[:-1])+".trk")
                # tck2trk = mrt.MRTrix2TrackVis()
                # tck2trk.inputs.in_file = new_path
                # tck2trk.inputs.out_filename = new_trk_path
                # tck2trk.inputs.image_file = f_path + "subjects/" + patient + "/tracking/preproc/" + patient + "_dmri_preproc.nii.gz"
                # tck2trk.run()   # matrix_file to apply fsl affine
                pass
        # it's not a file so we create a new folder and iterate over entry's contents   
        else:    
            with os.scandir(new_path) as new_it:
                recursor(new_path,new_it) 
                new_it.close()
         
def backup(filename,patient):   
    new_path = f_path + "subjects/" + patient + "/tracking/"
    if not os.path.isdir(new_path + "backup/"):
        os.mkdir(new_path + "backup/")
         
    new_path = f_path + "subjects/" + patient + "/tracking/"
    name, ext = os.path.splitext(filename)
    name = name.split("tracking/")[-1]
    exists = True ; count = -1
    while exists:
        count += 1
        exists = os.path.exists(filename + "_" + str(count) + ".tck")
    os.system('mv ' + new_path + name + ext + " " + new_path  + "backup/" + name + "_" + str(count) + ext)   
       

def bootstrap(data_to_boot,fun_to_boot):      
         
    data = np.array([data_to_boot]).flatten()
         
    data = np.nan_to_num(data) 
    if np.all(data==0):
        return 0
     
    def boot_iter(data, fun_to_boot, size_): 
        sample = np.random.choice(data, size=size_, replace=True) 
        return np.nan_to_num(fun_to_boot(sample))
        
    size_ = len(data)
    
    result_array = Parallel(n_jobs=-1,pre_dispatch='n_jobs',require="sharedmem")( #,require="sharedmem"
                                 delayed(boot_iter)(data, fun_to_boot, size_) 
                                 for _ in range(min(len(data),5000)*10) )
      
    biaised_result = np.nan_to_num(fun_to_boot(data))
    boot_1 = np.nan_to_num(np.nanmean(result_array))
    
    gc.collect()
    
    return 0.368*biaised_result + 0.632*boot_1
             
def generate_rois(f_path=f_path,p=None,wm_path=None,reg=False,mov=None,prefix="",replace=None): # replace must be in patient's space

    if not reg in ("True",True):
        
        log_prefix = "Tracking"  
        # affine registration with flirt 
        path = f_path + "subjects/" + p + "/tracking/preproc%s/"%prefix + p
        
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Affine registration launched for white matter parcellation\n")
        f = open(f_path + "subjects/" + p + "/tracking/logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Affine registration launched for white matter parcellation \n")
        f.close()
        
        if replace is None:  
            # cerebellum volbrain parcellation by default /
            rep_folder = f_path + "VolBrain/Results_native/native_%s_T1"%p 
            replace = f_path + "wm_masks/C_1/native_lab_ln_crop_mmni_mfjob377386.nii.gz" 
            replace_mask = f_path + "wm_masks/C_1/native_tissue_ln_crop_mmni_mfjob377386.nii.gz"
            """
            files = os.listdir(rep_folder)
       
            replace = np.array(files)[np.array([True if ("native" in file_.split('_')) and ("lab" in file_.split('_')) else False for file_ in files])]
            replace = rep_folder + replace[0]
            replace_mask = np.array(files)[np.array([True if ("native" in file_.split('_')) and ("tissue" in file_.split('_')) else False for file_ in files])]
            replace_mask = rep_folder + replace_mask[0]
            """
            
        original_labels, org_lab_affine = load_nifti(wm_path)
        """
        mrcmd = "mrregister -type rigid " + replace + " " + wm_path + " -mask1 " + replace_mask + " -transformed " + path + "_reg_cb.nii.gz -force"
        """
          
        # new_labels = np.concatenate(( np.zeros((len(new_labels),(len(original_labels[0])-len(new_labels[0]))//2 +4,(len(original_labels[0,0]) ))), new_labels , np.zeros((len(new_labels),(len(original_labels[0])-len(new_labels[0]))//2 -4,(len(original_labels[0,0]) )))), axis=1)
        
        new_labels, affine = load_nifti(replace)  
        new_labels = np.array([np.rot90(_) for _ in new_labels])
        """
        new_labels2 = copy.copy(new_labels)
        new_labels2[new_labels2!=0] = 1
        save_nifti(path+"_reg_cbmask.nii.gz",new_labels2,affine) 
        """
        to_add = np.zeros_like(original_labels)
        to_add = np.zeros_like(new_labels).astype(np.int64)
        vol2free_df = pd.read_csv( "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/volbrain_trad.txt", 
                                  sep=",", header=None, index_col=False).values 
        vol2free = {str(old_lab): new_lab for old_lab, new_lab in zip(vol2free_df[:,1],vol2free_df[:,0])} 
        intersect = [8, 47]
        original_labels[original_labels==8] = 7
        original_labels[original_labels==47] = 46
        original_labels[original_labels==9] = 12
        original_labels[original_labels==10] = 12
        original_labels[original_labels==48] = 51
        original_labels[original_labels==49] = 51
         
        cb = np.zeros_like(original_labels)
        cb[original_labels==7] = 1
        cb[original_labels==46] = 1
 
        save_nifti(path+"_reg_wmparc_formask.nii.gz", original_labels, org_lab_affine)
        original_labels, org_lab_affine = load_nifti(wm_path)
                     
        original_labels[original_labels==8] = 0
        original_labels[original_labels==47] = 0
                    
        counter = 0 ; new_lab = np.unique(new_labels)
        for org_label, trad in vol2free.items():   
            to_add[new_labels.astype(float)==float(trad)] = float(org_label)  
         
        def reg_func(moving, static, moving_affine, static_affine, static_mask=None, moving_mask=None):
            """Convenience function for registration using a pipeline.
               Uses variables in global scope, except for static_mask and moving_mask.
            """
            from dipy.align import affine_registration
             
            # pipeline = [translation, rigid] 
            affreg = AffineRegistration(level_iters=[200,50,20])

            static_masked, moving_masked = static, moving
            if static_mask is not None:
                static_masked = static*static_mask
            if moving_mask is not None:
                moving_masked = moving*moving_mask
            """
            transform = transform_centers_of_mass(static_masked, static_affine,
                                                  moving_masked, moving_affine)
            starting_affine = transform.affine 
            """
            xform, xopt, fopt = affreg.optimize(static_masked, moving_masked, 
                                                RigidTransform3D(), None,
                                                static_affine, moving_affine,
                                                starting_affine=np.array( ((1,0,0,0), (0,0,-1,0), (0,1,0,0), (0,0,0,1)) ),
                                                ret_metric=True )
            starting_affine = xform.affine
            """ 
            xformed_img, reg_affine = affine_registration(moving, static,
                                                          moving_affine=moving_affine,
                                                          static_affine=static_affine,
                                                          nbins=32, metric='MI',
                                                          pipeline=pipeline,
                                                          level_iters=level_iters,
                                                          sigmas=sigmas,
                                                          factors=factors,
                                                          static_mask=static_mask,
                                                          moving_mask=moving_mask) 
            """
            affine_map = AffineMap(starting_affine,
                                   static.shape, static_affine,
                                   moving.shape, moving_affine)

            resampled = affine_map.transform(moving,interpolation="nearest")
            print(resampled.shape,np.array( ((1,0,0,0), (0,0,-1,0), (0,1,0,0), (0,0,0,1)) ))

            return resampled, static_affine # xformed_img, reg_affine

        xformed_img, reg_affine = reg_func(to_add.astype(type(original_labels[0,0,0])), cb, affine, org_lab_affine) #, static_mask=cb)
        original_labels[np.logical_and(xformed_img!=0,np.logical_or(original_labels==7,original_labels==46))] = 0
        xformed_img[np.logical_and(xformed_img!=0,original_labels!=0)] = 0
        save_nifti(path+"_wmparc.nii.gz", (original_labels+xformed_img).astype(type(original_labels[0,0,0])), org_lab_affine)
        save_nifti(path+"_reg_cb.nii.gz", xformed_img.astype(type(original_labels[0,0,0])), org_lab_affine)


        # save_nifti(path+"_wmparc.nii.gz", original_labels, org_lab_affine)

        registered_t1_path = path + "_afreg_t1.nii.gz"
        # registered_wm_path = path + "_afreg_wm_parc.nii.gz"
        moving = f_path + "wm_masks/" + p + "/T1.nii.gz"  
        static = f_path + "subjects/" + p + "/T1/" + p + "_T1_corr_projected.nii.gz"
        # static = f_path + "subjects/" + p + "/dMRI/preproc/" + p + "_dmri_preproc.nii.gz" 
        # static = f_path + "subjects/%s/tracking/preproc/%s_dmri_upsampled.nii.gz"%(p,p)
          
        if (reg in ("affine")) or True:
            flirt_t1      = "flirt -in {newvol} -ref {refvol} -out {outvol} -omat {invol2refvol}.mat -dof 6 -v".format(newvol=moving, 
                             refvol=static, outvol=registered_t1_path, invol2refvol=path + "_afreg")
            """
            flirt_t1     += "flirt -in {newvol} -ref {refvol} -out {outvol} -init {invol2refvol}.mat -omat {invol2refvol}.mat -dof 6 ; ".format(newvol=moving, 
                             refvol=static, outvol=registered_t1_path, invol2refvol=path + "_afreg")
            flirt_t1     += "flirt -in {newvol} -ref {refvol} -out {outvol} -init {invol2refvol}.mat -omat {invol2refvol}.mat -dof 9".format(newvol=moving, 
                             refvol=static, outvol=registered_t1_path, invol2refvol=path + "_afreg")

            flirt_t1_aff  = "flirt -in {newvol} -ref {refvol} -out {outvol} -init {invol2refvol}.mat -omat {invol2refvol}.mat -dof 12".format(newvol=moving, 
                             refvol=static, outvol=registered_t1_path, invol2refvol=path + "_afreg")
            """
            flirt_wm_aff  = "flirt -in {newvol} -ref {refvol} -out {outvol} -init {invol2refvol}.mat -dof 6 -interp nearestneighbour -applyxfm -v".format(newvol=wm_path, refvol=static, outvol=path + "_fsl_affreg_wmparc" , invol2refvol=path + "_afreg")
            flirt_convert = "c3d_affine_tool -ref {refvol} -src {newvol} {invol2refvol}.mat -fsl2ras -oitk {invol2refvol}.tfm".format(newvol=moving, refvol=static, outvol=registered_t1_path, invol2refvol=path + "_afreg")
            ants_wm = "antsApplyTransforms --reference-image %s --input %s --output %s.nii.gz"%(static,path+"_new_wmparc.nii.gz",path + "_reg_wmparc_formask") + " --transform %s.tfm"%(path + "_afreg") + " --interpolation GenericLabel ; "
            # ants_wm = "antsApplyTransforms --reference-image %s --input %s --output %s.nii.gz -v"%(static,path+"_reg_cbmask.nii.gz",path + "_reg_cbmask") + " --transform %s.tfm"%(path + "_afreg") + " --interpolation GenericLabel ; "
            ants_wm = "antsApplyTransforms --reference-image %s --input %s --output %s.nii.gz -v"%(static,path+"_wmparc.nii.gz",path + "_reg_wmparc") + " --transform %s.tfm"%(path + "_afreg") + " --interpolation GenericLabel ; "
            flirt_command = flirt_t1 + " ; " + flirt_wm_aff + " ; " +  flirt_convert + " ; " + ants_wm # registration then conversion to ANTs
            print(flirt_command)              
            bashcmd = flirt_command.split()  
            process = subprocess.Popen(flirt_command, universal_newlines=True, shell=True, stdout=sys.stdout,
                                       stderr=subprocess.STDOUT)
            # wait until finish
            out, error = process.communicate()
        """
        # syn registration with ants 
        ants_cmd = "antsRegistration --float 1 --dimensionality 3 --use-histogram-matching 1 --transform SyN[0.2,3,0] --convergence [30x99x15x15,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 4x3x2x1mm"
          
        if reg in ("affine","defo"): 
            moving  = path + "_afreg_t1.nii.gz"
            output  = path + "_reg_t1" 
            ants_t1 = ants_cmd + " --metric CC[%s,%s]"%(static,moving) + " --initial-moving-transform %s.tfm"%(path + "_afreg") + " --output [%s,%s.nii.gz]"%(output,output) + " --write-composite-transform 1"
            moving  = wm_path
            output  = path + "_reg_wmparc" 
        if reg in ("apply"):
            moving  = wm_path
            output  = f_path + "subjects/" + p + "/tracking/preproc%s/"%prefix + wm_path.split(".")[0].split("/")[-1]  + "_reg"
        ants_wm = "antsApplyTransforms --reference-image %s --input %s --output %s.nii.gz"%(static,moving,output) + " --transform %s"%(path + "_reg_t1Composite.h5") + " --interpolation GenericLabel "
        
        ants_cmd = ants_t1 + " ; " if not reg == "apply" else ""
        ants_cmd += ants_wm
                                     
        bashcmd = ants_cmd.split()
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Deformable registration launched for white matter parcellation\n")
        f = open(f_path + "subjects/" + p + "/tracking/logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Deformable registration launched for white matter parcellation\n" + ants_cmd)
        f.close()
    
        process = subprocess.Popen(ants_cmd, universal_newlines=True, shell=True, stdout=sys.stdout,
                                   stderr=subprocess.STDOUT)
        # wait until finish
        out, error = process.communicate()
        """
"""   
def reg_csf(p,reg=False):

    wm_path = None

    if not reg:
        log_prefix = "Tracking"  
        # affine registration with flirt 
        path = f_path + "subjects/" + p + "/tracking/preproc/" + p
        registered_t1_path = path + "_afreg_t1.nii.gz"
        # registered_wm_path = path + "_afreg_wm_parc.nii.gz"
        moving = f_path + "subjects/" + p + "/T1_vertige/" + p + "_T1_brain.nii.gz"
        static = f_path + "subjects/" + p + "/T1_vertige/" + p + "_T1_corr_projected.nii.gz" 
         
        flirt_t1      = "flirt -in {newvol} -ref {refvol} -out {outvol} -omat {invol2refvol}.mat -dof 9".format(newvol=moving, 
                         refvol=static, outvol=registered_t1_path, invol2refvol=path + "_afreg")
        flirt_t1_aff  = "flirt -in {newvol} -ref {refvol} -out {outvol} -init {invol2refvol}.mat -omat {invol2refvol}.mat -dof 12".format(newvol=moving, 
                         refvol=static, outvol=registered_t1_path, invol2refvol=path + "_afreg")
        # flirt_convert = "c3d_affine_tool -ref {refvol} -src {newvol} {invol2refvol}.mat -fsl2ras -o {invol2refvol}.h5".format(newvol=moving, refvol=static, outvol=registered_t1_path, invol2refvol=path + "_afreg")
        flirt_command = flirt_t1 + " ; " +  flirt_t1_aff # + " ; " +  flirt_convert  registration then conversion to ANTs
                       
        bashcmd = flirt_command.split() 
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Affine registration launched for white matter parcellation\n")
        f = open(f_path + "subjects/" + p + "/tracking/logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Affine registration launched for white matter parcellation\n")
        f.close()
    
        process = subprocess.Popen(flirt_command, universal_newlines=True, shell=True, stdout=sys.stdout,
                                   stderr=subprocess.STDOUT)
        # wait until finish
        out, error = process.communicate()
        
        # syn registration with ants 
        ants_cmd = "antsRegistration --float 1 --dimensionality 3 --use-histogram-matching 1 --transform SyN[0.2,3,0] --convergence [30x99x11,1e-6,10] --shrink-factors 8x4x2 --smoothing-sigmas 2x2x2vox"
                                 
        moving  = path + "_afreg_t1.nii.gz"
        output  = path + "_reg_t1" 
        ants_t1 = ants_cmd + " --metric CC[%s,%s]"%(static,moving) + " --initial-moving-transform [%s,%s,1]"%(static,moving) + " --output [%s,%s.nii.gz]"%(output,output) + " --write-composite-transform 1"
            
        moving  = wm_path
        output  = path + "_reg_wm_parc" 
        ants_wm = ants_cmd + " --interpolation GenericLabel" + " --metric CC[%s,%s]"%(static,moving) + " --initial-moving-transform %s"%(path + "_reg_t1Composite.h5") + " --output [%s,%s.nii.gz]"%(output,output)
        
        ants_cmd = ants_t1 + " ; " + ants_wm
                                     
        bashcmd = ants_cmd.split()
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Deformable registration launched for white matter parcellation\n")
        f = open(f_path + "subjects/" + p + "/tracking/logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": Deformable registration launched for white matter parcellation\n")
        f.close()
    
        process = subprocess.Popen(ants_cmd, universal_newlines=True, shell=True, stdout=sys.stdout,
                                   stderr=subprocess.STDOUT)
        # wait until finish
        out, error = process.communicate()
"""  
def _resample_tg(tg, n_points):
    # reformat for dipy's set_number_of_points
    if isinstance(tg, np.ndarray):
        if len(tg.shape) > 2:
            streamlines = tg.tolist()
            streamlines = [np.asarray(item) for item in streamlines]
    elif isinstance(tg, list):
        streamlines = [np.asarray(item) for item in tg]
    else:
        streamlines = tg.streamlines

    return dps.set_number_of_points(streamlines, n_points)
    
def dipy_tracto(p,n_iterations=12,model=model,f_path=f_path,density_map_files=density_map_files,fod=True,single_bdl=None,filtering=filtering,postrack=postrack,resp=None):

    if isinstance(fod,str):
        fod = fod=="True"
        
    if isinstance(resp,str):
        resp = resp=="True"
        
    if isinstance(n_iterations,str):
        n_iterations = int(n_iterations)
        
    if model=="_prob" and single_bdl is None:
        n_iterations = 25
    
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
    cbmask = f_path + "subjects/%s/tracking/preproc/%s_reg_cbmask.nii.gz"%(p,p) 
    seg_path = f_path + "subjects/%s/masks_vertige/%s_segmentation.nii.gz"%(p,p)
    dmri_root = f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc"%(p,p) 
    dire = f_path + "subjects/%s/tracking/preproc/"%p
    ref_t1 = f_path + "subjects/" + p + "/T1/" + p + "_T1_corr_projected.nii.gz" 
    bvals, bvecs = read_bvals_bvecs(dmri_root+".bval", dmri_root+".bvec")
    gtab = gradient_table(bvals, bvecs)
    dmri_preproc = f_path + "subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz"%(p,p)
      
    data, ref_affine, ref_image = load_nifti(dmri_preproc, return_img=True)
     
    registered_data_wm, _ = load_nifti(dire+"/"+p+"_wm_dil.nii.gz") 
         
    # conv_bckgd = "mrconvert " + bck_path + " " + bck_path.split(".")[0] + ".mif.gz -force"
    # conv_csf += "mrconvert " + csf_path_2.split(".")[0] + ".mif.gz " + csf_path_2 + " -force ; " 
    # conv_csf += " ; maskfilter " + csf_path_2.split(".")[0] + "_eroded.nii.gz erode -npass 1 " + csf_path_2.split(".")[0] + "_eroded.nii.gz -force ; "
    # conv = conv_bckgd  + " ; " + conv_csf
    # bashcmd = conv.split()  
    # process = subprocess.Popen(conv, universal_newlines=True, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
    # wait until finish
    # out, error = process.communicate()

    ####### generate streamlines
    # constrained spherical deconvolution
    # CSA odf + stopping criterion
    # gfa = csa_model.fit(data, mask=white_matter).gfa
    # stopping_criterion = ThresholdStoppingCriterion(gfa, .25)
    
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
        sys.setrecursionlimit(10000)
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
                compressed_pickle(f_path+"subjects/%s/tracking/FOD/%s_csd_model%s"%(p,p,model),csd_model)
            else: 
                csd_model = decompress_pickle(f_path+"subjects/%s/tracking/FOD/%s_csd_model"%(p,p))  
             
            if not os.path.exists(f_path+"subjects/%s/tracking/FOD/%s_fod%s.pbz2"%(p,p,model)) and model=="_prob":
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
        """
        if not single_bdl is None:
            density_map_files = [density_map_files]
        else:
            density_path = "/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/"%p
            with os.scandir(density_path) as my_it:
                for an_entry in my_it:
                    density_map_files.append(density_path + an_entry.name)
        """     
        def single_iter(n_iter,single_bdl=None,pmf=pmf,registered_data_wm=registered_data_wm,stopping_criterion=stopping_criterion, csd_model=csd_model,  
                        ref_image=ref_image,output_name=output_name,model=model,f_path=f_path,density_map_files=density_map_files,filtering=filtering): 
            
            peak_model = None
            if model == "_prob": 
                rel_th = np.random.rand(1)*0.1 + 0.8 # between 0.8 and 0.9
                max_angle = np.random.rand(1)*5 + 20
                peak_model = ProbabilisticDirectionGetter.from_pmf( pmf, max_angle=max_angle, sphere=default_sphere, pmf_threshold=0.2,
                                                                    relative_peak_threshold = rel_th, min_separation_angle=5 )
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
                rel_th = np.random.rand(1)*0.15 + 0.75 # between 0.7 and 0.9
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
                seeding_mask[density_map_img<=0.01] = 0
                seeds_count = 4 # 5*np.sum(seeding_mask>0.01)
                
            seeds = utils.random_seeds_from_mask( seeding_mask, affine, seed_count_per_voxel=True,  
                                                  seeds_count = int(seeds_count) )#  
                                                                    
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
             
            sft = StatefulTractogram(streamlines, ref_image, Space.VOX)
            is_saved = save_tck(sft, output_name + "_%s.tck"%n_iter)
            
            # crop to non-dilated wm after filtering tracks that pass through bcgd and not through csf 
            crop = "tckresample " + output_name + "_%s.tck "%n_iter + output_name + "_%s.tck -step_size 0.5 -force ; "%n_iter if False and model=="_prob" else ""
            crop = "tckedit " + output_name + "_%s.tck "%n_iter + output_name + "_%s.tck -include "%n_iter + gm_path_2 + " -exclude " + csf_path_2   # " -include " + cbmask +  
            
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
            for n_iter in range(n_iterations):
            
                # done = Parallel(n_jobs=min(5,cpu_count),verbose=60,pre_dispatch="n_jobs")(delayed(single_iter)(n_iter=subiter,single_bdl=None) for subiter in np.arange(n_iter*cpu_count,(n_iter+1)*5))
                #for subiter in np.arange(n_iter*cpu_count,(n_iter+1)*cpu_count):
                #    print(" ======  Tracking: %s / %s   ======= "%(subiter,n_iterations))
                done = single_iter(n_iter=n_iter,single_bdl=single_bdl)  
                
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
                            elif os.path.getsize(output_name + ".trk")>=900000000 and model=="_prob":
                                print("File size exceeded") 
                                return True
                            elif os.path.getsize(output_name + ".trk")>=450000000 and not model=="_prob":
                                print("File size exceeded") 
                                return True
                        return False
                     
                    saved = "" ; check_ = False
                    with os.scandir(f_path + "subjects/%s/tracking/"%p) as it:
                        for entry in it: 
                            if len(entry.name.split('.'))>1:
                                if  entry.name.split('.')[-1]=="tck" \
                                    and ("prob" in entry.name.split('_') 
                                          or model == ""): 
                                           
                                    check_ = check_size()
                                    if not check_: 
                                        t1 = load_tck(entry.path , dmri_preproc)  
                                        trk = None  
                                        if  os.path.exists(output_name + ".trk"):
                                            t2 = load_trk(output_name + ".trk", dmri_preproc)  
                                            trk = StatefulTractogram(  list(t1.streamlines) + list(t2.streamlines),
                                                                       dmri_preproc,
                                                                       Space.RASMM,
                                                                       data_per_streamline={k: (list(t1.data_per_streamline[k]) + list(t2.data_per_streamline[k]))
                                                                                                for k in t2.data_per_streamline.keys()  }) 
                                        else:
                                            trk = t1   
                                        save_trk(trk,output_name + ".trk")  
                                    os.system("rm "+ entry.path)
                  
                    if check_:
                        break                  
                    # ensures no correspondances with classical streamlines when highlighting thalamic projections
                    """
                    if n_iter%2==0 or single_bdl:
                        
                        idx_above_prob = np.arange(len(t1.streamlines))[np.all(probs < 0.9, axis=-1)]
                        t1 = StatefulTractogram(  list(t1.streamlines[idx_above_prob]) ,
                                                       f_path + "subjects/%s/dMRI/preproc/"%p + p + "_dmri_preproc.nii.gz",
                                                       Space.RASMM    ) 
                        
                        pass
                    # select streamlines with probability maps
                    else:
                        fgarray = np.array(_resample_tg(t1, 100))
                        probs = np.zeros((len(t1.streamlines), len(density_map_files)) )
                        for i_dens_map, density_map in enumerate(density_map_files):
                        
                            if i_dens_map%5==0:
                                 print("[ prob ] " + datetime.datetime.now().strftime(
                                 "%d.%b %Y %H:%M:%S") + ": starting map %s, %s / %s\n"%(density_map.split('/')[-1].split('.')[0], i_dens_map, len(density_map_files)) )
                            density_map_img, _ = load_nifti(density_map)
                            
                            fiber_probabilities = dts.values_from_volume(density_map_img, fgarray, np.eye(4))
                            probs[:,i_dens_map] = np.mean(fiber_probabilities,axis=-1)
                        
                        prob_th = 0.005 if model=="_prob" else 0.01
                        idx_above_prob = np.arange(len(t1.streamlines))[np.any(probs >= prob_th, axis=-1).flatten()]
                        t1 = StatefulTractogram(  list(t1.streamlines[idx_above_prob]) ,
                                                       f_path + "subjects/%s/dMRI/preproc/"%p + p + "_dmri_preproc.nii.gz",
                                                       Space.RASMM    )     
                    
                    query(model,"_"+str(n_iter))
                    if os.path.exists(f_path + "subjects/" + patient + "/tracking/" + patient + "_dipy" + mod + "_cleaned%s.trk"%("_"+str(n_iter))):
                        os.system("rm "+ output_name + ".trk")
                    """
                """
                # merges results       
                with os.scandir(f_path + "subjects/%s/tracking/"%p) as it:
                    for entry in it:
                        if os.path.isfile(f_path + "subjects/%s/tracking/"%p + entry.name) \
                            and entry.name.split('.')[-1]=="trk" \
                            and ("prob" in entry.name.split('_') or model == ""):# \
                            #and "cleaned" in entry.name.split('_'):
                            
                            trk = None ; t1 = load_trk(f_path + "subjects/%s/tracking/"%p + entry.name, dmri_preproc)
                            if  os.path.exists(output_name + ".trk"):
                                t2 = load_trk(output_name + ".trk", dmri_preproc)
                                trk = StatefulTractogram(  list(t1.streamlines) + list(t2.streamlines),
                                                           dmri_preproc,
                                                           Space.RASMM,
                                                           data_per_streamline={k: (list(t1.data_per_streamline[k]) + list(t2.data_per_streamline[k]))
                                                                                    for k in t2.data_per_streamline.keys()  }) 
                            else:
                                trk = t1
                                
                            save_trk(trk,output_name + ".trk") 
                """
        if not done:
            print("Hello")
            if not single_bdl is None:
                if not os.path.isdir(f_path + "subjects/%s/tracking/Solo/"%p):
                    os.mkdir(f_path + "subjects/%s/tracking/Solo/"%p)
                output_name =  f_path + "subjects/%s/tracking/Solo/"%p + p + "_dipy" + model + "_" + single_bdl
                
            for n_iter in range(n_iterations):  
            
                print(" ======  Tracking: %s / %s   ======= "%(n_iter,n_iterations))
                single_iter(n_iter=n_iter,single_bdl=single_bdl)  
                    
                if  os.path.exists(output_name + ".trk"):
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
                        elif os.path.getsize(output_name + ".trk")>=900000000 and model=="_prob":
                            print("File size exceeded")
                            os.system("rm " + "/".join(output_name.split("/")[:-1]) + "*.tck")
                            return True
                        elif os.path.getsize(output_name + ".trk")>=450000000 and not model=="_prob":
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
                                                               Space.RASMM,
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
                                                       Space.RASMM,
                                                       data_per_streamline={k: (list(t1.data_per_streamline[k]) + list(t2.data_per_streamline[k]))
                                                                                for k in t2.data_per_streamline.keys()  }) 
                        else:
                            trk = t1
                            
                        save_trk(trk,output_name + ".trk") 
    return True
                    
def tracto(f_path=f_path,patient_list=patient_list,wm_mask=wm_mask,steps=steps,n_streams=n_streams,n_jobs=n_jobs,notlist=False,reg=reg,density_map_files=density_map_files,parrallel=False):     

    if "FOD" in steps:
        response_list = ""
        patient_list = json.load(open("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/subj_list.json.json","r"))
        for p in patient_list:
            data = "data_1" if p[0]=="H" else "data_2"
            resp = "dwi2response  tournier /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/tracking/preproc/{p}_dmri_upsampled.nii.gz /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/tracking/FOD/{p}_mrtrix_response.txt -mask /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/tracking/preproc/{p}_wm_new.nii.gz -fslgrad /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/{data}/{p}.bvec /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/{data}/{p}.bval -lmax 12 -peak_ratio 0.01 -force".format(p=p,data=data)
            
            cmd = rep 
            process = subprocess.Popen(step_6, universal_newlines=True, shell=True, stdout=sys.stdout,
                                                   stderr=subprocess.STDOUT)
            # wait until finish
            out, error = process.communicate()
            response_list += "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/tracking/FOD/{p}_mrtrix_response.txt "
            
        os.system("responsemean " + + "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/group_average_response.txt")
    """
    for p in patient_list:
        fod = "dwi2fod csd /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/tracking/preproc/{p}_dmri_upsampled.nii.gz /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/tracking/FOD/{p}_mrtrix_response.txt /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/tracking/FOD/{p}_mrtrix_fod.nii.gz -lmax 12 -mask /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/{p}/tracking/preproc/{p}_wm_new.nii.gz -fslgrad /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/{data}/{p}.bvec /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/{data}/{p}.bval -directions /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/H_0/tracking/FOD/sphere.txt -force".format(p=p,data=data)
        cmd = rep  fod
        process = subprocess.Popen(step_6, universal_newlines=True, shell=True, stdout=sys.stdout,
                                               stderr=subprocess.STDOUT)
        # wait until finish
        out, error = process.communicate()
    
    """
    # notlist = True
    f_path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/"
    f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset_new/PROJECT/"  
    f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/elikopy_subset_new2/PROJECT/"  
    patient_list = patient_list if isinstance(patient_list,list) else [patient_list]  
    """
    patient_list = ["U_"+str(_) for _ in range(78) if os.path.exists(f_path+"subjects/U_"+str(_)+"/dMRI/preproc/U_"+str(_)+"_dmri_preproc.nii.gz") \
                    and not os.path.exists("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset/PROJECT/subjects/U_%s/tracking/preproc/U_%s_reg_wmparc.nii.gz"%(_,_))]   
    """
    if steps==["rois"] and not parrallel and False:
        Parallel(n_jobs=n_jobs,verbose=500,pre_dispatch='2*n_jobs')( delayed(tracto)( f_path,[patient],wm_mask,["rois"],n_streams,n_jobs,
                                                                                      notlist,reg,density_map_files,parrallel=True ) 
                                                                     for patient in patient_list )
    else:
        for patient in patient_list: 
            path = f_path + "subjects/" + patient + "/"
            new_path = path + "tracking/" 
            if not os.path.isdir(new_path):
                os.mkdir(new_path) 
            elif os.path.isdir(new_path+"preproc") and "preproc" in steps and not refresh: 
                if os.path.exists("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset/PROJECT/subjects/%s/tracking/preproc/%s_reg_wmparc.nii.gz"%(patient,patient)):
                    if not os.path.exists(new_path + "ROIs/metrics.csv"):
                        steps = ["rois"] 
                    else:
                        print(patient+" already done",file=open("/auto/home/users/d/r/drimez/PREPROC.txt","a"))
                        break
                elif len(os.listdir(new_path+"preproc"))>15 and not refresh:
                    print(patient+" already done",file=open("/auto/home/users/d/r/drimez/PREPROC.txt","a"))
                    break
            
            log_prefix = "Tracking"
            print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Tracking launched for patient " + patient + "\n")
            f = open(new_path + "/logs.txt", "a+")
            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                "%d.%b %Y %H:%M:%S") + ": Tracking launched for patient " + patient + "\n")
            f.close()
            
            if not os.path.isdir(new_path+"preproc"):
                steps = ["preproc","5TT","reg_5TT"] + list([_ for _ in steps if not _ in ["preproc","5TT","reg_5TT","reg_prob"]]) #,"reg_prob"
            elif len(os.listdir(new_path+"preproc"))<15:
                steps = ["preproc","5TT","reg_5TT"] + list([_ for _ in steps if not _ in ["preproc","5TT","reg_5TT","reg_prob"]]) #,"reg_prob"
                
            if (not os.path.exists("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset/PROJECT/subjects/%s/tracking/preproc/%s_reg_wmparc.nii.gz"%(patient,patient))) or refresh:
                if (not (("dipy" in steps) or ("rois" in steps)) ):
                    steps = ["preproc","5TT","reg_5TT"] # ,"reg_prob"
                elif not patient in [_.split()[0] for _ in open("/auto/home/users/d/r/drimez/PREPROC2.txt","r").readlines()]:
                    steps = (["preproc","5TT","reg_5TT"]) + list([_ for _ in steps if not _ in ("preproc","5TT","reg_5TT")]) #,"reg_prob"
                else:
                    steps = list([_ for _ in steps if not _ in ("preproc","5TT","reg_5TT")])
            """   
            if steps[0]=="dipy":
                steps = ["preproc","5TT","reg_5TT","dipy"]
            """ 
            for starting_step in steps:  
                   
                preproc = path + "dMRI/preproc/" + patient + "_dmri_preproc.nii.gz"
                # prep_wm = path + "masks_vertige/" + patient + "_segmentation.nii.gz"
                wm_parc = f_path + "wm_masks/" + patient + "/wmparc.nii.gz"
                dmri    = new_path + "preproc/" + patient + "_dmri_preproc"
                wm      = new_path + "preproc/" + patient + "_wm"
                parc    = new_path + "preproc/" + patient + "_wmparc"
                     
                if starting_step in ("preproc"): 
                    if not os.path.isdir(new_path + "preproc/"):
                        os.mkdir(new_path + "preproc/") 
                    """
                    crop = "mrgrid " + preproc + " regrid -template " + f_path + "wm_masks/%s/wmparc.nii.gz  "%(patient) + f_path + "subjects/%s/tracking/preproc/%s_dmri_upsampled.nii.gz"%(patient,patient)
                    
                    bashcmd = crop.split()  
                    process = subprocess.Popen(crop, universal_newlines=True, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
                    # wait until finish
                    out, error = process.communicate() 
                    """
                    ######## Preproc copy 
                    copy =    "cp " + preproc + " " + dmri + ".nii.gz"
                    # copy_wm = "cp " + prep_wm + " " +  wm  + ".nii.gz"
                    cp_parc = "cp " + wm_parc + " " + parc + ".nii.gz" 
                    erase = "rm " + dmri + ".nii.gz"
                    
                    rois = "python tracto.py generate_rois " + f_path + " " + patient + " " + parc + ".nii.gz " + patient + " affine"
                    
                    reg_parc = new_path + "preproc/" + patient + "_reg_wmparc"
                    
                    
                    ######### Conversion
                    conv_1 = "mrconvert " + dmri + ".nii.gz " + dmri + ".mif.gz -force"
                    conv_2 = "mrconvert " +  wm  + ".nii.gz " +  wm  + ".mif.gz -force"
                    conv_3 = "mrconvert " + parc + ".nii.gz " +  parc + ".mif.gz -force"
                    step_1 = copy + " ; " + erase + " ; " + cp_parc + " ; " + rois # + " ; " + copy_wm 
                    
                    bashcmd = step_1.split()  
                    process = subprocess.Popen(step_1, universal_newlines=True, shell=True, stdout=sys.stdout,
                                               stderr=subprocess.STDOUT)
                    # wait until finish
                    out, error = process.communicate()
                    """
                    seg, seg_affine = load_nifti(wm+".nii.gz")
                
                    seg[seg.astype('uint8')<70] = 0
                    seg[seg.astype('uint8')>180] = 0
                    seg[seg!=0] = 1 
                    save_nifti(wm+".nii.gz",seg,seg_affine) 
                    """
                    """
                    step_1 = conv_2 + " ; " + conv_3
             
                    bashcmd = step_1.split()  
                    process = subprocess.Popen(step_1, universal_newlines=True, shell=True, stdout=sys.stdout,
                                               stderr=subprocess.STDOUT)
                    # wait until finish
                    out, error = process.communicate()
                    """
                if starting_step in ("5TT"): 
                    orgwm    = f_path + "wm_masks/" + patient + "/wm.seg.nii.gz"
                    step_5TT = "mrconvert " + orgwm + " " + wm + "_org.mif.gz -force ; "
                    step_5TT = "" ; parc = new_path + "preproc/" + patient + "_reg_wmparc_formask"
                    
                    out5TT = new_path + "preproc/" + patient + "_5TT"
                    step_5TT += "5ttgen freesurfer -nocrop -force %s.nii.gz %s.nii.gz -lut /home/users/d/r/drimez/LUT.txt"%(parc,out5TT) # + " ; 5ttedit -wm %s.mif.gz %s.mif.gz %s_cor.mif.gz -force"%(wm,out5TT,out5TT)
                    
                    T1 = f_path + "subjects/" + patient + "T1/" + patient + "_T1_corr_projected.nii.gz" 
                    # step_5TT = step_5TT + " ; mrconvert " +  out5TT + ".mif.gz " +  out5TT + ".nii.gz -force" # + T1
                    # step_5TT = step_5TT + " ; mrconvert " +  out5TT + "_cor.mif.gz " +  out5TT + "_cor.nii.gz -force" # + T1
                    """
                    new_T1 = new_path + "preproc/%s_T1"%patient  
                    conv_6 = "mrconvert " +  T1  + ".nii.gz " +  new_T1  + ".mif.gz ; "
                    step_5TT = "5ttgen fsl -premasked -nocrop " + T1 + "mif.gz %s.mif.gz -force"%out5TT + " ; 5ttedit -wm %s.mif.gz %s.mif.gz %s_cor.mif.gz -force"%(wm,out5TT,out5TT)
                    
                    conv_5 = " ; mrconvert " +  out5TT + "_cor.mif.gz " +  out5TT + ".nii.gz -force"                    # to construct files for ACT tracking
                     
                    step_5TT = conv_6 + step_5TT + conv_5
                    """
                    
                    bashcmd = step_5TT.split()  
                    process = subprocess.Popen(step_5TT, universal_newlines=True, shell=True, stdout=sys.stdout,
                                               stderr=subprocess.STDOUT)
                    # wait until finish
                    out, error = process.communicate()
                     
                    # Nipype flow to convert 5TT file to PVE
                    path_5tt = out5TT + ".nii.gz" 
                    pve_csf_file = new_path + "preproc/" + patient + "_csf_pve.nii.gz"
                    pve_gm_file = new_path + "preproc/" + patient + "_gm_pve.nii.gz"
                    pve_wm_file = new_path + "preproc/" + patient + "_wm_pve.nii.gz"
                    conv_5TT2nii(path_5tt,pve_csf_file,pve_wm_file,pve_gm_file,T1+".nii.gz",patient) 
                    
                if starting_step in ("reg_5TT"): 
                    p = patient
                    T1_path = f_path + "wm_masks/%s/T1.nii.gz"%(p)
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
                    
                    """
                    seg, seg_affine = load_nifti(wm_path)  
                    seg[seg.astype('uint8')<90] = 0
                    seg[seg.astype('uint8')>115] = 0
                    seg[seg!=0] = 1 
                    save_nifti(dire+"/"+p+"_wm_seg_dil.nii.gz",seg,seg_affine) 
                    
                    seg, seg_affine = load_nifti(wm_path) 
                    seg[seg.astype('uint8')<100] = 0
                    seg[seg.astype('uint8')>115] = 0
                    seg[seg!=0] = 1 
                    save_nifti(dire+"/"+p+"_wm_seg_th.nii.gz",seg,seg_affine) 
                    """
                    
                    sub, _ = load_nifti(gm_pve_sub) 
                    registered_data_wm, seg_affine = load_nifti(wm_dil)
                    registered_data_wm[registered_data_wm<0.1] = 0  
                    registered_data_wm[registered_data_wm!=0] = 1
                    registered_data_wm = registered_data_wm + sub 
                    registered_data_wm[registered_data_wm!=0] = 1
                    save_nifti(wm_dil_th,registered_data_wm,seg_affine) 
                    
                    sub[sub<0.05] = 0
                    registered_data_wm, seg_affine = load_nifti(wm_pve)   
                    registered_data_wm = registered_data_wm + sub 
                    registered_data_wm[registered_data_wm!=0] = 1
                    save_nifti(dire+"/"+p+"_wm_allnew.nii.gz",registered_data_wm,seg_affine) 
                    ref_t1 = f_path + "subjects/%s/tracking/preproc/%s_dmri_upsampled.nii.gz"%(p,p)
                    ref_t1 = f_path + "subjects/" + patient + "/T1/" + patient + "_T1_corr_projected.nii.gz" 
                      
                    conv_csf =  "flirt -in {newvol} -ref {refvol} -out {outvol} -omat {invol2refvol}.mat -dof 6 -interp nearestneighbour ; ".format(newvol=T1_path, 
                                 refvol=ref_t1, outvol=dire+"/"+p+"_T1_reg.nii.gz", invol2refvol=dire+p+"_seg_afreg") 
                    """
                    conv_csf += "flirt -in {newvol} -ref {refvol} -out {outvol} -init {invol2refvol}.mat -dof 6 -applyxfm -interp nearestneighbour ; ".format(newvol=dire+"/"+p+"_wm_seg_dil.nii.gz", refvol=ref_t1+".nii.gz",outvol=dire+"/"+p+"_wm_seg_dil.nii.gz", invol2refvol=dire+p+"_seg_afreg") 
                    conv_csf += "flirt -in {newvol} -ref {refvol} -out {outvol} -init {invol2refvol}.mat -dof 6 -applyxfm -interp nearestneighbour ; ".format(newvol=wm_pve, refvol=ref_t1+".nii.gz",outvol=dire+"/"+p+"_wm_new.nii.gz", invol2refvol=dire+p+"_seg_afreg") 
                    conv_csf += "flirt -in {newvol} -ref {refvol} -out {outvol} -init {invol2refvol}.mat -dof 6 -applyxfm -interp nearestneighbour ; ".format(newvol=dire+"/"+p+"_wm_allnew.nii.gz", refvol=ref_t1+".nii.gz",outvol=dire+"/"+p+"_wm_allnew.nii.gz", invol2refvol=dire+p+"_seg_afreg") 
                    conv_csf += "flirt -in {newvol} -ref {refvol} -out {outvol} -init {invol2refvol}.mat -dof 6 -applyxfm -interp nearestneighbour ; ".format(newvol=wm_dil_th, refvol=ref_t1+".nii.gz",outvol=dire+"/"+p+"_wm_dil.nii.gz", invol2refvol=dire+p+"_seg_afreg") 
                    conv_csf += "flirt -in {newvol} -ref {refvol} -out {outvol} -init {invol2refvol}.mat -dof 6 -applyxfm -interp nearestneighbour ; ".format(newvol=dire+"/"+p+"_wm_seg_th.nii.gz", refvol=ref_t1+".nii.gz",outvol=dire+"/"+p+"_wm_seg_th.nii.gz", invol2refvol=dire+p+"_seg_afreg") 
                    """
                    conv_csf += "c3d_affine_tool -ref {refvol} -src {newvol} {invol2refvol}.mat -fsl2ras -oitk {invol2refvol}.tfm ; ".format(newvol=T1_path, refvol=dire+"/"+p+"_T1_reg.nii.gz",  invol2refvol=dire+p+"_seg_afreg")
                    # conv_csf += "antsApplyTransforms --reference-image %s --input %s --output %s.nii.gz"%(dire+"/"+p+"_T1_reg.nii.gz",dire+"/"+p+"_wm_seg_dil.nii.gz",dire+"/"+p+"_wm_seg_dil") + " --transform %s.tfm"%(dire+p+"_seg_afreg") + " --interpolation GenericLabel -v ; "
                    conv_csf += "antsApplyTransforms --reference-image %s --input %s --output %s.nii.gz"%(dire+"/"+p+"_T1_reg.nii.gz",wm_pve,dire+"/"+p+"_wm_new") + " --transform %s.tfm"%(dire+p+"_seg_afreg") + " --interpolation GenericLabel -v ; " 
                    conv_csf += "antsApplyTransforms --reference-image %s --input %s --output %s.nii.gz"%(dire+"/"+p+"_T1_reg.nii.gz",dire+"/"+p+"_wm_allnew.nii.gz",dire+"/"+p+"_wm_allnew") + " --transform %s.tfm"%(dire+p+"_seg_afreg") + " --interpolation GenericLabel -v ; "   
                    conv_csf += "antsApplyTransforms --reference-image %s --input %s --output %s.nii.gz"%(dire+"/"+p+"_T1_reg.nii.gz",wm_dil_th,dire+"/"+p+"_wm_dil") + " --transform %s.tfm"%(dire+p+"_seg_afreg") + " --interpolation GenericLabel -v ; "     
                    # conv_csf += "antsApplyTransforms --reference-image %s --input %s --output %s.nii.gz"%(dire+"/"+p+"_T1_reg.nii.gz",dire+"/"+p+"_wm_seg_th.nii.gz",dire+"/"+p+"_wm_seg_th") + " --transform %s.tfm"%(dire+p+"_seg_afreg") + " --interpolation GenericLabel ; "      
                    # """                
                    bashcmd = conv_csf.split()  
                    process = subprocess.Popen(conv_csf, universal_newlines=True, shell=True, stdout=sys.stdout,
                                               stderr=subprocess.STDOUT)
                    # wait until finish
                    out, error = process.communicate()
                     
                    csf, seg_affine = load_nifti(csf_pve) 
                    csf[csf<0.6] = 0 # puts wm labels to 0 to keep only csf ones 
                    csf[csf!=0] = 1
                    
                    gm, _ = load_nifti(gm_pve) 
                    sub, _ = load_nifti(gm_pve_sub) 
                    gm = gm + sub
                    gm[gm<0.5] = 0 # puts wm labels to 0 to keep only gm ones 
                    gm[gm!=0] = 1 
                      
                    # wm_bckgd = np.logical_not(np.logical_and(seg,csf)) # gm and NOT csf
                    # wm_bckgd = np.logical_not(seg) # gm and NOT csf
                    # wm_bckgd = binary_dilation(wm_bckgd,iterations=0).astype(np.float64)  
                    # save_nifti(bck_path,wm_bckgd,seg_affine)
                    save_nifti(csf_path_2,csf.astype(np.float64) ,seg_affine)
                    save_nifti(gm_path_2,gm.astype(np.float64) ,seg_affine)
                    
                    
                    conv_csf = "antsApplyTransforms --reference-image %s --input %s --output %s"%(dire+"/"+p+"_T1_reg.nii.gz",csf_path_2,csf_path_2) + " --transform %s.tfm"%(dire+p+"_seg_afreg") + " --interpolation GenericLabel ; "     
                    conv_csf += "antsApplyTransforms --reference-image %s --input %s --output %s"%(dire+"/"+p+"_T1_reg.nii.gz",gm_path_2,gm_path_2) + " --transform %s.tfm"%(dire+p+"_seg_afreg") + " --interpolation GenericLabel ; "   
                    """
                    conv_csf =  "flirt -in {newvol} -ref {refvol} -out {outvol} -init {invol2refvol}.mat -dof 6 -applyxfm -interp nearestneighbour ; ".format(newvol=csf_path_2, 
                                 refvol=ref_t1, outvol=csf_path_2, invol2refvol=dire+p+"_seg_afreg") 
                    conv_csf += "flirt -in {newvol} -ref {refvol} -out {outvol} -init {invol2refvol}.mat -dof 6 -applyxfm -interp nearestneighbour".format(newvol=gm_path_2, 
                                 refvol=ref_t1, outvol=gm_path_2, invol2refvol=dire+p+"_seg_afreg") 
                    """                                           
                    bashcmd = conv_csf.split()  
                    process = subprocess.Popen(conv_csf, universal_newlines=True, shell=True, stdout=sys.stdout,
                                               stderr=subprocess.STDOUT)
                    # wait until finish
                    out, error = process.communicate() 
        
                if starting_step in ("reg_prob"):
                 
                    # preproc = f_path + "subjects/%s/tracking/preproc/%s_dmri_upsampled.nii.gz"%(p,p)
                    preproc = f_path + "subjects/" + patient + "/dMRI/preproc/" + patient + "_dmri_preproc.nii.gz" 
                    data = f_path + "data_2/" if patient.split('_')[0]=="V" else f_path + "data_1/"
                    data = f_path + "subjects/" + patient + "/dMRI/preproc/"
                    fbvec = data + patient + "_dmri_preproc.bvec"
                    fbval = data + patient + "_dmri_preproc.bval"
                    gtab = dpg.gradient_table(fbval, fbvec)
                    mapping = dipy_syn_reg.syn_register_dwi(preproc, gtab, template="/auto/home/users/d/r/drimez/AFQ_data/00Average_Brain.nii.gz")[1]
                    _, affine = load_nifti(preproc)
                    
                    density_path = "/auto/home/users/d/r/drimez/AFQ_data/subjects/%s/"%patient
                    if not os.path.isdir(density_path):
                        os.mkdir(density_path)
                        
                    for prob_path in density_map_files:
                        prob_map, _ = load_nifti(prob_path)
                        warped_prob_map = mapping.transform_inverse(prob_map) # , interpolation='nearest') 
                        this_name = prob_path.split('.')[0].split('/')[-1]
                        save_nifti(density_path + this_name + ".nii.gz", warped_prob_map, affine) 
                        
                    mapping = dipy_syn_reg.syn_register_dwi(preproc, gtab, template="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/atlases_dany/reference_brains/MNI152NLin6_res-1x1x1_T1w.nii.gz")[1]
                    # _, affine = load_nifti("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/atlases_dany/reference_brains/MNI152NLin6_res-1x1x1_T1w.nii.gz")
                    
                    temp_path = "/auto/home/users/d/r/drimez/AFQ_data/templates/"
                    temp_map_files = ["ATR_L_prob_map.nii.gz", "ATR_R_prob_map.nii.gz", "CGC_L_prob_map.nii.gz", 
                                      "CGC_R_prob_map.nii.gz", "FA_prob_map.nii.gz", "FP_prob_map.nii.gz", 
                                      "HCC_L_prob_map.nii.gz", "HCC_R_prob_map.nii.gz", "SLF_L_prob_map.nii.gz", 
                                      "SLF_R_prob_map.nii.gz"]
                    for prob_path in temp_map_files:
                        try:
                            prob_map, _ = load_nifti(temp_path+prob_path)
                            warped_prob_map = mapping.transform_inverse(prob_map.astype(float)) # , interpolation='nearest') 
                            this_name = prob_path.split('.')[0].split('/')[-1]
                            save_nifti(density_path + this_name + ".nii.gz", warped_prob_map, affine) 
                        except Exception as err:
                            print(err)
                            pass
        
                fod = new_path + "FOD/" + patient 
                if starting_step in ("fod"):  
                    ######## FOD
                    if not os.path.isdir(new_path + "FOD/"):
                        os.mkdir(new_path + "FOD/") 
                     
                    data = f_path + "data_1/" if patient.split('_')[0]=="H" else f_path + "data_2/"
                    bvecs = data + patient + ".bvec"
                    bvals = data + patient + ".bval"
                    
                    rep = "dwi2response tournier " + dmri + ".mif.gz " + fod + "_response.txt -debug -fslgrad " + bvecs + " " + bvals  
                    step_2 = rep + " ; dwi2fod -fslgrad " + bvecs + " " + bvals + " csd " + dmri + ".mif.gz " + fod + "_response.txt " + fod + "_fod.mif.gz"  
                    
                    bashcmd = step_2.split()  
                    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": starting FOD step\n")
                    f = open(new_path + "/logs.txt", "a+")
                    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": starting FOD step\n")
                    f.close()
                
                    process = subprocess.Popen(step_2, universal_newlines=True, shell=True, stdout=sys.stdout,
                                               stderr=subprocess.STDOUT)
                    # wait until finish
                    out, error = process.communicate()
         
                ####### Tracking 
                if not os.path.isdir(new_path + "tckgen/"):
                    os.mkdir(new_path + "tckgen/")
                        
                algo = tracking_algo
                step_3 = "tckgen -algorithm " + algo + " -angle 45 -select 200000 -step 0.2 -minlength 20 -act " + new_path + "preproc/" + patient + "_5TT.mif.gz -crop_at_gmwmi -debug -seed_dynamic " + fod + "_fod.mif.gz" # for dynamic seeding 
                # step_3 = step_3 if wm_mask in (None,'None') else step_3 + " -seed_image " + wm + ".mif"
                step_3 = step_3 + " -backtrack -rk4" if algo in ("iFOD2","Tensor_Prob") else step_3
                input_name = fod + "_fod.mif.gz" if algo in ("iFOD2","SD_STREAM") else dmri + ".mif.gz -rk4"
                output_name = new_path + patient + "_" + algo 
                output = output_name + ".tck"
                step_3 = step_3 + " " + input_name + " " + output 
                
                if starting_step in ("track"):
                    if os.path.exists(output): # moves previous results to backup folder
                        backup(output,patient)
             
                    bashcmd = step_3.split()  
                    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": starting tracking step\n")
                    f = open(new_path + "/logs.txt", "a+")
                    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": starting tracking step\n")
                    f.close()
                
                    process = subprocess.Popen(step_3, universal_newlines=True, shell=True, stdout=sys.stdout,
                                               stderr=subprocess.STDOUT)
                    # wait until finish
                    out, error = process.communicate()
             
                if starting_step in ("sift"): 
                    if not os.path.isdir(new_path + "sift/"):
                        os.mkdir(new_path + "sift/")
                
                    ######## SIFT
                    step_4 = "tcksift " + output + " " + fod + "_fod.mif.gz " + output_name + "_sift.tck -term_number " + str(n_streams) + " -csv " + new_path + "sift.csv -debug -force"  
            
                    bashcmd = step_4.split()  
                    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": starting SIFT step\n")
                    f = open(new_path + "/logs.txt", "a+")
                    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": starting SIFT step\n")
                    f.close()
                    
                    old_path = os.getcwd()
                    os.system('cd ' + new_path + "sift/")
                    process = subprocess.Popen(step_4, universal_newlines=True, shell=True, stdout=sys.stdout,
                                               stderr=subprocess.STDOUT)
                    # wait until finish
                    out, error = process.communicate()
                    os.system('cd ' + old_path)
         
                if starting_step in ("whole"): 
            	      ######### Select only those streamline vertices within a mask
                    step_5 = "tckedit " + output_name + "_sift.tck " + output_name + "_sift_cropped.tck -mask " +  wm  + ".mif.gz -force"  
            
                    bashcmd = step_5.split()  
                    process = subprocess.Popen(step_5, universal_newlines=True, shell=True, stdout=sys.stdout,
                                               stderr=subprocess.STDOUT)
                    # wait until finish
                    out, error = process.communicate()
         
                    ######### Stats
                    if not os.path.isdir(new_path + "stats/"):
                        os.mkdir(new_path + "stats/")
                    
                    step_6 = "tckstats " + output_name + "_sift_cropped.tck -dump " + new_path + "stats/" + patient + "_whole.txt -output mean,median,std,min,max,count -debug -force"  
            
                    bashcmd = step_6.split()  
                    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": starting whole brain stats step\n")
                    f = open(new_path + "/logs.txt", "a+")
                    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": starting whole brain stats step\n")
                    f.close()
                
                    process = subprocess.Popen(step_6, universal_newlines=True, shell=True, stdout=sys.stdout,
                                               stderr=subprocess.STDOUT)
                    # wait until finish
                    out, error = process.communicate()
                # print(patient+" done",file=open("/auto/home/users/d/r/drimez/PREPROC2.txt","a"))
                if starting_step in ("rois") and ( (not os.path.exists(new_path + "ROIs/metrics.csv")) or refresh):  
                    
                    if not os.path.isdir(new_path + "ROIs/"):
                        os.mkdir(new_path + "ROIs/")
                    
                  	######### Extract streamlines based on white matter mask & corresponding microstructure stats 
                    step_7_prefix = "tckedit " + output_name + "_sift_cropped.tck "
                    step_7_suffix = ".txt -output mean,median,std,min,max,count -debug -force ; " 
                    output_name = new_path + "ROIs/" + patient + "_"  
                    
                    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": starting ROI analysis\n")
                    f = open(new_path + "/logs.txt", "a+")
                    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": starting ROI analysis\n")
                    f.close()
                     
                    if parrallel:
                        print(patient,file=open("/auto/home/users/d/r/drimez/rois.txt","a"))
                     
                    wm_parc_path  = new_path + "preproc/" + patient + "_reg_wmparc.nii.gz" 
                    wm_parc, wm_affine = load_nifti(wm_parc_path)   
                    roi = cp.copy(wm_parc) ; conv = "" 
                    labels = pd.read_csv("/auto/home/users/d/r/drimez/LUT.txt",sep="  ",header=None,index_col=False,names=["id","name","r","g","b","a"])
                    dti_path = f_path + "/subjects/%s/dMRI/microstructure/dti/%s_"%(patient,patient)
                    odi_path = f_path + "/subjects/%s/dMRI/microstructure/noddi/%s_noddi_"%(patient,patient)
                    fa, _ = load_nifti(dti_path + "FA.nii.gz") 
                    md, _ = load_nifti(dti_path + "MD.nii.gz") 
                    ad, _ = load_nifti(dti_path + "AD.nii.gz") 
                    rd, _ = load_nifti(dti_path + "RD.nii.gz") 
                    
                    noddi = False
                    if noddi: 
                        fintra, _ = load_nifti(odi_path + "fintra.nii.gz")
                        fextra, _ = load_nifti(odi_path + "fextra.nii.gz")
                        fiso, _ = load_nifti(odi_path + "fiso.nii.gz")
                        odi, _ = load_nifti(odi_path + "odi.nii.gz")
                        metrics = {"subjectID":[],"tractID":[],"nodeID":[],"FA":[],"MD":[],"AD":[],"RD":[],"fintra":[],"fextra":[],"fiso":[],"odi":[]} 
                    else:
                        metrics = {"subjectID":[],"tractID":[],"nodeID":[],"FA":[],"MD":[],"AD":[],"RD":[]}
                         
                    precison = 8
                    def sample_rois(label_id,precison=precison,wm_parc=wm_parc,labels=labels,mmetrics=metrics): 
                        if not label_id in labels["id"].values:
                            print("label %s not found!\n"%label_id)
                            f = open(f_path + "subjects/" + patient + "/tracking/logs.txt", "a+")
                            f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                                "%d.%b %Y %H:%M:%S") + "label %s not found!\n"%label_id)
                            f.close() 
                            return None
                        else:
                            label_name = labels["name"][labels["id"]==label_id].values[0]
                            metrics = cp.copy(mmetrics)
                            metrics["subjectID"].append(patient)
                            metrics["tractID"].append(label_name)
                            metrics["nodeID"].append(0) # no subgroups in ROIs
                            metrics["FA"].append((np.round(bootstrap(fa[wm_parc==label_id],np.mean),precison), np.round(bootstrap(fa[wm_parc==label_id],np.std),precison)))  
                            metrics["MD"].append((np.round(bootstrap(md[wm_parc==label_id],np.mean),precison), np.round(bootstrap(md[wm_parc==label_id],np.std),precison)))
                            metrics["AD"].append((np.round(bootstrap(ad[wm_parc==label_id],np.mean),precison), np.round(bootstrap(ad[wm_parc==label_id],np.std),precison)))
                            metrics["RD"].append((np.round(bootstrap(rd[wm_parc==label_id],np.mean),precison), np.round(bootstrap(rd[wm_parc==label_id],np.std),precison)))
                            if noddi: 
                                metrics["fintra"].append((np.round(bootstrap(fintra[wm_parc==label_id],np.mean),precison), np.round(bootstrap(fintra[wm_parc==label_id],np.std),precison)))
                                metrics["fextra"].append((np.round(bootstrap(fextra[wm_parc==label_id],np.mean),precison), np.round(bootstrap(fextra[wm_parc==label_id],np.std),precison)))
                                metrics["fiso"].append((np.round(bootstrap(fiso[wm_parc==label_id],np.mean),precison), np.round(bootstrap(fiso[wm_parc==label_id],np.std),precison)))
                                metrics["odi"].append((np.round(bootstrap(odi[wm_parc==label_id],np.mean),precison), np.round(bootstrap(odi[wm_parc==label_id],np.std),precison)))
                            return mmetrics
                            
                    dfs = Parallel(n_jobs=n_jobs,verbose=100,pre_dispatch='2*n_jobs')( delayed(sample_rois)(label_id) 
                                                                                       for label_id in np.unique(wm_parc.flatten()).astype(np.int64)[1:] )  
                    dfs_to_use = [pd.DataFrame(ddf) for ddf in dfs if ddf is not None]
                    
                    pd.concat(dfs_to_use,axis=0,ignore_index=True).to_csv(new_path + "ROIs/metrics.csv", line_terminator="\n", sep=",", index=False)
                    
                if starting_step in ("tck2trk"):
                    ###### convert tck files to trk files
                    """
                    with os.scandir(new_path) as it:
                        recursor(new_path,patient,it)  
                        it.close()
                    
                    tck2trk = mrt.MRTrix2TrackVis()
                    tck2trk.inputs.in_file = output_name + "_sift_cropped.tck"
                    tck2trk.inputs.out_filename = output_name + "_sift_cropped.trk"
                    tck2trk.inputs.image_file = f_path + "subjects/H_0/dMRI/preproc/" + patient + "_dmri_preproc.nii.gz"
                    tck2trk.run()     
                    """
                    tractogram = load_tck(output_name + ".tck", f_path + "subjects/" + patient + "/dMRI/preproc/" + patient + "_dmri_preproc.nii.gz")
                    saved = save_trk(tractogram, output_name + ".trk")
                    os.system('rm ' + output_name + ".tck")
                    tractogram = load_tck(output_name + "_sift.tck", f_path + "subjects/" + patient + "/dMRI/preproc/" + patient + "_dmri_preproc.nii.gz")
                    saved = save_trk(tractogram, output_name + "_sift.trk")
                    os.system('rm ' + output_name + "_sift.tck")
                    tractogram = load_tck(output_name + "_sift_cropped.tck", f_path + "subjects/" + patient + "/dMRI/preproc/" + patient + "_dmri_preproc.nii.gz")
                    saved = save_trk(tractogram, output_name + "_sift_cropped.trk")
                    os.system('rm ' + output_name + "_sift_cropped.tck")
                        
                if starting_step in ("dipy"):
                    
                    if not os.path.isdir(new_path + "FOD/"):
                        os.mkdir(new_path + "FOD/") 
                         
                    for model in ["_prob",""]: # "", "_prob",
                        dipy_tracto(patient,model=model,filtering=filtering)
                    
                    """
                    Parallel(n_jobs=-1,verbose=50)(delayed(dipy_tracto)(patient,model=model,filtering=filtering) for model in ["_prob"])
                    """
                        
                if starting_step in ("sample"):
                    if not os.path.isdir(new_path + "AFQ/stats/"):
                        os.mkdir(new_path + "AFQ/stats/")
                        
                    ######### Sample microstructure images along tracks 
                    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": starting sampling\n")
                    f = open(new_path + "/logs.txt", "a+")
                    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
                        "%d.%b %Y %H:%M:%S") + ": starting sampling\n")
                    f.close()
                    
                    step_8 = ""
                    dti_path = f_path + "/subjects/%s/dMRI/microstructure/dti/%s_"%(patient,patient)
                    odi_path = f_path + "/subjects/%s/dMRI/microstructure/noddi/%s_noddi_"%(patient,patient)
                    for mod in ["","_prob"]:
                        tracks_path = new_path + "AFQ/tracks" + mod + '/'
                        with os.scandir(tracks_path) as iterator:
                            for entry in iterator:
                                tracks = entry.name
                                output_name = new_path + "AFQ/stats/" + tracks + "_"
                                metrics = ["FA","MD","AD","RD"]
                                for img in metrics:
                                    step_8 = step_8 + "tcksample " + tracks + ".tck " + dti_path + img + ".nii.gz " + output_name + img + ".tsf ; " 
                                metrics = ["fbundle","fintra","fextra","fiso","odi"]
                                for img in metrics:
                                    step_8 = step_8 + "tcksample " + tracks + ".tck " + odi_path + img + ".nii.gz " + output_name + img + ".tsf ; " 
                            iterator.close()
                     
                    metrics = {"subjectID":[],"tractID":[],"nodeID":[],"FA":[],"MD":[],"AD":[],"RD":[],"fbundle":[],"fintra":[],"fextra":[],"fiso":[],"odi":[]} 
                    precison = 8
                    for label_id in np.unique(wm_parc.flatten()).astype(np.int64)[1:]:  
                    
                        label_name = labels["name"][labels["id"]==label_id].values[0]
                        metrics["subjectID"].append(patient)
                        metrics["tractID"].append(label_name)
                        metrics["nodeID"].append(0) # no subgroups in ROIs
                        metrics["FA"].append((fa[wm_parc==label_id].mean().round(precison), fa[wm_parc==label_id].std().round(precison)))
                        metrics["MD"].append((md[wm_parc==label_id].mean().round(precison), md[wm_parc==label_id].std().round(precison)))
                        metrics["AD"].append((ad[wm_parc==label_id].mean().round(precison), ad[wm_parc==label_id].std().round(precison)))
                        metrics["RD"].append((rd[wm_parc==label_id].mean().round(precison), rd[wm_parc==label_id].std().round(precison)))
                        metrics["fbundle"].append((fbundle[wm_parc==label_id].mean().round(precison), fbundle[wm_parc==label_id].std().round(precison)))
                        metrics["fintra"].append((fintra[wm_parc==label_id].mean().round(precison), fintra[wm_parc==label_id].std().round(precison)))
                        metrics["fextra"].append((fextra[wm_parc==label_id].mean().round(precison), fextra[wm_parc==label_id].std().round(precison)))
                        metrics["fiso"].append((fiso[wm_parc==label_id].mean().round(precison), fiso[wm_parc==label_id].std().round(precison)))
                        metrics["odi"].append((odi[wm_parc==label_id].mean().round(precison), odi[wm_parc==label_id].std().round(precison)))
                         
                    pd.DataFrame(metrics).to_csv(new_path + "ROIs/metrics.csv", line_terminator="\n", sep=",", index=False)
                 
if __name__ == "__main__": 
    """
    sphere = get_sphere("symmetric724")
    x,y,z = sphere.x, sphere.y, sphere.z
    r, els, azs = cart2sphere(x,y,z) 
    with open("/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/H_0/tracking/FOD/sphere.txt","w") as writer:
        for lign in range(len(els)):
            writer.write(str(azs[lign]) + " " + str(els[lign]) + "\n")
        writer.close()   
    """
    
    function = None ; args = None
    if len(sys.argv)>2:
        function = str(sys.argv[1]) 
        args = [str(arg) for arg in sys.argv[2:]] 
    elif len(sys.argv)>1: 
        args = [str(arg) for arg in sys.argv[2:]]
    else:
        args = [f_path,patient_list]
     
    # assert function in (None,"generate_rois","dipy_tracto"), "Typo mistake :/"
    if function == "generate_rois":
        generate_rois(*args)
    elif function == "dipy_tracto": 
        dipy_tracto(*args)
    else:
        tracto(*args)    
        
    """
    generate_rois(f_path=f_path,p="H_0",wm_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/H_0/tracking/preproc/H_0_csf_mask.nii.gz",reg="affine",mov=None,prefix="_atlas")  
    """
 

"""      
for p in H_0 H_1 H_2 H_3 H_4  
do
	dwi2response tournier /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/$p/tracking/preproc/$p_dmri_upsampled.nii.gz /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/$p/tracking/FOD/$p_mrtrix_response.txt -mask /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/$p/tracking/preproc/$p_wm_new.nii.gz -fslgrad /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/data_1/$p.bvec /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/data_1/$p.bval -lmax 10
done
for_each * : dwi2response tax /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/H_0/tracking/preproc/H_0_dmri_upsampled.nii.gz /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/H_0/tracking/FOD/H_0_mrtrix_response.txt -mask /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/H_0/tracking/preproc/H_0_wm_new.nii.gz -fslgrad /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/data_1/H_0.bvec /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/data_1/H_0.bval -lmax 10 -peak_ratio 0.01 -force
responsemean */response.txt ../group_average_response.txt
dwi2fod csd /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/H_0/tracking/preproc/H_0_dmri_upsampled.nii.gz /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/H_0/tracking/FOD/H_0_mrtrix_response.txt /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/H_0/tracking/FOD/H_0_mrtrix_fod.nii.gz -lmax 10 -mask /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/H_0/tracking/preproc/H_0_wm_new.nii.gz -fslgrad /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/data_1/H_0.bvec /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/data_1/H_0.bval -directions /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/subjects/H_0/tracking/FOD/sphere.txt -force
tcksift H_0_dipy_prob_cleaned.tck FOD/H_0_mrtrix_fod.nii.gz H_0_dipy_prob_sift.tck -term_number 
"""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
