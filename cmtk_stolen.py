# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 23:24:04 2022
functions from cmtk
@author: rimez
"""
from dipy.io.image import load_nifti_data, load_nifti, save_nifti
import os
import subprocess
import dipy
import nibabel as nib
import numpy as np
import nibabel.trackvis as tv
from dipy.align.streamlinear import length

def compute_length_array(trkfile=None, streams=None, savefname=None):
    """Computes the length of the fibers in a tractogram and returns an array of length.
    Parameters
    ----------
    trkfile : TRK file
        Path to the tractogram in TRK format
    streams : the fibers data
        The fibers from which we want to compute the length
    savefname : string
        Output filename to write the length array
    Returns
    -------
    fibers_length : numpy.array
        Array of fiber lengths
    """
    if streams is None and trkfile is not None:
        print(f'Compute length array for fibers in {trkfile}')
        streams, hdr = tv.read(trkfile, as_generator=True)
        n_fibers = hdr["n_count"]
        if n_fibers == 0:
            msg = (
                f'Header field n_count of trackfile {trkfile} is set to 0. '
                "No track seem to exist in this file."
            )
            print(msg)
            raise Exception(msg)
    else:
        n_fibers = len(streams)

    fibers_length = np.zeros(n_fibers, dtype=np.float)
    for i, fib in enumerate(streams):
        fibers_length[i] = length(fib[0])

    # store length array

    np.save(savefname, fibers_length)
    print(f'Store lengths array to: {savefname}')

    return fibers_length

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    OutputMultiPath,
    InputMultiPath,
)

class ExtractPVEsFrom5TTInputSpec(BaseInterfaceInputSpec):
    in_5tt = File(desc="Input 5TT (4D) image", exists=True, mandatory=True)

    ref_image = File(
        desc="Reference 3D image to be used to save 3D PVE volumes",
        exists=True,
        mandatory=True,
    )

    pve_csf_file = File(
        desc="CSF Partial Volume Estimation volume estimated from", mandatory=True
    )

    pve_gm_file = File(
        desc="GM Partial Volume Estimation volume estimated from", mandatory=True
    )

    pve_wm_file = File(
        desc="WM Partial Volume Estimation volume estimated from", mandatory=True
    )


class ExtractPVEsFrom5TTOutputSpec(TraitedSpec):
    partial_volume_files = OutputMultiPath(
        File,
        desc="CSF/GM/WM Partial Volume Estimation images estimated from",
        exists=True,
    )

class ExtractPVEsFrom5TT(BaseInterface):
    """Create Partial Volume Estimation maps for CSF, GM, WM tissues from `mrtrix3` 5TT image.
    Examples
    --------
    >>> from cmtklib.diffusion import ExtractPVEsFrom5TT
    >>> pves = ExtractPVEsFrom5TT()
    >>> pves.inputs.in_5tt = 'sub-01_desc-5tt_dseg.nii.gz'
    >>> pves.inputs.ref_image = 'sub-01_T1w.nii.gz'
    >>> pves.inputs.pve_csf_file = '/path/to/output_csf_pve.nii.gz'
    >>> pves.inputs.pve_gm_file = '/path/to/output_gm_pve.nii.gz'
    >>> pves.inputs.pve_wm_file = '/path/to/output_wm_pve.nii.gz'
    >>> pves.run()  # doctest: +SKIP
    """
    
    input_spec = ExtractPVEsFrom5TTInputSpec
    output_spec = ExtractPVEsFrom5TTOutputSpec
  
    def _run_interface(self, runtime):
        img_5tt = nib.load(self.inputs.in_5tt)
        data_5tt = img_5tt.get_data()

        ref_img = nib.load(self.inputs.ref_image)
        # hdr = ref_img.get_header()
        affine = ref_img.get_affine()

        print("Shape : {}".format(data_5tt.shape))

        # The tissue type volumes must appear in the following order for the anatomical priors to be applied correctly during tractography:
        #
        # 0: Cortical grey matter
        # 1: Sub-cortical grey matter
        # 2: White matter
        # 3: CSF
        # 4: Pathological tissue
        #
        # Extract from https://mrtrix.readthedocs.io/en/latest/quantitative_structural_connectivity/act.html

        # Create and save PVE for CSF
        pve_csf = data_5tt[:, :, :, 3].squeeze()
        pve_csf_img = nib.Nifti1Image(pve_csf.astype(np.float), affine)
        nib.save(pve_csf_img, os.path.abspath(self.inputs.pve_csf_file))

        # Create and save PVE for WM
        pve_wm = data_5tt[:, :, :, 2].squeeze()
        pve_wm_img = nib.Nifti1Image(pve_wm.astype(np.float), affine)
        nib.save(pve_wm_img, os.path.abspath(self.inputs.pve_wm_file))

        # Create and save PVE for GM
        pve_gm = data_5tt[:, :, :, 0].squeeze() + data_5tt[:, :, :, 1].squeeze()
        pve_gm_img = nib.Nifti1Image(pve_gm.astype(np.float), affine)
        nib.save(pve_gm_img, os.path.abspath(self.inputs.pve_gm_file))

        # Dilate PVEs and normalize to 1
        fwhm = 2.0
        radius = np.float(0.5 * fwhm)
        sigma = np.float(fwhm / 2.3548)

        print("sigma : %s" % sigma)

        fslmaths_cmd = "fslmaths %s -kernel sphere %s -dilD %s" % (
            os.path.abspath(self.inputs.pve_csf_file),
            radius,
            os.path.abspath(self.inputs.pve_csf_file),
        )
        print("Dilate CSF PVE")
        print(fslmaths_cmd)
        process = subprocess.Popen(
            fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        _ = process.communicate()[0].strip()

        fslmaths_cmd = "fslmaths %s -kernel sphere %s -dilD %s" % (
            os.path.abspath(self.inputs.pve_wm_file),
            radius,
            os.path.abspath(self.inputs.pve_wm_file),
        )
        print("Dilate WM PVE")
        print(fslmaths_cmd)
        process = subprocess.Popen(
            fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        _ = process.communicate()[0].strip()

        fslmaths_cmd = "fslmaths %s -kernel sphere %s -dilD %s" % (
            os.path.abspath(self.inputs.pve_gm_file),
            radius,
            os.path.abspath(self.inputs.pve_gm_file),
        )
        print("Dilate GM PVE")
        print(fslmaths_cmd)
        process = subprocess.Popen(
            fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        _ = process.communicate()[0].strip()

        fslmaths_cmd = "fslmaths %s -kernel gauss %s -fmean %s" % (
            os.path.abspath(self.inputs.pve_csf_file),
            sigma,
            os.path.abspath(self.inputs.pve_csf_file),
        )
        print("Gaussian smoothing : CSF PVE")
        print(fslmaths_cmd)
        process = subprocess.Popen(
            fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        _ = process.communicate()[0].strip()

        fslmaths_cmd = "fslmaths %s -kernel gauss %s -fmean %s" % (
            os.path.abspath(self.inputs.pve_wm_file),
            sigma,
            os.path.abspath(self.inputs.pve_wm_file),
        )
        print("Gaussian smoothing : WM PVE")
        print(fslmaths_cmd)
        process = subprocess.Popen(
            fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        _ = process.communicate()[0].strip()

        fslmaths_cmd = "fslmaths %s -kernel gauss %s -fmean %s" % (
            os.path.abspath(self.inputs.pve_gm_file),
            sigma,
            os.path.abspath(self.inputs.pve_gm_file),
        )
        print("Gaussian smoothing : GM PVE")
        print(fslmaths_cmd)
        process = subprocess.Popen(
            fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        _ = process.communicate()[0].strip()

        pve_csf = nib.load(os.path.abspath(self.inputs.pve_csf_file)).get_data()
        pve_wm = nib.load(os.path.abspath(self.inputs.pve_wm_file)).get_data()
        pve_gm = nib.load(os.path.abspath(self.inputs.pve_gm_file)).get_data()

        pve_sum = pve_csf + pve_wm + pve_gm
        pve_csf = np.divide(pve_csf, pve_sum)
        pve_wm = np.divide(pve_wm, pve_sum)
        pve_gm = np.divide(pve_gm, pve_sum)

        pve_csf_img = nib.Nifti1Image(pve_csf.astype(np.float), affine)
        nib.save(pve_csf_img, os.path.abspath(self.inputs.pve_csf_file))

        pve_wm_img = nib.Nifti1Image(pve_wm.astype(np.float), affine)
        nib.save(pve_wm_img, os.path.abspath(self.inputs.pve_wm_file))

        pve_gm_img = nib.Nifti1Image(pve_gm.astype(np.float), affine)
        nib.save(pve_gm_img, os.path.abspath(self.inputs.pve_gm_file))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["partial_volume_files"] = [
            os.path.abspath(self.inputs.pve_csf_file),
            os.path.abspath(self.inputs.pve_gm_file),
            os.path.abspath(self.inputs.pve_wm_file)
        ]
        return outputs
        
        
        
def conv_5TT2nii(path_5tt,pve_csf_file,pve_wm_file,pve_gm_file,ref,p):
  
    conv_csf  = "mrconvert " + path_5tt + " -coord 3 3 -axes 0,1,2 " + pve_csf_file + " -force ; "
    conv_csf += "mrconvert " + path_5tt + " -coord 3 2 -axes 0,1,2 " + pve_wm_file + " -force ; "
    conv_csf += "mrconvert " + path_5tt + " -coord 3 1 -axes 0,1,2 " + pve_gm_file.split(".")[0] + "_sub.nii.gz -force ; "
    conv_csf += "mrconvert " + path_5tt + " -coord 3 0 -axes 0,1,2 " + pve_gm_file + " -force "
    conv_csf += "; fslmaths " + pve_wm_file + " -add " + pve_gm_file.split(".")[0] + "_sub.nii.gz " + pve_wm_file.split(".")[0] + "_dil.nii.gz"
    
    process = subprocess.Popen(
        conv_csf, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    _ = process.communicate()[0].strip()
     
    # Dilate PVEs and normalize to 1
    fwhm = 2.0
    radius = np.float(0.5 * fwhm)
    sigma = np.float(fwhm / 2.3548)

    print("sigma : %s" % sigma)

    """
    fslmaths_cmd = "fslmaths %s -kernel sphere %s -dilD %s" % (
        os.path.abspath(pve_csf_file),
        radius,
        os.path.abspath(pve_csf_file),
    )
    
    print("Dilate CSF PVE")
    print(fslmaths_cmd)
    process = subprocess.Popen(
        fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    _ = process.communicate()[0].strip()

    """
    fslmaths_cmd = "fslmaths %s -kernel sphere %s -dilD %s" % (
        os.path.abspath(pve_wm_file.split(".")[0]+"_dil.nii.gz"),
        radius,
        os.path.abspath(pve_wm_file.split(".")[0]+"_dil.nii.gz"),
    )
    
    print("Dilate WM PVE")
    print(fslmaths_cmd)
    process = subprocess.Popen(
        fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    _ = process.communicate()[0].strip()

    fslmaths_cmd = "fslmaths %s -kernel sphere %s -dilD %s" % (
        os.path.abspath(pve_gm_file),
        radius,
        os.path.abspath(pve_gm_file),
    )
    
    print("Dilate GM PVE")
    print(fslmaths_cmd)
    process = subprocess.Popen(
        fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    _ = process.communicate()[0].strip()
    
    
    csf, affine = load_nifti(pve_csf_file)
    not_csf = np.zeros_like(csf)
    not_csf[csf==0] = 1
    save_nifti(pve_csf_file,not_csf,affine)
     
    fslmaths_cmd = "fslmaths %s -kernel gauss %s -fmean %s" % (
        os.path.abspath(pve_csf_file),
        sigma,
        os.path.abspath(pve_csf_file),
    )
    print("Gaussian smoothing : CSF PVE")
    print(fslmaths_cmd)
    process = subprocess.Popen(
        fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    _ = process.communicate()[0].strip()
     
    not_csf, affine = load_nifti(pve_csf_file)
    not_csf -= not_csf.min()
    not_csf /= not_csf.max()
    csf = np.zeros_like(not_csf)
    csf = 1 - not_csf
    save_nifti(pve_csf_file,csf,affine)
 
    fslmaths_cmd = "fslmaths %s -kernel gauss %s -fmean %s" % (
        os.path.abspath(pve_wm_file),
        sigma,
        os.path.abspath(pve_wm_file.split(".")[0]+"_dil.nii.gz"),
    ) 
    print("Gaussian smoothing : WM PVE")
    print(fslmaths_cmd)
    process = subprocess.Popen(
        fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    _ = process.communicate()[0].strip()
 
    fslmaths_cmd = "fslmaths %s -kernel gauss %s -fmean %s" % (
        os.path.abspath(pve_gm_file),
        sigma,
        os.path.abspath(pve_gm_file),
    )
    print("Gaussian smoothing : GM PVE")
    print(fslmaths_cmd)
    process = subprocess.Popen(
        fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    _ = process.communicate()[0].strip()
    """ 
    fslmaths_cmd = "fslmaths %s -kernel gauss %s -fmean %s" % (
        os.path.abspath(pve_gm_file.split(".")[0] + "_sub.nii.gz"),
        radius/2,
        os.path.abspath(pve_gm_file.split(".")[0] + "_sub.nii.gz"),
    )
    
    # fslmaths_cmd += " ; fslmaths " + pve_gm_file + " -add " + pve_gm_file.split(".")[0] + "_sub.nii.gz " + pve_gm_file
    
    print("Dilate subGM PVE")
    print(fslmaths_cmd)
    process = subprocess.Popen(
        fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    _ = process.communicate()[0].strip()
    """
    """
    dire = "/".join(path_5tt.split('/')[:-1]) + "/" + p
    
    conv_csf = "flirt -in {newvol} -ref {refvol} -out {outvol} -omat {invol2refvol}.mat -dof 6 -applyxfm".format(newvol=pve_csf_file, 
                 refvol=ref, outvol=pve_csf_file, invol2refvol=dire+"_afreg.mat")
    conv_csf += "flirt -in {newvol} -ref {refvol} -out {outvol} -omat {invol2refvol}.mat -dof 6 -applyxfm".format(newvol=pve_gm_file, 
                 refvol=ref, outvol=pve_gm_file, invol2refvol=dire+"_afreg.mat")
    conv_csf += "flirt -in {newvol} -ref {refvol} -out {outvol} -omat {invol2refvol}.mat -dof 6 -applyxfm".format(newvol=pve_wm_file, 
                 refvol=ref, outvol=pve_wm_file, invol2refvol=dire+"_afreg.mat")
                                                
    process = subprocess.Popen(
        conv_csf, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    _ = process.communicate()[0].strip()                                            
    """ 
     
"""       
def dilate(data):       
        
    # Dilate PVEs and normalize to 1
    fwhm = 2.0
    radius = np.float(0.5 * fwhm)
    sigma = np.float(fwhm / 2.3548)

    print("sigma : %s" % sigma)

    fslmaths_cmd = "fslmaths %s -kernel sphere %s -dilD %s" % (
        os.path.abspath(pve_csf_file),
        radius,
        os.path.abspath(pve_csf_file),
    )
    print("Dilate CSF PVE")
    print(fslmaths_cmd)
    process = subprocess.Popen(
        fslmaths_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    _ = process.communicate()[0].strip()
""" 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
