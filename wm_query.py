from dipy.io.streamline import load_trk, save_trk, save_vtk, load_vtk
from dipy.io.stateful_tractogram import Space, StatefulTractogram
import subprocess 
import datetime
import sys
import os

f_path = "/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/PROJECT/"
f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset/PROJECT_old/" 
f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset_new/PROJECT/" 
f_path="/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/elikopy_subset_new2/PROJECT/" 
patient = "V_25" ; mod = ""
 
def query(mod,merge=False,single_bdle__=None,patient=patient):

    wmparc = f_path + "subjects/" + patient + "/tracking/preproc/" + patient + "_reg_wmparc"
     
    if not os.path.isdir(f_path + "subjects/" + patient + "/tracking/query/"):
        os.mkdir(f_path + "subjects/" + patient + "/tracking/query/")
    if not os.path.isdir(f_path + "subjects/" + patient + "/tracking/query/dipy" + mod + "/"):
        os.mkdir(f_path + "subjects/" + patient + "/tracking/query/dipy" + mod + "/")
        
    track_name_in = f_path + "subjects/" + patient + "/tracking/" + patient + "_dipy"  + mod
    track_name_out = f_path + "subjects/" + patient + "/tracking/query/dipy" + mod + "/" + patient + "_dipy" + mod
    
    # tractogram = load_trk(track_name_in + ".trk", f_path + "subjects/%s/dMRI/preproc/"%patient + patient + "_dmri_preproc.nii.gz")
    # save_vtk(tractogram, track_name_in + ".vtk")
        
    if not (single_bdle__ in (None,"None","'None'")):  
        track_name_out = f_path + "subjects/%s/tracking/Solo/query"%patient + mod + "/" + patient + "_dipy" + mod   
        if not os.path.isdir(f_path + "subjects/%s/tracking/Solo/query"%patient + mod + "/"):
            os.mkdir(f_path + "subjects/%s/tracking/Solo/query"%patient + mod + "/")
    
    wm_query_cmd = "tract_querier -t " + track_name_in + ".trk -a " + wmparc + ".nii.gz -q /auto/home/users/d/r/drimez/wmql_tracts.qry -o " + track_name_out + ".trk"
    
    log_prefix = "WM query"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": WM query launched for " + "dipy" + mod + " tracts\n")
    f = open(f_path + "subjects/" + patient + "/tracking/logs.txt", "a+")
    f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": WM query launched for " + "dipy" + mod + " tracts\n" + wm_query_cmd)
    f.close() 
    
    bashcmd = wm_query_cmd.split()
    process = subprocess.Popen(wm_query_cmd, universal_newlines=True, shell=True, stdout=sys.stdout, stderr=sys.stderr)
    # wait until finish
    out, error = process.communicate()
     
    # convert files into trk
    """
    trk = None 
    with os.scandir(track_name_out) as iterator: 
        for entry in iterator:
            if entry.name.split(".")[-1]=="vtk":
                tractogram = load_vtk(f_path + "subjects/" + patient + "/tracking/query/dipy" + mod + "/" + entry.name,
                                      f_path + "subjects/%s/dMRI/preproc/"%patient + patient + "_dmri_preproc.nii.gz")
                save_trk(tractogram, f_path + "subjects/" + patient + "/tracking/query/dipy" + mod + "/" + ".".join(entry.name.split(".")[:-1]) + ".trk")
                os.system("rm " + f_path + "subjects/" + patient + "/tracking/query/dipy" + mod + "/" + entry.name)
            if merge and not entry.name.split(".")[-1]=="vtk" and not entry.name.split(".")[0] in ("EP","PT_central","PT_ant","PT_post","PT_cingulate","IFOF","temporal"):
                tractogram = load_trk(f_path + "subjects/" + patient + "/tracking/query/dipy" + mod + "/" + ".".join(entry.name.split(".")[:-1]) + ".trk",
                                      f_path + "subjects/%s/dMRI/preproc/"%patient + patient + "_dmri_preproc.nii.gz")
                if trk is None:
                    trk = tractogram
                else:
                    trk = StatefulTractogram(  list(trk.streamlines) + list(tractogram.streamlines),
                                               f_path + "subjects/%s/dMRI/preproc/"%patient + patient + "_dmri_preproc.nii.gz",
                                               Space.RASMM,
                                               data_per_streamline={k: (list(trk.data_per_streamline[k]) + list(tractogram.data_per_streamline[k]))
                                                                        for k in tractogram.data_per_streamline.keys()  }) 
        if not trk is None:
            save_trk(trk, f_path + "subjects/" + patient + "/tracking/" + patient + "_dipy" + mod + "_cleaned.trk")
            
        iterator.close()
    """
    """
    else:
        wm_query_cmd = "tract_querier -t " + track_name_in + ".trk -a " + wmparc + ".nii.gz -q /auto/home/users/d/r/drimez/wmql_tracts.qry -o " + track_name_out + ".trk"
        
        log_prefix = "WM query"
        print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": WM query launched for " + "dipy" + mod + "tracts\n")
        f = open(f_path + "subjects/" + patient + "/tracking/logs.txt", "a+")
        f.write("[" + log_prefix + "] " + datetime.datetime.now().strftime(
            "%d.%b %Y %H:%M:%S") + ": WM query launched for " + "dipy" + mod + "tracts\n" + wm_query_cmd)
        f.close() 
        
        bashcmd = wm_query_cmd.split()
        process = subprocess.Popen(wm_query_cmd, universal_newlines=True, shell=True, stdout=sys.stdout, stderr=sys.stderr)
        # wait until finish
        out, error = process.communicate()
    """
if __name__ == "__main__": 

    args = None
    if len(sys.argv)>1: 
        args = [str(arg) for arg in sys.argv[1:]] 
        query(*args) 
    else:
        for model_type in ["_prob",""]:
            query(model_type)
    
    
    
    
    
    
    
    
    
    
    
    