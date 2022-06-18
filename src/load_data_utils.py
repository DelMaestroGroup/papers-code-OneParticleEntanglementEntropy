"""Functions for loading data from files."""

from multiprocessing.sharedctypes import Value
from pathlib import Path
from typing import Iterable, List, Dict, Union, Any
import numpy as np
import codecs
import LL_utils


def load_entanglement_from_file(dat_path: Path):
    """
    Reads entanglement and header form data file.
    Returns interaction strength V array and entropies Sα where Sα[10] is Renyi with Sα=1/2.
    """
    with open(dat_path, "r", encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
    nLines = len(lines)

    params = dict()
    V = []
    Sα = []

    header = lines.pop(0)
    header = header.replace("#", "")
    # read header to params
    header_list = header.split(",")
    for paramText in header_list:
        if "=" in paramText:
            name, value = paramText.split('=')
            # remove and whitespace
            name = ''.join(name.split()).replace("\n", "")
            value = ''.join(value.split()).replace("\n", "")
            # add to dict
            params[name] = int(float(value)) if int(
                float(value)) == float(value) else float(value)
        else:
            paramText = ''.join(paramText.split()).replace("\n", "")
            params[paramText] = True

    # read data to V and Sα
    data = np.loadtxt(dat_path, encoding="utf-8", max_rows=nLines-4)
    V = data[:, 0]
    Sα = data[:, 1:]

    # for line in lines[:-1]:
    #     if line[0] != "#":
    #         dat = line.split()
    #         if len(dat) == 12:
    #             V.append( float(dat.pop(0)))
    #             Sα.append( [float(d) for d in dat] )

    return params, np.array(V), np.array(Sα)


def load_entanglement_from_files(dat_path_template: str, replace_list: List[Any]):
    """
    Reads entanglement and header form data for multiple calculations from files.
    Returns interaction strength V[iv, ifile] array and entropies Sα[iv,iα-1,ifile] where Sα[iv, 11-1, ifile] is Renyi with Sα=1/2.
    Here iv is index for the interaction strength, iα the power of the Renyi entropy, and ifile the corresponding index for the values in replace_list.

    Arguments: 
            dat_path_template [str] :   String template such that 
                                        dat_path_template.format(replace_list[ifile]) 
                                        or dat_path_template.format(**replace_list[ifile])
                                        is the filename to load from
            replace_list [List[Any]]:   List of replacements for the string template, can 
                                        be a list of values or a list of dicts to fill 
                                        multiple parameters in the template at once.

    Returns:
            params [List[Dict]]     :   params[ifile] is the parameters in the header of file ifile
            V [np.array]            :   V[iv, ifile] is the interaction strength V[iv] from file ifile
            Sα [np.array]           :   Sα[iv, iα-1, ifile] is the Renyi entropy with power iα (iα=1-1 is 
                                        van Neumann, iα=11-1 is α=1/2) for interaction V[iv] from file ifile

    Example:
            params, V, Sα = load_entanglement_from_files("./out/name_n{:02d}", [list(1:4),10] )
                will load data from files ["./out/name_n01","./out/name_n02","./out/name_n03","./out/name_n10"]
    """

    # initialize returns
    params = []
    V = []
    Sα = []

    # load from files
    for replacer in replace_list:
        if isinstance(replacer, dict):
            dat_path = Path(dat_path_template.format(**replacer))
        else:
            dat_path = Path(dat_path_template.format(replacer))
        params_ifile, V_ifile, Sα_ifile = load_entanglement_from_file(dat_path)

        params.append(params_ifile)
        iVsort = np.argsort(V_ifile)
        V.append(V_ifile[iVsort])
        Sα.append(Sα_ifile[iVsort, :])

    return params, np.stack(V, axis=-1), np.stack(Sα, axis=-1)


class QuenchParticleEE:
    data: Dict
    t: np.ndarray
    S: np.ndarray
    Serr: np.ndarray
    V: np.ndarray

    def __init__(self, S_of_t_vec, V_vec, Serr=None, Sinf=None):

        t_vec = S_of_t_vec[0][:, 0]

        nS = np.shape(S_of_t_vec[0])[1] - 1
        nt = np.size(t_vec)
        nV = np.size(V_vec)

        S_mat = np.zeros((nS, nt, nV))
        for (iV, V), (iS, S) in zip(enumerate(V_vec), enumerate(S_of_t_vec)):
            S_mat[:, :, iV] = np.transpose(S[:, 1:])

        self.data = {
            "t": np.array(t_vec),
            "V": np.array(V_vec),
            "S": np.array(S_mat),
            "Si": lambda ia: np.array(S_mat)[ia-1, :, :] if ia != 0.5 else np.array(S_mat)[10, :, :],
            "St": lambda it: np.array(S_mat)[:, it, :],
            "SV": lambda iV: np.array(S_mat)[:, :, iV],
        }
        self.S = np.array(S_mat)
        self.Serr = Serr  # placeholder
        self.t = np.array(t_vec)
        self.Δt = self.t[1]-self.t[0]
        self.V = np.array(V_vec)

        self.tres = lambda V: LL_utils.v_a_timeres(V)*self.t

    def __repr__(self):
        return "Quench data handler with fields: S, t, V. Can be called with arguments ('Sind',i=slice/none,V=slice/none,t=slice/none) or with ('S',i=RenyiIndex/none,V=Vvalue/none,t=tvalue/none) to access elements."

    def __call__(self, idstr: str = "", i=None, V=None, t=None, get_closest=False):
        def slicer_val(x, arr):
            if x is None:
                return np.s_[:]
            if isinstance(x, slice) or isinstance(x, list) or isinstance(x, np.ndarray):
                return x
            if (not get_closest) and (not ((x in arr) or (1.e-14 > np.min(np.abs(x-arr))))):
                raise ValueError(
                    f"There is no value {x} in the array. To get the closest value, use options get_closest=True.")
            ind = np.argmin(np.abs(x-arr))
            return np.s_[ind]

        def slicer(x): return np.s_[x] if x is not None else np.s_[:]

        if idstr == "t":
            return self.t
        elif idstr == "V":
            return self.V
        elif idstr in ["data", "all", "full"]:
            return self.data
        elif idstr == "S":
            i_slice = np.s_[i-1 if i !=
                            0.5 else 10] if not isinstance(i, slice) else i
            V_slice = slicer_val(V, self.V)
            t_slice = slicer_val(t, self.t)
            return self.S[i_slice, t_slice, V_slice]
        elif idstr == "Sind" or idstr == "":
            i_slice = slicer(i)
            V_slice = slicer(V)
            t_slice = slicer(t)
            return self.S[i_slice, t_slice, V_slice]
        elif idstr == "Serrind" or idstr == "":
            if self.Serr is None:
                raise IndexError("Serrind has to be computed first.")
            i_slice = slicer(i)
            V_slice = slicer(V)
            return self.Serr[i_slice, V_slice]
        elif idstr == "Serr" or idstr == "":
            if self.Serr is None:
                raise IndexError("Serr has to be computed first.")
            i_slice = np.s_[i-1 if i !=
                            0.5 else 10] if not isinstance(i, slice) else i
            V_slice = slicer_val(V, self.V)
            return self.Serr[i_slice, V_slice]

        raise IndexError(f"There is no field named {idstr}.")

# ED quench results
def load_quench_data(N, V_array, folder="./", num_t=None, tstart=0.0, tend=100.0, dt=0.1, V0=0.0, Vp0=0.0, Vp=0.0, t=1.0, prefix="particle_entanglement_n", n=1):
    if num_t is None:
        num_t = np.size(np.arange(tstart, tend+dt/2, dt))
    S_of_t_vec = []
    for V in V_array:
        fn = quench_datfile_name(
            N, V, tstart, tend, dt, V0, Vp0, Vp, t, prefix, n)
        path = Path(folder).joinpath(fn)
        filecp = codecs.open(path, encoding='utf-8')
        S_of_t_vec.append(np.loadtxt(filecp, max_rows=num_t+2))

    return QuenchParticleEE(S_of_t_vec, V_array)


def quench_datfile_name(N, V, tstart=0.0, tend=100.0, dt=0.1, V0=0.0, Vp0=0.0, Vp=0.0, t=1.0, prefix="particle_entanglement_n", n=1):
    return f"{prefix}{n:02d}_M{2*N:02d}_N{N:02d}_t{t:+5.3f}_Vp{Vp:+5.3f}_Vp0{Vp0:+5.3f}_V{V:+5.3f}_V0{V0:+5.3f}_dt{dt:5.4f}_tstart{tstart:06.3f}_tendf_{tend:5.3f}.dat"

# DMRG quench results
def itensor_quench_datfile_name(N, Vp=0.0, tsta=0.0, tend=100.0, tstep=0.1, V0=0.0, Vp0=0.0, Vsta=-2.0, Vend=-2.0, Vnum=41, t=1.0, prefix="particle_entanglement_n", n=1, tdvp="", trotter1="", bc="", n2=""):
    return f"{prefix}{n:02d}_M{2*N:02d}_N{N:02d}_t{t:+5.3f}_Vp{Vp:+5.3f}_tsta{tsta:+5.3f}_tend{tend:+5.3f}_tstep{tstep:+5.3f}_Vsta{Vsta:+5.3f}_Vend{Vend:+5.3f}_Vnum{Vnum:04d}{n2}{tdvp}{trotter1}{bc}.dat"


def load_itensor_quench_data(N, folder="../data/paricleEE_quench_itensor", Vp=0.0, tsta=0.0, tend=100.0, tstep=0.1, V0=0.0, Vp0=0.0, Vsta=-2.0, Vend=2.0, Vnum=41, t=1.0, prefix="particle_entanglement_n", n=1, tdvp=False, trotter1=False, obc=False, n2=False):

    fn = itensor_quench_datfile_name(N, Vp, tsta, tend, tstep, V0, Vp0, Vsta, Vend, Vnum, t, prefix, n, n2="_n02" if n2 else "",
                                     tdvp="_tdvp" if tdvp else "", trotter1="_trotter1" if trotter1 else "", bc="_obc" if obc else "")
    path = Path(folder).joinpath(fn)
    with codecs.open(path, encoding='utf-8') as filecp:
        lines = filecp.readlines()
        nLines = len(lines)
        V_array = [float(x.strip()) for x in lines[3].split()[2:]]
        nTimes, nV = [int(x.strip()) for x in lines[5].split()[2:]]
    with codecs.open(path, encoding='utf-8') as filecp:
        data = np.loadtxt(filecp, max_rows=nLines-3-8, ndmin=2, skiprows=8)
    data = data.reshape((-1, nV, 12), order="F")
    S_of_t_vec = [data[:, iV, :] for iV in range(nV)]

    return QuenchParticleEE(S_of_t_vec, V_array)
 
 