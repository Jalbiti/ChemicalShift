################################################################
## 17/05/22
## Distance, outliner in training set, test training
## Available add QM data 
## 100 filling, ring current
################################################################

import numpy as np
import pandas as pd
import csv
import _pickle as pickle
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
import itertools 
from statistics import mean
from statistics import stdev
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)
import os
import wget
from Bio import PDB
import pynmrstar
from ast import Continue
import math
from scipy.special import ellipk, ellipe #first and second kind

#Not used packages
#from sklearn.linear_model import LinearRegression
#from sklearn.neural_network import MLPRegressor
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import RidgeCV, LinearRegression
#from sklearn.compose import TransformedTargetRegressor
#from sklearn.preprocessing import normalize, StandardScaler, QuantileTransformer, quantile_transform, MinMaxScaler
#from sklearn.metrics import ConfusionMatrixDisplay, matthews_corrcoef, classification_report
#from sklearn.decomposition import PCA
#from sklearn import cluster
#from sklearn.cluster import KMeans
#from sklearn.svm import SVC
#from sklearn.multioutput import MultiOutputClassifier
#from os.path import exists
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Parser to read the pdbstructure
parser = PDB.PDBParser()

np.random.seed(10)

#Test for the training set. Tupple with PDB name and BMRB code. PDB and BMRB are stored at /orozco/projects/NMR_i-motif/CSML/model_QM/models/Iexp_a1r1N120_ownatoms_unpdb_totalclean/NMRStar
#Just the chemical shifts with C well calibrated.
combine = [('1KKA', 5256), ('1L1W', 5321), ('1LC6',5371), ('1LDZ', 4226), ('1NC0', 5655), ('1OW9', 5852), ('1PJY', 5834), ('1R7W', 6076), 
('1R7Z',6077), ('1YSV',6077), ('2FDT', 10018), ('2GM0', 7098), ('2JXQ', 15571), ('2JXS', 15572), ('2K3Z', 15780), ('2K41',15781), ('2KOC', 5705), 
('2KYD',16980), ('2LBL', 17565), ('2LBJ', 17565), ('2LDL', 17671), ('2LDT', 17682), ('2LHP', 17682), ('2LI4', 17877), ('2LK3', 17972), 
('2LP9', 18239), ('2LPA',18240), ('2LU0', 18503), ('2LUB', 18515), ('2LV0', 18549), ('2RN1', 11014), ('2Y95', 16714), ('4A4S', 18036), 
('4A4T', 18034), ('4A4U', 18035)] 

#Test for training without 2KYD. For some unkowned reason this structure fails in the calculation of the ring current.
combine = [('1KKA', 5256), ('1L1W', 5321), ('1LC6',5371), ('1LDZ', 4226), ('1NC0', 5655), ('1OW9', 5852), ('1PJY', 5834), ('1R7W', 6076), 
('1R7Z',6077), ('1YSV',6077), ('2FDT', 10018), ('2GM0', 7098), ('2JXQ', 15571), ('2JXS', 15572), ('2K3Z', 15780), ('2K41',15781), ('2KOC', 5705), 
('2LBL', 17565), ('2LBJ', 17565), ('2LDL', 17671), ('2LDT', 17682), ('2LHP', 17682), ('2LI4', 17877), ('2LK3', 17972), 
('2LP9', 18239), ('2LPA',18240), ('2LU0', 18503), ('2LUB', 18515), ('2LV0', 18549), ('2RN1', 11014), ('2Y95', 16714), ('4A4S', 18036), 
('4A4T', 18034), ('4A4U', 18035)] 

combine = [('1KKA', 5256), ('1L1W', 5321)]

#Test for the training set.
combine_test = [('1XHP', 6320), ('1Z2J', 6543), ('1ZC5', 6633), ('2JWV', 15538),
           ('2K65', 15858), ('2K66', 15859),
           ('2LI4', 17877), ('2LQZ', 18336), ('2LUN',18532),
           ('2LX1', 18656), ('2M12', 18838), ('2M21', 18891), ('2M22', 18892),
           ('2M23', 18893), ('2M24', 18894), ('2M8K', 19260), ('2MEQ', 18975),
           ('2MHI', 19634), ('2MI0', 19662), ('2MIS', 19692), 
           ('2QH2', 7403), ('2QH3', 7404), ('2QH4', 7405)] #('2LPS', 5962)  #2M23 supersedes 2K64/ ('2K64', 15857),  2M24 supersedes ('2K63', 15856),

#Code for atoms to be added to features

atomlist = {"C1'": 15,'C2': 10,"C2'": 16,"C3'": 17,'C4': 11,"C4'": 18,'C5': 12,
            "C5'": 27,'C6': 13,'C8': 14,'H1': 0,"H1'": 4,'H2': 8,"H2'": 5,'H3': 9,
            "H3'": 6,"H4'": 7,'H5': 1,'H6': 2,'H8': 3,'N1': 24,'N2': 29,'N3': 25,
            'N4': 34,'N6': 36,'N7': 30,'N9': 28,'O2': 26,"O2'": 22,"O3'": 20,
            'O4': 37,"O4'": 23,"O5'": 21,'O6': 35,'OP1': 33,'OP2': 31,'OP3': 32,'P': 19, 
            "HO2'": 38, "H5''": 39, "HO5'": 40, "HO3'": 41, "H5'": 42}

#Atoms to obtain a model
atoms_all = ['H1\'', 'H2\'', 'H3\'', 'H4\'', 'H2', 'H5', 'H6', 'H8', # removed 'H5\''
             'C1\'', 'C2\'', 'C3\'', 'C4\'', 'C5\'', 'C2', 'C5', 'C6', 'C8']

#List of atoms to define the aromatic rings. Purines are considered as 2 aromatics rings. Used in get_com_resdata.
Pur5 = ['N9', 'C8', 'N7', 'C5', 'C4'] # Ring of 5 atoms of purines (A, G)
Pur6 = ['C5', 'C4', 'N3', 'C2', 'N1', 'C6']  # Ring of 6 atoms of purines (A, G)
Pyr6 = ['N1', 'C2', 'N3', 'C4', 'C5', 'C6']  # Ring of 6 atoms of pyrimidine (C, U)

#List of atoms that belongs to the aromatic ring. This is used to avoid the contribution of these atoms to the own aromatic ring.
skip_atoms = ['H2', 'H5', 'H6', 'H8', 'C2', 'C5', 'C6', 'C8']

# List of structures and chemical shift values computed by QM to be used in the training.
# Tuple with pdb file and csv file.  
combineQM = []

########################################################
#Functions to the geometry calculation of aromatic rings.
########################################################

# Get rings that are close to the target atom. Look for atoms that are close and obtain the ring names.
# Take dictionary with the closest atoms to the target_atom as a argument. 
def get_close_rings(near_dic):
    close_rings = set()  # Use a set for faster membership checks
    set_value = 10  # Set value to consider as a close atom
    for atom_dic, value in near_dic.items():
        if value > math.exp(-2 * (set_value) / 5):
            close_rings.add(atom_dic.split('-')[0])  # Use add() to add rings to the set
    return list(close_rings)  # Convert the set back to a list if needed #Example: ['C17', 'G1', 'G2', 'G3', 'C16']

#Calcule the geometric center of mass of an aromatic ring
# Take list of atoms of the ring as a argument
def get_com(atom_list):
    for a in atom_list:
         coords = np.asarray([a.coord for a in atom_list], dtype=np.float32) #Is necessary set dtype??
    return np.average(coords, axis=0) #Example [13.617     -4.1080003  9.6058] ; array of coordinates

# Calculate and store the center of mass of all aromtic rings present in the structure. Split purines in the two aromatic rings.
# Use residues from Biopython as a argument
def get_com_resdata(residues):
    com_resdata = dict() # To store the final data
    for res in residues: # Analyze each residue
        temp_pur5 = list() # Create three temporal lists to store the atoms of the aromatic rings
        temp_pur6 = list() # Each resiude analyze is cleaned
        temp_pyr6 = list()
        if res.resname == "G" or res.resname == "A": # For purines
            for atom in res: #Store the atoms of the rings to the temporal list
                if atom.name in Pur5:
                    temp_pur5.append(atom)
                if atom.name in Pur6:
                    temp_pur6.append(atom)
            com_pur5 = get_com(temp_pur5) #Get center of mass
            com_pur6 = get_com(temp_pur6)
# Save the center of mass and the atoms of the aromitc ring in the dicctionry. Atoms are used later to calculate plane.
# Format: key: 'G2-pur5'; value: tuple of com and list of atoms in the ring
        com_resdata[f'{res.resname}{res.full_id[3][1]}-pur5'] = com_pur5, temp_pur5 
        com_resdata[f'{res.resname}{res.full_id[3][1]}-pur6'] = com_pur6, temp_pur6 
        if res.resname == "C" or res.resname == "U": # For pyrimidines
            for atom in res: 
                if atom.name in Pyr6:
                    temp_pyr6.append(atom)
            com_pyr6 = get_com(temp_pyr6)
            com_resdata[f'{res.resname}{res.full_id[3][1]}-pyr6'] = com_pyr6,temp_pyr6
# Example get_com_resdata = (array([13.617    , -4.1080003,  9.6058   ], dtype=float32), [<Atom N9>, <Atom C8>, <Atom N7>, <Atom C5>, <Atom C4>])
    return com_resdata 

# Calculate and store the center of mass of all aromtic rings present in the structure. Split purines in the two aromatic rings.
# Use residues from Biopython as a argument
# Save the center of mass and the atoms of the aromitc ring in the dicctionry. Atoms are used later to calculate plane.
# Format: key: 'G2-pur5'; value: tuple of com and list of atoms in the ring
def get_com_resdata_optimized(residues):
    com_resdata = dict() # To store the final data
    for res in residues:
        res_key = f'{res.resname}{res.full_id[3][1]}'
        if res.resname in ["G", "A"]:
            temp_pur5 = [atom for atom in res if atom.name in Pur5]
            temp_pur6 = [atom for atom in res if atom.name in Pur6]
            com_resdata[f'{res_key}-pur5'] = get_com(temp_pur5), temp_pur5
            com_resdata[f'{res_key}-pur6'] = get_com(temp_pur6), temp_pur6
        elif res.resname in ["C", "U"]:
            temp_pyr6 = [atom for atom in res if atom.name in Pyr6]
            com_resdata[f'{res_key}-pyr6'] = get_com(temp_pyr6), temp_pyr6
# Example get_com_resdata = (array([13.617    , -4.1080003,  9.6058   ], dtype=float32), [<Atom N9>, <Atom C8>, <Atom N7>, <Atom C5>, <Atom C4>])
    return com_resdata

# Extract for the com_resdata only the values of the close ring of the target atom.
# Take close_rings (list of close aromatic rings) and com_resdata (com of all aromtic rings)
def get_com_close_rings(close_rings, com_resdata):
    com_resdata_targetatom = dict() # To store the final data
    for res in close_rings: 
        if "G" in res or "A" in res: #Res 
            com_resdata_targetatom[f'{res}-pur5'] = com_resdata[f'{res}-pur5']  # Same format that in com_resdata
            com_resdata_targetatom[f'{res}-pur6'] = com_resdata[f'{res}-pur6']
        elif "C" in res or "U" in res:
            com_resdata_targetatom[f'{res}-pyr6'] = com_resdata[f'{res}-pyr6']
    return com_resdata_targetatom

# Get the plane equation of a aromatic ring
# Three atoms and the center of mass as arguments
def get_plane(atom1,atom2,atom3, com):
    p1 = np.array(atom1.coord) #Defines coordenates of each atom
    p2 = np.array(atom2.coord)
    p3 = np.array(atom3.coord)
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp
    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, com)
    # print(f'The equation is {a/c}x + {b/c}y + {c/c}z = {d/c}')
    # print(a,b,c,d)
    return a,b,c,d

#Get the z distance from a target arom to the plane. Formula: distance from a point to plane
# Arguments are the plane equation and the target arom
def get_z_coord(a,b,c,d, target_atom):
    z_value = abs(a*target_atom.coord[0]+b*target_atom.coord[1]+c*target_atom.coord[2]-d)/math.sqrt(a**2+b**2+c**2)
    #and the distance from the point to the plane is#
    # print(z_value)
    return z_value

# Get the p distance. Distance from the center of mass to the projection of the target atom in the plane.
# Argumetns are the plane equation, the target atom and the center of mass of the plane.
def get_p_coord(a,b,c,d, target_atom, com):
    z = get_z_coord(a,b,c,d, target_atom)
    # punto_p = target_atom.coord + z*[a,b,c]/math.sqrt(a**2+b**2+c**2)
    # To check if d is negative. Punto_p is the projection of target atom in the plane
    direction = 1 if d > 0 else -1
    punto_p = target_atom.coord + direction * z * np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
    #Calculate the distance between the center of mass and the punto_p
    p = np.linalg.norm(punto_p - com)
    return p, z # Return a tuple of the p and z values

# Store the p and z values from the target atom to the closet aromatic rings.
# Arguments are the target atom and the diccionary of the closest rings with 
# his center of mass and the atoms list that are in the ring.
def get_distance_to_com(target_atom, com_resdata_targetatom):
    cylindrical_coordinates = {} # To store the final data
    for key, value in com_resdata_targetatom.items(): # Iterate with each aromatic ring 
        res_of_ring, type_of_ring = key.split("-")
        com = value[0]  #Center of mass
        atom1, atom2, atom3 = value[1][:3] #T3 atoms of the aromatic ring
        a, b, c, d = get_plane(atom1, atom2, atom3, com) # Get plane equation
        p, z = get_p_coord(a, b, c, d, target_atom, com) # Get distance in cylindrical coordinates
        cylindrical_coordinates[key] = (p, z)  # key example: "G1-H1'"
    return cylindrical_coordinates #Example key: "G1-H1'", value: 'C3-pyr6': (17.421, 8.145)

# Calculate the geometryc term of the aromatic current effect
# Argumetns are calculated in the rc_shield funcion.
def  G(k_neg, k_pos, rho, z_neg, z_pos):
    neg_term = (ellipk(k_neg**2) + ((1 - (rho**2) - (z_neg**2)) / ((1 - rho)**2  + (z_neg**2))) * ellipe(k_neg**2)) / np.sqrt((1 + rho)**2  + (z_neg**2))
    pos_term = (ellipk(k_pos**2) + ((1 - (rho**2) - (z_pos**2)) / ((1 - rho)**2  + (z_pos**2))) * ellipe(k_pos**2)) / np.sqrt((1 + rho)**2  + (z_pos**2)) 
    G_r = neg_term + pos_term
    return(G_r)

# Calculate the contribution of the aromatic current effect.
# In order for us to calculate the RC shielding we use the JohnsonBovey model
# Argument is the dictionary with the aromatic ring and the distance from target atom.
def rc_shield(rings_dict):
  #i (1st value) denotes the intensity factors given by the following table
    #Gua-5, Gua-6, Ade-5, Ade-6, Cyt, Ura
  #a denotes ring radio
    descr = {'G5':(0.655, 1.154), 'G6': (0.3, 1.361), 'A5': (0.66, 1.154),
         'A6': (0.9, 1.343), 'C6': (0.275, 1.3675), 'U6': (0.11, 1.379)}
    m = 9.109 * (10**(-31))
    c = 299792458 #speed of light?
    e = -1.6*(10**(-19))
    sigma=0
    sum_G_r = 0
    z_bar = 0.64 # 5g_hat is the theoretical average distance for 25]5g
    for key, val in rings_dict.items():
        # print(key)
        residue_type = str(key).split("-")[0][0] # Check the residue name G,C,U,A
        ring_type = str(key).split("-")[1][-1] # Check the ring type: pur5, pur6, pyr6
        intensity_factor, ring_radius = descr[residue_type+ring_type] # Select the radio and intensity factor
        # rho, z = val[0], val[1] 
        rho, z = val[0]/ring_radius, val[1]/ring_radius #in terms of ring_radius     
        z_neg,  z_pos = z - z_bar, z + z_bar
        k_neg, k_pos = np.sqrt((4*rho)/((1+rho)**2+(z_neg**1))), np.sqrt((4*rho)/((1+rho)**2+(z_pos**1))) #Module for ellipk and ellipe
        G_r = G(k_neg, k_pos, rho, z_neg, z_pos)
        sum_G_r = sum_G_r + G_r # Sumatory of geometry term
        # deltaN = 2.130*(intensity_factor/ring_radius)*G_r*2
    return sum_G_r

# Create diccionaty with closest atoms to target
# Structure from biopython
def near(structure, ns, atomname,resnumber, chainid, atomlist, target_limit):
    target_atom = structure[0][chainid][resnumber][atomname] # Select the target atom. [0] for selecting the first strucutre of ensemble.
    close_atoms = ns.search(target_atom.coord, target_limit) # Target_limit is the distance cutoff. List of atoms close to the target atom
    resname = structure[0][chainid][target_atom.full_id[3][1]].resname # Example "G"
    atom_key = f'{resname}{resnumber}-{atomname}'
    diccionario = {atom_key: {}} # Save the close_atoms for each target atom with key "G1-H1'"
    for atom in close_atoms:
      resid = atom.full_id[3][1] # Define number of residue of the atom in the close_atoms
      # if resid != resnumber: #to avoid add to the diccionary atoms of the same residue, if necessary index next 3 lines
      resname2 = structure[0][atom.full_id[2]][atom.full_id[3][1]].resname
      # distance =  math.exp(-2*(round(atom - target_atom, 3))/5) ## distancia a la inversa, important
      distance =  math.exp(-2*(atom - target_atom)/5) ## distancia a la inversa, important. Sin round
      # distance =  round(atom - target_atom, 3) ## distancia directa
      diccionario[atom_key][f'{resname2}{resid}-{atom.id}'] = distance # 
    diccionario[atom_key].pop(atom_key, None)
    return diccionario # Example key: ["G1-H1'"]["C3-OP1"], value: 0.015488245866928456

# Read csv file with QM data. First line not read, example rest of the lines: U1-H5,5.041
# Check the atomlist as a function input
def getcsML(csvfile, atomlist):
    csML = dict()
    with open(csvfile, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first line
        for row in reader:
            atom = row[0].split('-')
            if atom[1] in atomlist:
                csML[row[0]] = float(row[1])
    return csML

#Get NMR experimental data from file save in local computer. First check if the BMRB file is downloaded, if not download it to local
def getcs(bmrbid, atomlist):
    file_path = f'/orozco/projects/NMR_i-motif/CSML/model_QM/models/Iexp_a1r1N120_ownatoms_unpdb_totalclean/NMRStar/{bmrbid}'
    if not os.path.exists(file_path): #check file exitst
        entrydownload = pynmrstar.Entry.from_database(bmrbid, convert_data_types=True) #convert_data_types to import number as floats
        entrydownload.write_to_file(file_path)
    #convert_data_types to import number as floats    
    entry = pynmrstar.Entry.from_file(file_path, convert_data_types=True) 
    #entry.write_to_file(f'{bmrbid}')
    cs_result_sets = [] # To store all chemical shift present in the BMRB file
    cs = {} # To store the final data
    for chemical_shift_loop in entry.get_loops_by_category("Atom_chem_shift"):
        for record in chemical_shift_loop.get_tag(['Comp_index_ID', 'Comp_ID', 'Atom_ID', 'Atom_type', 'Val', 'Val_err']):
            if record[2] in atomlist: #select only the atom of the selected model of atoms_all
                cs[f'{record[1]}{record[0]}-{record[2]}'] = record[4] #cs[U1-H5] = 5.041
    return cs

# Mix CS data with list of the closet atoms. PDB file expected to be in local, if not download from website.
# Combine are the training set, studied_atoms is a list with contain the atom studied in the model
def get_dictionaries(combine, atomlist, studied_atoms):
#To store the final data is a list where elements are dictionaries for each structure present in combine
  cs_all = list()
  diccionario = list()
  all_cylindrical_coordinates = list()
  pdb_base_path = '/orozco/projects/NMR_i-motif/CSML/model_QM/models/PDBs/'
  for i, (pdb_id, bmrb_id) in enumerate(combine):
    try:
      # Get structure and near atoms
      pdb_path = f'{pdb_base_path}{pdb_id}.pdb'
      if not os.path.exists(pdb_path): #Chek if PDB file exits
          url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
          wget.download(url, out=pdb_path)
      structure = parser.get_structure(pdb_id, pdb_path) #parse PDB structure
      atoms = PDB.Selection.unfold_entities(structure[0], 'A') #list with all atom in the pdb, A refers to atoms
      cs = getcs(bmrb_id, studied_atoms) # Get experimetal data of the selected atom and residue type
      data_items = cs.items()
      data_list = list(data_items)
      experimental = pd.DataFrame(data_list) #dataframe with residue-atom and cs value
      new_values = []
      for atom in experimental[0]:
        residue = str(atom).split("-")[0]
        at = str(atom).split("-")[1] #atom
        atom_ref = residue[0] + "-" + at
      # To avoid train model with atoms with very few cs experimental data
        if at == 'C5' and residue[0] in ['G', 'A']:
            print(f'{atom} pass')
            continue
        if at == 'C2' and residue[0] in ['C', 'G', 'U']:
            print(f'{atom} pass')
            continue
        if at == 'C6' and residue[0] in ['G', 'A']:
            print(f'{atom} pass') 
            continue       
        exp_c = float(experimental.iloc[np.where(experimental[[0]] == atom)[0]][1]) 
        new_values.append(exp_c)
      # Add new values to new experimental data as we will train with the new Ys
      experimental['labels'] = new_values
      ## Keep only those atoms that have a corresponding value in the reference
      # final_experimental = experimental[experimental[1].astype(float) != experimental['labels']]
      final_experimental = experimental[np.array(experimental['labels']) != float(-1)]
      new_cs = final_experimental.drop([1], axis=1) #Final dataframe with residue-atom and cs value
      # new_cs = experimental.drop([1], axis=1)
      cs = dict(zip(new_cs[0], new_cs['labels'])) #transfer dataframe to dictionary[residue-atom]=cs value
      # cs_all.append(cs)
      heavyatoms = [atom for atom in atoms if atom.element != 'H']
      ns = PDB.NeighborSearch(heavyatoms) # Fast atom neighbor lookup using a KD tree (implemented in C). Map the distance of the atoms with the heavyatoms   
      new_dict = dict()
      for atom in atoms:
      # for atom in heavyatoms:
          #For each atom of the model create a diccionaty with all the closest atoms and the value of the distance.
          if atom.name in studied_atoms: #Select only the atom of the selected model of atoms_all
            new_dict.update(near(structure = structure,
                                 ns = ns,
                                 atomname = atom.name,
                                 resnumber = atom.full_id[3][1],
                                 chainid = atom.full_id[2],
                                 atomlist = atomlist,
                                 target_limit = 10)) # Here is the cut off to keep the atoms close.
      diccionario.append(new_dict) #dictionary with of the atoms of the model, is in a list of len == 1
      cs_all.append(cs)
#########Search close aromatic rings and get polar coordinates
      residues = PDB.Selection.unfold_entities(structure[0], 'R')  # Store the residues in a list
      all_com_resdata_targetatom = dict() 
      cylindrical_coordinates = dict()
      for atom in atoms:
          atomname = atom.name
          if atomname not in studied_atoms: # To select store only data of the atom model
              continue
          resnumber = atom.full_id[3][1]
          chainid = atom.full_id[2]
          resname = structure[0][chainid][atom.full_id[3][1]].resname
          atoms_near = near(structure, ns, atomname, resnumber, chainid, atomlist, target_limit = 10) #Create list of atoms close
          close_rings = get_close_rings(atoms_near[f"{resname}{resnumber}-{atomname}"]) # Store rings that are close
          if atomname in skip_atoms: # To remove own ring for aromatic atoms
            for residue in close_rings: #Is necessary this loop? Maybe just close_rings.remove(residue)
                if residue[1:] == str(resnumber):
                        close_rings.remove(residue)  
          com_resdata = get_com_resdata(residues) # Get data of all ring center of mass
          com_resdata_targetatom = get_com_close_rings(close_rings, com_resdata) # Keep just data of rings close to target atom
          all_com_resdata_targetatom[f'{resname}{resnumber}-{atomname}'] = com_resdata_targetatom # Save all data
          cylindrical_coordinates[f'{resname}{resnumber}-{atomname}'] = get_distance_to_com(atom, com_resdata_targetatom) # Save distances rho, z
      all_cylindrical_coordinates.append(cylindrical_coordinates) # List to store the data of each combine
    except:
      print(f'get_dictionaries-{combine[i]} pass') #if something happens with one PDB, just skipped
      continue
  #make the same that before but with PDB with QM data. Iterate with combineQM list
  for i, (pdb_id, bmrb_id) in enumerate(combineQM): ###### import QM data ######
    # try:
      # Get structure and near atoms
      pdbstructure = f'/orozco/projects/NMR_i-motif/CSML/model_QM/models/QMdata/{pdb_id}.pdb' #directory of PDB
      print(pdbstructure)
      structure = parser.get_structure(pdb_id, pdbstructure)
      atoms = PDB.Selection.unfold_entities(structure[0], 'A') #list with all atom in the pdb
      # Get CS
      csvfileQM = f'/orozco/projects/NMR_i-motif/CSML/model_QM/models/QMdata/{bmrb_id}.csv' #directory of the QM CS data
      cs = getcsML(csvfileQM, atomlist) #get QM cs data 
      experimental = pd.DataFrame(list(cs.items()))
      new_values = [float(experimental.iloc[np.where(experimental[[0]] == atom)[0]][1]) for atom in experimental[0]]
      #new_values = []
      #for atom in experimental[0]:
      #  residue = str(atom).split("-")[0]
      #  at = str(atom).split("-")[1]
      #  atom_ref = residue[0] + "-" + at
      #  exp_c = float(experimental.iloc[np.where(experimental[[0]] == atom)[0]][1])
      #  new_values.append(exp_c)
      # Add new values to new experimental data as we will train with the new Ys
      experimental['labels'] = new_values
      ## Keep only those atoms that have a corresponding value in the reference
      # final_experimental = experimental[experimental[1].astype(float) != experimental['labels']]
      final_experimental = experimental[np.array(experimental['labels']) != float(-1)]
      new_cs = final_experimental.drop([1], axis=1)
      # new_cs = experimental.drop([1], axis=1)
      cs = dict(zip(new_cs[0], new_cs['labels']))
      cs_all.append(cs)
      heavyatoms = [atom for atom in atoms if atom.element != 'H']
      ns = PDB.NeighborSearch(heavyatoms)
      new_dict = dict()
      for atom in atoms: 
      # for atom in heavyatoms:
          if atom.name in studied_atoms:
            new_dict.update(near(structure = structure,
                                 ns = ns,
                                 atomname = atom.name,
                                 resnumber = atom.full_id[3][1],
                                 chainid = atom.full_id[2],
                                 atomlist = atomlist,
                                 target_limit = 10)) # Limit
      diccionario.append(new_dict)
#########Search close aromatic rings and get polar coordinates
      residues = PDB.Selection.unfold_entities(structure[0], 'R') 
      all_com_resdata_targetatom = dict()
      cylindrical_coordinates = dict()
      for atom in atoms:
          atomname = atom.name
          if atomname not in studied_atoms:
              continue
          resnumber = atom.full_id[3][1]
          chainid = atom.full_id[2]
          resname = structure[0][chainid][atom.full_id[3][1]].resname
          atoms_near = near(structure, ns, atomname, resnumber, chainid, atomlist, target_limit = 10)
          close_rings = get_close_rings(atoms_near[f"{resname}{resnumber}-{atomname}"])
          com_resdata = get_com_resdata(residues)
          com_resdata_targetatom = get_com_close_rings(close_rings, com_resdata)
          all_com_resdata_targetatom[f'{resname}{resnumber}-{atomname}'] = com_resdata_targetatom
          cylindrical_coordinates[f'{resname}{resnumber}-{atomname}'] = get_distance_to_com(atom, com_resdata_targetatom)
      all_cylindrical_coordinates.append(cylindrical_coordinates)
  return diccionario, cs_all, all_cylindrical_coordinates

#Remove outliners with the 3-sigma rule
# Argument is the list with cs of differents strucures
def remove_outliners(cs_all):
  allvalues = list() # To store all values stored in a list. 
  for dictcs in cs_all: # Add all values of the different diccionaries to a list
    allvalues = allvalues+list(dictcs.values())
  meanv = mean(allvalues) # Get mean of all values
  stdevv = stdev(allvalues) # Get standart desviation
  print(f'mean is {round(meanv, 2)} and std is {round(stdevv, 2)}')
  #Define max and min of acceptable values with the 3-sigma rule
  max_val = meanv+3*stdevv 
  min_val = meanv-3*stdevv
  delcount = 0 #To count the number of outliners
  #Iterate over all dictionaries in cs_all(one for each BMRB file)
  #Remove values
  for dictcs in cs_all:
      keys_to_remove = [k for k in dictcs if not min_val <= dictcs[k] <= max_val]
      delcount += len(keys_to_remove)
      for k in keys_to_remove:
          del dictcs[k]
  #for dictcs in cs_all: #Iterate over all dictionaries in cs_all(one for each BMRB file)
  #    for k in list(dictcs): #Iterate over all values for each dictionary
  #        if dictcs[k] > max or dictcs[k] < min: #If value is out of limits is eliminated 
  #            # print(f'{k}: {dictcs[k]} ppm')
  #            del dictcs[k]
  #            delcount += 1
  print(f'{delcount} cs removed')
  return cs_all # Return a list with same format but cleaned

#Generate feature to train the model. Process data to the predictor().
def get_features(cs_all, diccionario, all_cylindrical_coordinates):
  features =[] #list with closest atom for each atom in the model
  labels =[] #list of all CS values of the atom 
  keys_order =[] #list of all residue-atom of the atoms in the model
  rc = [] # list with the ring contribution
  print(f'cs len is: {len(cs_all)}')
  print(f'diccionario len is: {len(diccionario)}')
  # cs, diccionario and rc follow the same order
  # diccionario and rc have same keys
  for l in range(len(cs_all)):
    for k in cs_all[l].keys():
      if k in diccionario[l].keys():
        keys_order.append(k)
        features.append(diccionario[l][str(k)])
        labels.append(float(cs_all[l][str(k)]))
        sum_gr = rc_shield(all_cylindrical_coordinates[l][str(k)])
        # to avoid if calculation rc fails
        # Not necessary now
        # if np.isnan(sum_gr):
        #    sum_gr = 0
        # print(f'sum_gr is: {sum_gr}')
        rc.append(sum_gr)
  return features, labels, keys_order, rc

#Encode atom name to number using atomlist. atom_dict2 is atomlist 
def encode_atom(atom, atom_dict2):
  # define mapping of chars to integers and viceversa (previously done)
  char_to_int = atom_dict2
  # integer encode input data
  integer_encoded = char_to_int[atom]
  return integer_encoded

#Encode nucleotide to number. Two options, one integer or binary code.
def onehot_encoding(char):
    # define encoding input values
    nucleotides = 'ACGU'
    # define mapping of chars to integers and viceversa
    char_to_int = dict((c, i) for i, c in enumerate(nucleotides))
    # integer encode input data. Encode nucleotide in one integer
    integer_encoded = char_to_int[char]
    letter = [integer_encoded]    
    # one hot encode
    # onehot_encoded = []
    ## If necessary to encode in binary code, just uncomment the following two lines and coment the previous line. Example: A = [1, 0, 0, 0]
    # letter = [0 for _ in range(len(nucleotides))]
    # letter[integer_encoded] = 1
    return(letter)

#Just in case that len of res is langer of 1 as input to onehot_encoding()
def encode_residue(res):
  encoded = onehot_encoding(res[0])
  # a = [int(res[1:])]
  # encoded.extend(a)
  return encoded

#Return in diferent lists atom and residue type of the closet atoms and the distance bewteen atom target and closest atoms.
#Encoded info, featurized the raw data
def top_atoms(dictlist,atom_dict2, N):
  length = len(dictlist)
  if (length == 0): 
    encoded_atoms = [""] * N
    encoded_residue = [""] * N
    values = [0] * N
    length = [0] * N
  else:
    sorted_values = sorted(dictlist.values(), reverse=True) # Sort the values
    sorted_dictlist = {}
    for i in sorted_values:
        for k in dictlist.keys():
            if dictlist[k] == i:
                sorted_dictlist[k] = dictlist[k]
                break
    # N = 3
    atoms = list(dict(itertools.islice(sorted_dictlist.items(), N)).keys())
    encoded_atoms = []
    encoded_residue = []
    for i in range(len(atoms)):
      residue = str(atoms[i]).split("-")[0]
      encoded_residue.extend(encode_residue(residue))
      atom = str(atoms[i]).split("-")[1]
      encoded_atoms.append(encode_atom(atom,atom_dict2))
    values = (list(dict(itertools.islice(sorted_dictlist.items(), N)).values()))
    if (len(atoms) < N):
      # Get missing near atoms
      diff = N - len(atoms)
      missing_atoms = [100] * diff
      missing_residue = [100] * diff
      missing_values = [100] * diff
      missing_length = [0] * diff
      # join existing + missing
      encoded_atoms = np.concatenate((encoded_atoms, missing_atoms))
      encoded_residue = np.concatenate((encoded_residue, missing_residue))
      values = np.concatenate((values, missing_values))
      # length = np.concatenate((length, missing_length), axis = 0)
  return encoded_atoms, encoded_residue, values, length

#For now will create a separate function to encode the corresponding atom and its residue
def main_entry(entry, atom_dict2):
  residue = str(entry).split("-")[0]
  atom = str(entry).split("-")[1]
  res = encode_residue(residue)
  res.append(encode_atom(atom, atom_dict2))
  return(res)

  # ## We will now try to have the reference as a feature rather than the type of atom
  # atom_ref = residue[0] + "-" + atom
  # try :
  #   ref_c = float(ref.iloc[np.where(ref[[0]] == atom_ref)[0]][1])
  #   new_values.append(ref_c)
  # except:
  #   new_values.append(0)

#Get the X and y values to train the values. Split between train and test
#Get features from top_atoms()
def predictor(features, labels, keys_order, rc, atom_dict, N):
  X = []
  for i in range(len(features)):
    #Dictionary of raw data of the diferent pdb with the closest atoms to the target atom
    dictlist = features[i]
    # print([keys_order[i]])
    # print(dictlist)
    # Encode all data of the closest atoms
    encoded_atoms, encoded_residue, values, length = top_atoms(dictlist, atom_dict, N)
    main_ = main_entry(keys_order[i], atom_dict)
    residue = str(keys_order[i]).split("-")[0]
    key_enconded_residue = encode_residue(residue[0]) ## encoded residue of the key atom
    at = str(keys_order[i]).split("-")[1]
    atom_ref = residue[0] + "-" + at
    rc_value = []
    rc_value.append(rc[i])
    # if (i == 0):
    #   print(encoded_atoms)
    #   print(encoded_residue)
    #   print(values)
    #   print(length)
    #Residue type of the target atom 
    feats = key_enconded_residue ##Before this was the fist entry +1 
    # feats = encoded_atoms ##Before this was the fist entry, without the extend
    #Influence of aromatics rings
    feats.extend(rc_value)
    #Type atom of the closest atoms
    feats.extend(encoded_atoms) ##Before this was the fist entry +1
    #Type residue of the closest atoms
    feats.extend(encoded_residue) 
    #Distance value of the closest atoms
    feats.extend(values) 
    # All this is features are the X in the model
    X.append(feats)
  # print(X[0])
  # Get labels, the value of the CS of the target atom
  y = labels
  return X, y
  # return X_train_norm, X_test_norm, y_train, y_test, sc

#Process the PDB and BMRB file sto extract the information and after call get_features to encode that information 
def pipeline(combine, atomlist, studied_atoms, N):
  diccionario, cs_all, all_cylindrical_coordinates = get_dictionaries(combine, atomlist, studied_atoms)
  if combine != 'combine_test': # Just remove outliner in the training set
    cs_all = remove_outliners(cs_all)
  features, labels, keys_order, rc = get_features(cs_all, diccionario, all_cylindrical_coordinates)
  # print(features[0])
  # print(labels[0])
  # print(keys_order[0])
  return features, labels, keys_order, rc
  # return predictor(features, labels, keys_order, atomlist, N)

# Make the regressor into a function
def atom_model(atom, N = 15):
    print(N)
    np.random.seed(10) 
  #  try:
    #Define the atom to study
    studied_atoms = [atom]
    #Process the PDB and BMRB files to featurize 
    features, labels, keys_order, rc = pipeline(combine, atomlist, studied_atoms, N)
    print(features)
    print("Now labels")
    print(labels)
    #Split the data
    X_train, y_train = predictor(features, labels, keys_order, rc, atomlist, N)
    # y are my training labels
    # mu is the mean
    # std is the standard deviation
    def gaussian_noise(y,mu=0.0,std=0.1):
        noise = np.random.normal(mu, std, size = len(y))
        y_noisy = y + noise
        return y_noisy
    # Add Gaussian Noise to CS and append to augment training data
    #y_train_noise = gaussian_noise(y_train)
    #X_train_all = np.array(pd.concat((X_train, X_train), ignore_index=True))
    #y_train_all = np.array(pd.concat([y_train, y_train_noise], ignore_index=True))
    #Process PDB and BMRB of the test
    features_train, labels_train, keys_order_train, rc_train = pipeline(combine_test, atomlist, studied_atoms, N)
    #Split the train data
    X_test, y_test = predictor(features_train, labels_train, keys_order_train, rc_train, atomlist, N)
    #Train the model
    #reg = ExtraTreesRegressor(n_estimators=ntree, random_state=0, verbose=0).fit(X_train_all, y_train_all)
    reg = ExtraTreesRegressor(n_estimators=ntree, random_state=0, verbose=0).fit(X_train, y_train)
    #Set model name for each atom and residue pair
    filename = atom + '_model'
    #Print the lenght of the features. 3 of them to check are the same number
    print(f'a:{len(features)}')
    print(f'a:{len(labels)}')
    print(f'a:{len(keys_order)}')
    print(f'Y len{len(y_test)}')
    len_dataset_exp[f'{atom}-len'] = len(y_test)
    #Predict the y values of the set X saved to test  
    y_pred = reg.predict(X_test)
    #Use scipy module to score(r2 value) the prediction of the test
    slope, intercept, r_value, p_value, std_err = stats.linregress((y_test, y_pred))
    #Print the score of the module ExtraTreesRegressor
    print("R2test :", reg.score(X_test, y_test))
    #Print pearson R of the test
    print("Pearson R :", pearsonr(y_test, y_pred))
    #Print R2 of the scipy
    print("R2predscipy :", r_value**2)
    #Print MAE of the test
    print("MAE: ", median_absolute_error(y_test, y_pred))
    #Plot the values of the prediction of the test. Plot expt values vs predicted values
    fig1 = plt.figure()
    plt.title(f'{atom}_pred')
    plt.scatter(y_pred, y_test)
    plt.xlabel("y_pred")
    plt.ylabel("y_test")
    plt.savefig(f'{atom}_pred.png')
    #Save to csv file the scipy R2 and MAE of each value
    try:
        statsdict[atom] = {'R2': r_value**2, 'MAE': median_absolute_error(y_test, y_pred), 'P': pearsonr(y_test, y_pred)[0]}
    except:
        print('a')
        pass
    #Save model file
    pickle.dump(reg, open(filename, 'wb'))

#New atoms_all for trained depending atom type and residue type

#atoms_all = [('H1\'', ['C'])]
statsdict = dict()
len_dataset_exp = dict()
N = 100
ntree = 200
studied_atoms = ['H5']
#studied_atoms = atoms_all
# atoms_all = studied_atoms


for a in atoms_all:
  print(a)
  atom_model(a, N)

print(len_dataset_exp)
(pd.DataFrame.from_dict(data=statsdict, orient='index').rename_axis('Atom').to_csv('stats.csv', header=True))
