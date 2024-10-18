######################
# Import libraries
######################
import matplotlib.pyplot as plt
from matplotlib import cm
from rdkit.Chem.Draw import SimilarityMaps
from numpy import loadtxt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_ketcher import st_ketcher
import joblib
import pickle
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from IPython.display import HTML
from molvs import standardize_smiles
from math import pi
import zipfile
import base64


######################
# Page Title
######################

st.write("<h1 style='text-align: center; color: #FF8C00;'> Toxicity Estimator</h1>", unsafe_allow_html=True)
st.write("<h5 style='text-align: justify; color: black;'> Assessment of the acute toxicity of xenobiotics in oral and intravenous administration to rats. Find the toxicity of a compound in a database or predict its hazard level using QSAR models. Classification by toxicity classes for oral administration of toxicants is carried out in accordance with the classification of the World Health Organization</h5>", unsafe_allow_html=True)
if st.sidebar.button('Application description'):
    st.sidebar.write('The Toxicity Estimator application  allows to predict the level of toxicity (LD50,rat, oral or intravenous) for  compounds.  This application makes predictions based on Quantitative Structure-Activity Relationship (QSAR) models build on curated datasets. If experimental toxicity values are available for the compound, they are displayed in the summary table. The  models were developed using open-source chemical descriptors based on Morgan fingerprints and MACCS keys, along with the CatBoost. CatBoost is a machine learning method based on gradient boosting over decision trees. The models were generated applying the best practices for QSAR model development and validation widely accepted by the community. The applicability domain (AD) of the models was calculated as Dcutoff = ⟨D⟩ + Zs, where «Z» is a similarity threshold parameter defined by a user (0.5 in this study) and «⟨D⟩» and «s» are the average and standard deviation, respectively, of all Euclidian distances in the multidimensional descriptor space between each compound and its nearest neighbors for all compounds in the training set. Batch processing is available through https://github.com/ovttiras/Toxicity_Estimator.')


with open("manual.pdf", "rb") as file:
    btn=st.sidebar.download_button(
    label="Click to download brief manual",
    data=file,
    file_name="manual of Toxicity Estimator.pdf",
    mime="application/octet-stream"
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def rdkit_numpy_convert(f_vs):
    output = []
    for f in f_vs:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
        return np.asarray(output)
def toxicity_labels(s):
    if s.Hazard_Categories =='Ia class, Extremely hazardous' or s.Hazard_Categories =='Ib class,Highly hazardous':
        return ['background-color: red']*len(s)
    elif s.Hazard_Categories ==('II class,Moderately hazardous'):
        return ['background-color: yellow']*len(s)
    elif s.Hazard_Categories ==('III class, Slightly hazardous'):
        return ['background-color: blue']*len(s)
    else:
        return ['background-color: green']*len(s)

st.write("<h3 style='text-align: center; color: black;'> Step 1. Draw molecule or select input molecular files.</h3>", unsafe_allow_html=True)
files_option1 = st.selectbox('', ('Draw the molecule and click the "Apply" button','SMILES', '*CSV file containing SMILES', 'MDL multiple SD file (*.sdf)'))
if files_option1 == 'Draw the molecule and click the "Apply" button':
    smiles = st_ketcher(height=400)
    st.write(f'The SMILES of the created  chemical: "{smiles}"')
    if len(smiles)!=0:
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles),isomericSmiles = False)
        smiles=standardize_smiles(canon_smi)
        m = Chem.MolFromSmiles(smiles)
        inchi = str(Chem.MolToInchi(m))
        
if files_option1 == 'SMILES':
    SMILES_input = ""
    compound_smiles = st.text_area("Enter only one structure as a SMILES", SMILES_input)
    if len(compound_smiles)!=0:
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(compound_smiles),isomericSmiles = False)
        smiles=standardize_smiles(canon_smi)
        m = Chem.MolFromSmiles(smiles)
        inchi = str(Chem.MolToInchi(m))
        im = Draw.MolToImage(m)
        st.image(im)

if files_option1 == '*CSV file containing SMILES':     
    # Read input
    uploaded_file = st.file_uploader('The file should contain only one column with the name "SMILES"')
    if uploaded_file is not None:
        df_ws=pd.read_csv(uploaded_file, sep=';')
        count=0
        failed_mols = []
        bad_index=[]
        index=0
        for i in df_ws.SMILES: 
            index+=1           
            try:
                canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(i),isomericSmiles = False)
                df_ws.SMILES = df_ws.SMILES.replace (i, canon_smi)             
            except:
                failed_mols.append(i)
                bad_index.append(index)
                canon_smi='wrong_smiles'
                count+=1
                df_ws.SMILES = df_ws.SMILES.replace (i, canon_smi)
        st.write('CHEMICAL STRUCTURE VALIDATION AND STANDARDIZATION:')
        st.write(f'Original data: {len(df_ws)} molecules')
        st.write(f'Failed data: {count} molecules')

        if len(failed_mols)!=0:
            number =[]
            for i in range(len(failed_mols)):
                number.append(str(i+1))
            
            
            bad_molecules = pd.DataFrame({'No. failed molecule in original set': bad_index, 'SMILES of wrong structure: ': failed_mols, 'No.': number}, index=None)
            bad_molecules = bad_molecules.set_index('No.')
            st.dataframe(bad_molecules)


        moldf = []
        for i,record in enumerate(df_ws.SMILES):
            if record!='wrong_smiles':
                canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(record),isomericSmiles = False)
                standard_record = standardize_smiles(canon_smi)
                m = Chem.MolFromSmiles(standard_record)
                moldf.append(m)
        
        st.write('Kept data: ', len(moldf), 'molecules') 

# Read SDF file 
if files_option1 == 'MDL multiple SD file (*.sdf)':
    uploaded_file = st.file_uploader("Choose a SDF file")
    if uploaded_file is not None:
        st.header('CHEMICAL STRUCTURE VALIDATION AND STANDARDIZATION:')
        supplier = Chem.ForwardSDMolSupplier(uploaded_file,sanitize=False)
        failed_mols = []
        all_mols =[]
        wrong_structure=[]
        wrong_smiles=[]
        bad_index=[]
        for i, m in enumerate(supplier):
            structure = Chem.Mol(m)
            all_mols.append(structure)
            try:
                Chem.SanitizeMol(structure)
            except:
                failed_mols.append(m)
                wrong_smiles.append(Chem.MolToSmiles(m))
                wrong_structure.append(str(i+1))
                bad_index.append(i)

        
        st.write('Original data: ', len(all_mols), 'molecules')
        st.write('Failed data: ', len(failed_mols), 'molecules')
        if len(failed_mols)!=0:
            number =[]
            for i in range(len(failed_mols)):
                number.append(str(i+1))
            
            
            bad_molecules = pd.DataFrame({'No. failed molecule in original set': wrong_structure, 'SMILES of wrong structure: ': wrong_smiles, 'No.': number}, index=None)
            bad_molecules = bad_molecules.set_index('No.')
            st.dataframe(bad_molecules)

        # Standardization SDF file
        all_mols[:] = [x for i,x in enumerate(all_mols) if i not in bad_index] 
        records = []
        for i in range(len(all_mols)):
            record = Chem.MolToSmiles(all_mols[i])
            records.append(record)
        
        moldf = []
        for i,record in enumerate(records):
            canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(record),isomericSmiles = False)
            standard_record = standardize_smiles(canon_smi)
            m = Chem.MolFromSmiles(standard_record)
            moldf.append(m)
        
        st.write('Kept data: ', len(moldf), 'molecules') 


class Models():
    def __init__(self,administration:str, way_exp_data:str, way_model:str, descriprors:list, descripror_way_zip:str, descripror_way_csv:str, model_AD_limit:float):
        self.administration=administration
        self.way_exp_data=way_exp_data
        self.way_model=way_model
        self.descriprors=descriprors
        self.descripror_way_zip=descripror_way_zip
        self.descripror_way_csv=descripror_way_csv
        self.model_AD_limit=model_AD_limit
        # Load model and experimental dates
        self.model = pickle.load(open(self.way_model, 'rb'))
        self.df = pd.read_csv(self.way_exp_data)
        self.res = (self.df.groupby("inchi").apply(lambda x: x.drop(columns="inchi").to_dict("records")).to_dict())
        # Calculate molecular descriptors
        self.f_vs = self.descriprors
        self.X = rdkit_numpy_convert(self.f_vs)
        self.zf = zipfile.ZipFile(self.descripror_way_zip) 
        self.df = pd.read_csv(self.zf.open(self.descripror_way_csv))
        self.x_tr=self.df.to_numpy()

class one_molecules(Models):                
    def seach_predic(self):
        # search experimental toxicity value
        if inchi in self.res:
            exp_tox=float(self.res[inchi][0]['TOX_VALUE'])
            cas_id=str(self.res[inchi][0]['CAS_Number'])
            value_ped_tox='see experimental value'
            cpd_AD_vs_tox='-'            
        else:
         #Predict toxicity
            y_pred_con_tox = self.model.predict(self.X)           
            y_pred_con_tox_t=y_pred_con_tox[0]
            MolWt=ExactMolWt(Chem.MolFromSmiles(smiles))
            value_ped_tox=round((10**(y_pred_con_tox_t*-1)*1000)*MolWt, 4)
            # Estimination AD for toxicity
            neighbors_k_vs_tox = pairwise_distances(self.x_tr, Y=self.X, n_jobs=-1)
            neighbors_k_vs_tox.sort(0)
            similarity_vs_tox = neighbors_k_vs_tox
            cpd_value_vs_tox = similarity_vs_tox[0, :]
            cpd_AD_vs_tox = np.where(cpd_value_vs_tox <= self.model_AD_limit, "Inside AD", "Outside AD")
            exp_tox="-"
            cas_id="not detected" 

        if self.administration=='Oral':
            categories=''
            if isinstance(value_ped_tox,  float):
                if value_ped_tox  <5:
                    categories='Ia class, Extremely hazardous'
                elif 5<=value_ped_tox<=50:
                    categories='Ib class,Highly hazardous'
                elif 50<=value_ped_tox<=2000:
                    categories='II class,Moderately hazardous'
                elif 2000<value_ped_tox<5000:
                    categories='III class, Slightly hazardous'
                else:
                    categories='IV class, Unlikely to present acute hazard'
            else:
                if exp_tox<5:
                    categories='Ia class, Extremely hazardous'
                elif 5<=exp_tox<=50:
                    categories='Ib class,Highly hazardous'
                elif 50<=exp_tox<=2000:
                    categories='II class,Moderately hazardous'
                elif 2000<exp_tox<5000:
                    categories='III class, Slightly hazardous'
                else:
                    categories='IV class, Unlikely to present acute hazard'


            st.header('**Prediction results:**')    
            common_inf = pd.DataFrame({'SMILES':smiles, 'Predicted value toxicity, rat, oral, Ld50, mg/kg': value_ped_tox,
                'Applicability domain_tox': cpd_AD_vs_tox,
                'Experimental value toxicity, rat, oral, Ld50': exp_tox, 
                'CAS number': cas_id, 'Hazard_Categories':categories}, index=[1])
            predictions_pred=common_inf.astype(str) 
            st.dataframe(predictions_pred.style.apply(toxicity_labels, axis=1))
            st.write('Classification into toxicity classes (see column "Hazard_Categories") is carried out in accordance with the classification of the World Health Organization (https://www.who.int/publications/i/item/9789240005662)')
            st.image('Toxicity_labels.png') 
        else:
            st.header('**Prediction results:**')    
            common_inf = pd.DataFrame({'SMILES':smiles, 'Predicted value toxicity, rat, intravenous, Ld50, mg/kg': value_ped_tox,
            'Applicability domain_tox': cpd_AD_vs_tox,
            'Experimental value toxicity,rat, intravenous, Ld50': exp_tox, 
            'CAS number': cas_id}, index=[1])
            predictions_pred=common_inf.astype(str) 
            st.dataframe(predictions_pred) 

class set_molecules(Models):
    def seach_predic_csv(self):
        # search experimental value     
        exp_tox=[]
        cas_id=[]
        y_pred_con_tox=[]
        cpd_AD_vs_tox=[]
        struct=[]
        number =[]
        count=0
        for m in moldf:
            inchi = str(Chem.MolToInchi(m))
            i=Chem.MolToSmiles(m)
            struct.append(i)
            # search experimental toxicity value
            if inchi in self.res:
                exp_tox.append(self.res[inchi][0]['TOX_VALUE'])
                cas_id.append(str(self.res[inchi][0]['CAS_Number']))
                y_pred_con_tox.append('see experimental value')
                cpd_AD_vs_tox.append('-')
                count+=1         
                number.append(count)
                
            else:
                # Estimination AD for toxicity
                neighbors_k_vs_tox = pairwise_distances(self.x_tr, Y=self.X, n_jobs=-1)
                neighbors_k_vs_tox.sort(0)
                similarity_vs_tox = neighbors_k_vs_tox
                cpd_value_vs_tox = similarity_vs_tox[0, :]
                cpd_AD_vs_tox_r = np.where(cpd_value_vs_tox <= self.model_AD_limit, "Inside AD", "Outside AD")
                # calculate toxicity
                y_pred_tox = self.model.predict(self.X)                    
                MolWt=ExactMolWt(m)
                value_ped_tox=(10**(y_pred_tox*-1)*1000)*MolWt
                value_ped_tox=round(value_ped_tox[0], 4)
                y_pred_con_tox.append(value_ped_tox)
                cpd_AD_vs_tox.append(cpd_AD_vs_tox_r[0])
                exp_tox.append("-")
                cas_id.append("not detected")
                count+=1         
                number.append(count) 

 
        if self.administration=='Oral':
            categories=''
            if isinstance(value_ped_tox,  float):
                if value_ped_tox  <5:
                    categories='Ia class, Extremely hazardous'
                elif 5<=value_ped_tox<=50:
                    categories='Ib class,Highly hazardous'
                elif 50<=value_ped_tox<=2000:
                    categories='II class,Moderately hazardous'
                elif 2000<value_ped_tox<5000:
                    categories='III class, Slightly hazardous'
                else:
                    categories='IV class, Unlikely to present acute hazard'
            else:
                if exp_tox<5:
                    categories='Ia class, Extremely hazardous'
                elif 5<=exp_tox<=50:
                    categories='Ib class,Highly hazardous'
                elif 50<=exp_tox<=2000:
                    categories='II class,Moderately hazardous'
                elif 2000<exp_tox<5000:
                    categories='III class, Slightly hazardous'
                else:
                    categories='IV class, Unlikely to present acute hazard'
            common_inf = pd.DataFrame({'SMILES':struct, 'No.': number,'Predicted value toxicity, rat, oral, Ld50, mg/kg': y_pred_con_tox,
            'Applicability domain_tox': cpd_AD_vs_tox,
            'Experimental value toxicity, Ld50': exp_tox,
            'CAS number': cas_id, 'Hazard_Categories':categories}, index=None)
            predictions_pred = common_inf.set_index('No.')
            predictions_pred=predictions_pred.astype(str)
            st.dataframe(predictions_pred.style.apply(toxicity_labels, axis=1))
            st.write('Classification into toxicity classes (see column "Hazard_Categories") is carried out in accordance with the classification of the World Health Organization (https://www.who.int/publications/i/item/9789240005662)')
            st.image('Toxicity_labels.png') 

        else:
            common_inf = pd.DataFrame({'SMILES':struct, 'No.': number,'Predicted value toxicity, Ld50, mg/kg': y_pred_con_tox,
            'Applicability domain_tox': cpd_AD_vs_tox,
            'Experimental value toxicity, Ld50': exp_tox,
            'CAS number': cas_id}, index=None)
            predictions_pred = common_inf.set_index('No.')
            predictions_pred=predictions_pred.astype(str)
            st.dataframe(predictions_pred)

        def convert_df(df):
            return df.to_csv().encode('utf-8')  
        csv = convert_df(predictions_pred)

        st.download_button(
            label="Download results of prediction as CSV",
            data=csv,
            file_name='Results.csv',
            mime='text/csv',
        )

st.write("<h3 style='text-align: center; color: black;'> Step 2. Select administration of substance.</h3>", unsafe_allow_html=True)
files_option2 = st.selectbox('', ('Oral','Intravenous'))
if (files_option1 =='Draw the molecule and click the "Apply" button' or files_option1 =='SMILES')  and files_option2 =='Intravenous':
    if st.button('Run predictions!'):
        Intrav_one=one_molecules('Intravenous', 'datasets/rat_intravenous_LD50_inchi.csv', 'Models/Intravenous_CatBoost_MF.pkl',
                             [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024, useFeatures=False, useChirality=False)],
                              'Models/Intravenous_x_tr_MF.zip', 'Intravenous_x_tr_MF.csv', 4.8)
        Intrav_one.seach_predic()

if (files_option1 =='Draw the molecule and click the "Apply" button' or files_option1 =='SMILES')  and files_option2 == 'Oral':
    if st.button('Run predictions!'):
        Oral_one=one_molecules('Oral', 'datasets/rat_oral_LD50_inchi.csv', 'Models/Oral_CatBoost_MACCS.pkl',
                             [MACCSkeys.GenMACCSKeys(m)],
                              'Models/Oral_x_tr_MACCS.zip', 'Oral_x_tr_MACCS.csv', 2.48)
        Oral_one.seach_predic()
 
if (files_option1  =='*CSV file containing SMILES' or files_option1=='MDL multiple SD file (*.sdf)')  and files_option2 =='Intravenous':
    if st.button('Run predictions!'):
        Intrav_set=set_molecules('Intravenous', 'datasets/rat_intravenous_LD50_inchi.csv', 'Models/Intravenous_CatBoost_MF.pkl',
                             [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024, useFeatures=False, useChirality=False)],
                              'Models/Intravenous_x_tr_MF.zip', 'Intravenous_x_tr_MF.csv', 4.8)
        Intrav_set.seach_predic_csv()

if (files_option1  =='*CSV file containing SMILES' or files_option1=='MDL multiple SD file (*.sdf)')  and files_option2 =='Oral':
    if st.button('Run predictions!'):
        Oral=set_molecules('Oral', 'datasets/rat_oral_LD50_inchi.csv', 'Models/Oral_CatBoost_MACCS.pkl',
                             [MACCSkeys.GenMACCSKeys(m)],
                              'Models/Oral_x_tr_MACCS.zip', 'Oral_x_tr_MACCS.csv', 2.48)
        Oral.seach_predic_csv()