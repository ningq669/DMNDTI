# DMNDTI：A Duplex Multi-view Network-based Method for Drug-target Interaction Prediction
### Quick start
We provide an example script to run experiments on our dataset: 

### All process
 -Run `./main.py`   You can run the entire model


### Code and data

#### 
- `CLaugmentdti.py`: data augment for graph contrastive learning
- `model.py`: DMNDTI model
- `utils.py`: tool kit
- `main.py`: use the dataset to run DMNDTI 
- `GCNLayer.py`: a GCN layers 
- `DAE-model/run_DAE.py`: DAE feature

#### data `data/` directory
- `drug_id.txt`: list of drug ids
- `protein_id.txt`: list of protein idss
- `disease_name.txt`: list of disease names
- `side-effect-name.txt`: list of side effect names
- `mat_drug_se.txt` 		: Drug-SideEffect association matrix
- `mat_protein_protein.txt` : Protein-Protein interaction matrix
- `mat_drug_protein.txt`: Drug_Protein interaction matrix
- `mat_drug_drug.txt` 		: Drug-Drug interaction matrix
- `mat_protein_disease.txt` : Protein-Disease association matrix
- `mat_drug_disease.txt` 	: Drug-Disease association matrix




