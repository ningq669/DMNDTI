import numpy as np
from DAE import DAE



drug_train = np.loadtxt(r"../feature/drug_vector.txt")
protein_train = np.loadtxt(r"../feature/protein_vector.txt")

drug_size=drug_train.shape[1]
protein_size=protein_train.shape[1]

print(drug_size,protein_size)
data1=DAE(drug_train,drug_size,20,32,1,64,[64])
np.savetxt('../DAE/drug_dae_d64.txt',data1)

data2=DAE(protein_train,protein_size,20,32,1,64,[64])
np.savetxt('../DAE/protein_dae_d64.txt',data2)