import numpy as np

file_name = './failed_runs'

lines = np.loadtxt(file_name,dtype= str,delimiter='/')

lr_dict = {
    '1e-05': 0.00001,
    '1e-06': 0.000001
}
models_dict = {
    'VanillaVAE': 'configs/vae.yaml',
    'IWAE':'configs/iwae.yaml',
    'LogCoshVAE':'configs/logcosh_vae.yaml',
    'BetaVAE':'configs/bbvae.yaml',
    'BetaTCVAE':'configs/betatc_vae.yaml',
    'CategoricalVAE':'configs/cat_vae.yaml',
    'DIPVAE':'configs/dip_vae.yaml'
}
lr_list = []
ld_list = []
optimizer_list =[]
init_list=[]
model_list =[]
model_list_conv=[]

for i in lines:
    model_list.append(i[-1])
    lr_list.append(i[-2])
    ld_list.append(i[-3])
    optimizer_list.append(i[-4])
    init_list.append(i[-5])

for i in model_list:
    model_list_conv.append("../code/"+models_dict[i])

ld_list = list(map(int,ld_list))

lr_list = list(map(float,lr_list))
#lr_list = ['%.8f' % elem for elem in lr_list]

print(lr_list)
print(ld_list)
print(optimizer_list)
print(init_list)
print(model_list_conv)


print("Hello")