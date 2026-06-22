import torch, glob

files = glob.glob('/sci/labs/benjamin.yakir/netanel.azran/data/hf_cache/models--ibm*/**/*.ckpt', recursive=True)
print('Found:', files)
ckpt = torch.load(files[0], map_location='cpu', weights_only=False)
hp = ckpt['hyper_parameters']
print('trainer_config:', hp['trainer_config'])
print('label_dict:', hp['label_dict'])
