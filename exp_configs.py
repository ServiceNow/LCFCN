from haven import haven_utils as hu

EXP_GROUPS = {}
EXP_GROUPS['trancos'] =  {"dataset": {'name':'trancos', 
                          'transform':'rgb_normalize'},
         "model": {'name':'lcfcn','base':"fcn8_vgg16"},
         "batch_size": [1,5,10],
         "max_epoch": [100],
         'dataset_size': [
                          {'train':'all', 'val':'all'},
                          ],
         'optimizer':['adam'],
         'lr':[1e-5]
         }

EXP_GROUPS['shanghai'] =  {"dataset": {'name':'shanghai', 
                            'transform':'rgb_normalize'},
         "model": {'name':'lcfcn','base':"fcn8_vgg16"},
         "batch_size": [1],
         "max_epoch": [100],
         'dataset_size': {'train':'all', 'val':'all'},
         'optimizer':['adam'],
         'lr':[1e-5]
         }

EXP_GROUPS['trancos_debug'] =  {"dataset": {'name':'trancos', 
                          'transform':'rgb_normalize'},
         "model": {'name':'lcfcn','base':"fcn8_vgg16"},
         "batch_size": [1,5,10],
         "max_epoch": [100],
         'dataset_size': [
                        {'train':1, 'val':1},
                          # {'train':'all', 'val':'all'},
                          ],
         'optimizer':['adam'],
         'lr':[1e-5]
         }

EXP_GROUPS = {k: hu.cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}