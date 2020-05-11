from . import fcn8_resnet, fcn8_vgg16


def get_base(base_name, exp_dict, n_classes):
    if base_name == "fcn8_resnet":
        model = fcn8_resnet.FCN8()
    
    elif base_name == "fcn8_vgg16":
        model = fcn8_vgg16.FCN8_VGG16(n_classes=n_classes)

    else:
        raise ValueError('%s does not exist' % base_name)

    return model