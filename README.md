# Where are the Blobs: Counting by Localization with Point Supervision [[Paper]](https://arxiv.org/abs/1807.09856)

## Description

TBA


## Requirements

- Pytorch version 0.4 or higher.

## Running pretrained models

To obtain the test results, run the following command,

```
python summary.py -e trancos
```


## Training the models

To train the model,

```
python train.py -e trancos
```


## Testing the models

To test the model,

```
python test.py -e trancos
```

## Citation 
If you find the code useful for your research, please cite:

```bibtex
@Article{laradji2018blobs,
    title={Where are the Blobs: Counting by Localization with Point Supervision},
    author={Laradji, Issam H and Rostamzadeh, Negar and Pinheiro, Pedro O and Vazquez, David and Schmidt, Mark},
    journal = {ECCV},
    year = {2018}
}
```
