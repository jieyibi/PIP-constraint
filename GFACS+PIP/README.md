## GFACS with NLS for TSPTW

### Training

Train GFACS model for tsptw with `$N` nodes
```raw
$ python train_nar.py $N
```

### Dataset Generation

```raw
$ python utils.py
```

### Testing

Test GFACS for TSPTW with `$N` nodes
```raw
$ python test.py $N -p "path_to_checkpoint"
```

