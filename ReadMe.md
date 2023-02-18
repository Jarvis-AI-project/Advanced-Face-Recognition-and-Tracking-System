## Things to include:

- liveness detection
- false/ spoof face detection and ignorence
- face tracking
- self improving and making its own datasets
- preserve detailed log file

## Flow of program

- list files in data directory -done
- extract frontal face (harcascade -done | )
- resize all images to same size 150x150x1
- augment positive dataset
- generate test and train object (test_images, test_labels, train_images, train_labels)
