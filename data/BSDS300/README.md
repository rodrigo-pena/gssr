# The Berkeley Segmentation Dataset (BSDS300)

Source: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/

### Instructions to get the necessary files

- Go to https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
- Download the [images][images] (22MB) and the [human segmentations][human] (27MB) files.
- Extract these files and copy the folders `human/` and `images/`, as well as the files `iids_train.txt` and `iids_test.txt` onto the current directory.
- IMPORTANT: for the BSDS300-related code in the `gssr` repository to work, you must have the following directory structure:

```
BSDS300 /
			 /human/
			 /images/
			 iids_train.txt
			 iids_test.txt
```

### Provided files

| File             | Description                                                  |
| :--------------- | :----------------------------------------------------------- |
| `seg-format.txt` | Specification of the structure of the `.seg` files under the folder `human/` |



[images]: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz
[human]: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-human.tgz