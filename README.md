# A Computer Vision Approach to Tracking Boxing

## Set-up

```bash
pip install -r requirements.txt
```

## To run

```bash
python main.py -p /path/to/videos -o /path/to/output
```
This will iterate thru each video, find all the punches thrown and list the hits landed and output a JSON file with the results. 

There is also a set of sample videos in this directory, to run an inference on those, simply run:

```bash
python main.py -p ./samples/dataset
```

## To train

Since there are multiple models at play here, you need multiple datasets to train. Will explain the formatting of each in the following sections. 

## HitNet

HitNet is the network that classifies a punch as land or not land. To train this a dataset of images are needed. Ideally it would be a directory of named images with "land" as a part of the name for punches that land and "miss" in the name for punches that miss.

```
├── punch_imgs_cropped
│   ├── punch_land118.jpg
│   ├── punch_land100.png
│   ├── punch_miss101.png
│   ├── punch_miss102.png
|   ...
```

When it is structured this way, the training script will automatically generate a dataset based on the file name and begin training. 

Assuming a correctly set up dataset one can call the training script as such:

```bash
cd compubox/
python3 compubox/models/train/train_hitnet.py --d /path/to/dataset --epochs <num epochs> --learning_rate <desired learning rate>
```
This will save a model `hitnet_model.pth` into the root of the compubox directory.

## ActionNet

ActionNet is the model that classifies the action of each punch. The dataset required for this is a folder with a named directory for each punch and in each directory a set of videos of the punch. 

```
├── Boxing_clips
│   ├── 1
│   │   ├── 20231128_042445000_iOS.MOV
│   │   ├── 20231128_042501000_iOS.MOV
|   |   ...
│   ├── 2
│   │   ├── 20231128_043327000_iOS.MOV
│   │   ├── 20231128_043330000_iOS.MOV
│   │   ├── 20231128_043333000_iOS.MOV
|   |   ...
```

Once this set of videos are on the device, one can call the script to annotate said videos and generate a JSON dataset of each video, its punch and the time series of poses:

```bash
python3 compubox/models/train/annotate_videos.py --dataset-path /home/shangar21/Downloads/Boxing_clips
```

This will output a file `dataset.json` into the root of the compubox directory. This is the file that will ultimately be used to train the ActionNet network. Once this json is generated, one can call the training script with:

```bash
python3 compubox/models/train/train_actionnet.py --dataset-path ./dataset.json --epochs <num empochs> --learning_rate <desired learning rate>
```

## Samples
A sample with one person throwing one punch:


```
{"20231128_042445000_iOS.MOV": {"1.0": {"punches": ["1"], "landed": false}}}
```


https://github.com/shangar21/compubox/assets/56494763/85612571-4b61-4bfc-bcd0-431b441d1570

The output when running on the sample in this repository:

```
{
   "cross_hook.mp4":{
      "1.0":{
         "punches":[
            "2"
         ],
         "landed":false
      },
      "2.0":{
         "punches":[
            "2"
         ],
         "landed":false
      }
   },
   "20231128_043330000_iOS.MOV":{
      "1.0":{
         "punches":[
            "2"
         ],
         "landed":false
      }
   },
   "20231128_044321000_iOS.MOV":{
      "1.0":{
         "punches":[
            "3"
         ],
         "landed":false
      }
   },
   "jab_cross.mp4":{
      "1.0":{
         "punches":[
            "2"
         ],
         "landed":false
      },
      "2.0":{
         "punches":[
            "2"
         ],
         "landed":false
      }
   },
   "20231128_042445000_iOS.MOV":{
      "1.0":{
         "punches":[
            "2"
         ],
         "landed":false
      }
   }
}
```







