#  Neural View Synthesis and Matching for Semi-Supervised Few-Shot Learning of 3D Pose
## Release Notes
The official PyTorch implementation of Neural View Synthesis and Matching for Semi-Supervised Few-Shot Learning of 3D Pose, publish on NeuralIPS 2021.
![Example figure](https://github.com/Angtian/NeuralVS/blob/main/MatchingExamples.png)

## Installation
To install required libs:
```
git clone https://github.com/Angtian/NeuralVS.git
cd NeuralVS
pip install -r requirements.txt
```
We use the same data preprocess as NeMo does. You can run the data preprocess part in [NeMo](https://github.com/Angtian/NeMo) repo or run the following code:
```
git clone https://github.com/Angtian/NeMo.git
cd NeMo
chmod +x PrepareData.sh
./PrepareData.sh
cd ..
mv ./NeMo/data ./
```

## Matching using Single Image
Here we provide the code to run the pose matching experiment using single anchor image (section 4.3). To run the experiment using ImageNet pretrained backbone:
```
python .\code\SingleAnchorMatching.py --do_plot
```
To run the experiment with other backbone:
```
python .\code\SingleAnchorMatching.py --load_path {Path_to_saved_model} --do_plot
```

## Code for Semi-supervised learning 
Coming Soon
