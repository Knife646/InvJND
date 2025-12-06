# InvJND
The official code and dataset of our paper:

"InvJND: Just Noticeable Difference Estimation via Deep Invertible Network"

# CPL-Set2
CPL-Set2 consists of 8965 pristine images and their corresponding CPL images, selected through carefully designed experiment. Firstly, we use the widely adopted HDR-VDP as an automated tool to efficiently generate pseudo-CPL labels at scale. Second, we incorporate human annotation for a subset of the dataset specifically for testing purposes.

![Image](https://github.com/Knife646/InvJND/blob/main/figure/CPL-Set2.png)

 If you need the CPL-Set2 for academic usage, please contact us via Email.

 # InvJND Pipeline
Pipeline of our proposed InvJND.

 ![Image](https://github.com/Knife646/InvJND/blob/main/figure/InvJND.png)

Overall, our InvJND model contains three modules: Residual Dense Block, Invertible Neural Networks and HF Modulation Block. We employ the Residual Dense Block to enhance the features and reconstruct the output features. The Invertible Neural Network is used to transform between shallow and deep feature space, and the HF Modulation Block modulates the high-frequency information in the deep feature space.

# Training and Generating
If you want to train InvJND, just execute the following command:

     train.py

If you want to generate JND maps, just execute the following command:

     test.py

The generated JND maps will be saved to ''/result''

Note that the key environment used in our experiments are:

     python == 3.7
   
     torch == 1.13.1 + cu117
   
     torchvision == 0.14.1

# Citation
If you find this repo useful, please cite our paper. Code will be available soon.

     
