# Ray-X-wgan (rxwgan)


## How to run?

Go to the share file:

### First create your model (please check the content into the script):
```
python create_models.py
```

This will create the critic and generator (h5) models into your workspace.

### Run the training:

```
run_wgangp.py -i path_to_table.csv -o output_file -g generator.h5 -c critic.h5 --batch_size 16 --fold 0 --n_splits 5 --save_for_each 50 --disp_for_each 50 --epochs 1000
```

If for some reason you lost the machine, you can restart from the last saved configuration. For example, the last saved models came from epoch 100. To restart from epoch 100 + (1):

```
run_wgangp.py -i path_to_table.csv -o output_file -g generator_epoch_100.h5 -c critic_epoch_100.h5 --history history_epoch_100.json ----batch_size 16 --fold 0 --n_splits 5 --save_for_each 50 --disp_for_each 50 --start_from_epoch 101 --epochs 1000
```


## Reference Papers:

### Generative Adversarial Network in Medical Imaging: A Review (https://arxiv.org/abs/1809.07294)

Generative adversarial networks have gained a lot of attention in the computer vision community due to their capability of data generation without explicitly modelling the probability density function. The adversarial loss brought by the discriminator provides a clever way of incorporating unlabeled samples into training and imposing higher order consistency. This has proven to be useful in many cases, such as domain adaptation, data augmentation, and image-to-image translation. These properties have attracted researchers in the medical imaging community, and we have seen rapid adoption in many traditional and novel applications, such as image reconstruction, segmentation, detection, classification, and cross-modality synthesis. Based on our observations, this trend will continue and we therefore conducted a review of recent advances in medical imaging using the adversarial training scheme with the hope of benefiting researchers interested in this technique.

### U-Net: Convolutional Networks for Biomedical Image Segmentation (https://arxiv.org/abs/1505.04597):

There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at this http URL .

### A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation (https://arxiv.org/abs/1702.05970):

We propose a generalized focal loss function based on the Tversky index to address the issue of data imbalance in medical image segmentation. Compared to the commonly used Dice loss, our loss function achieves a better trade off between precision and recall when training on small structures such as lesions. To evaluate our loss function, we improve the attention U-Net model by incorporating an image pyramid to preserve contextual features. We experiment on the BUS 2017 dataset and ISIC 2018 dataset where lesions occupy 4.84% and 21.4% of the images area and improve segmentation accuracy when compared to the standard U-Net by 25.7% and 3.6%, respectively.

### Efficient Multi-Scale 3D CNN with Fully Connected CRF for Accurate Brain Lesion Segmentation (https://arxiv.org/abs/1603.05959):

We propose a dual pathway, 11-layers deep, three-dimensional Convolutional Neural Network for the challenging task of brain lesion segmentation. The devised architecture is the result of an in-depth analysis of the limitations of current networks proposed for similar applications. To overcome the computational burden of processing 3D medical scans, we have devised an efficient and effective dense training scheme which joins the processing of adjacent image patches into one pass through the network while automatically adapting to the inherent class imbalance present in the data. Further, we analyze the development of deeper, thus more discriminative 3D CNNs. In order to incorporate both local and larger contextual information, we employ a dual pathway architecture that processes the input images at multiple scales simultaneously. For post-processing of the network's soft segmentation, we use a 3D fully connected Conditional Random Field which effectively removes false positives. Our pipeline is extensively evaluated on three challenging tasks of lesion segmentation in multi-channel MRI patient data with traumatic brain injuries, brain tumors, and ischemic stroke. We improve on the state-of-the-art for all three applications, with top ranking performance on the public benchmarks BRATS 2015 and ISLES 2015. Our method is computationally efficient, which allows its adoption in a variety of research and clinical settings. The source code of our implementation is made publicly available.

### Data augmentation using learned transformations for one-shot medical image segmentation (https://arxiv.org/abs/1902.09383):

Image segmentation is an important task in many medical applications. Methods based on convolutional neural networks attain state-of-the-art accuracy; however, they typically rely on supervised training with large labeled datasets. Labeling medical images requires significant expertise and time, and typical hand-tuned approaches for data augmentation fail to capture the complex variations in such images. 
We present an automated data augmentation method for synthesizing labeled medical images. We demonstrate our method on the task of segmenting magnetic resonance imaging (MRI) brain scans. Our method requires only a single segmented scan, and leverages other unlabeled scans in a semi-supervised approach. We learn a model of transformations from the images, and use the model along with the labeled example to synthesize additional labeled examples. Each transformation is comprised of a spatial deformation field and an intensity change, enabling the synthesis of complex effects such as variations in anatomy and image acquisition procedures. We show that training a supervised segmenter with these new examples provides significant improvements over state-of-the-art methods for one-shot biomedical image segmentation. Our code is available at this https URL.

### Automatic skin lesion segmentation with fully convolutional-deconvolutional networks (https://arxiv.org/abs/1703.05165):

This paper summarizes our method and validation results for the ISBI Challenge 2017 - Skin Lesion Analysis Towards Melanoma Detection - Part I: Lesion Segmentation



