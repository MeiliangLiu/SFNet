# SFNet: A Spatio-Frequency Domain Deep Learning Network for Efficient Alzheimer's Disease Diagnosis

The pyTorch implementation of SFNet

# Abstract
Alzheimer’s disease (AD) is a progressive neurodegenerative disorder that predominantly affects the elderly population and currently has no cure. Magnetic Resonance Imaging (MRI), as a non-invasive imaging technique, is essential for the early diagnosis of AD. MRI inherently contains both spatial and frequency information, as raw signals are acquired in the frequency domain and reconstructed into spatial images via the Fourier transform. However, most existing AD diagnostic models extract features from a single domain, limiting their capacity to fully capture the complex neuroimaging characteristics of the disease. While some studies have combined spatial and frequency information, they are mostly confined to 2D MRI, leaving the potential of dual-domain analysis in 3D MRI unexplored. To overcome this limitation, we propose a Spatio-Frequency Network (SFNet), the first end-to-end deep learning framework that simultaneously leverages spatial and frequency domain information to enhance 3D MRI-based AD diagnosis. SFNet integrates an enhanced dense convolutional network to extract local spatial features and a global frequency module to capture global frequency-domain representations. Additionally, a novel multi-scale attention module is proposed to further refine spatial feature extraction. Experiments on the Alzheimer's Disease Neuroimaging Initiative (ANDI) dataset demonstrate that SFNet outperforms existing baselines and reduces computational overhead in classifying cognitively normal (CN) and AD, achieving an accuracy of 95.1\%.


# Highlight

1. The first model to integrate local spatial features and global frequency-domain dependencies from 3D MRI, substantially enhancing the accuracy of Alzheimer's disease classification.

2. A novel multi-scale attention module designed to expand the receptive field and effectively capture spatial features at multiple scales.

3. A low-rank MLP layer employed in the frequency domain, enabling a reduction in model parameters and computational complexity without sacrificing performance.

4. Visualization of learnable global filters reveals spectral response patterns, improving the interpretability of the model.

