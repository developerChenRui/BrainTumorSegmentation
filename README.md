# Brain  Tumor  Segmentation
基于深度学习的方法，设计出一个对于 MRI 脑肿瘤的自动分割医学智能系统，针对特定脑肿 瘤 HGG 类型将其划分为五类即非肿瘤区域、瘤腺体(edema)、骨疽(necrosis)、无强化肿块 (non-enhancing tumor)以及强化肿块区域(enhancing tumor)。我们对不同序列下(T1、 T1c、T2、Flair)的 MRI 图像采用了偏移场矫正[1]、灰度归一化预处理[2]，并且对样本数 据进行均衡化后作为网络的四个通道输入到 CNN 网络进行训练，以达到网络对每一类都能学 习其特征的目的。训练得到网络权重后，测试相应 MRI 肿瘤图像。一些小的块状组织会被错 分为肿瘤，为此，我们进行了后期处理，利用高级形态学处理[3]，设置了一个阈值，将小于 这个阈值的块状组织移除。最后设计了 GUI 以展示我们脑肿瘤分割系统。
</br>

<img src="https://github.com/developerChenRui/BrainTumorSegmentation/blob/master/GUI.png" width="50%" height="50%">
