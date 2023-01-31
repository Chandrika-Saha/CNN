# Bangla Isolated Handwritten Basic Characters, Compound Characters and Digits Recognition Using Deep Convolutional Neural Network 

Bangla is one of the mostly spoken languages all over the world. 
Optical Character Recognition (OCR) is a vital task for any language 
because of its use in different sectors of life. Handwritten Character
Recognition (HCR) is a form of OCR that classifies human written
texts. HCR is a very useful tool for modern era of digitization. We
need HCR for data digitizing, robotics vision, helping visually disabled
people and many more sectors. For building a complete HCR, first the
text component from an image need to be located, then sentences and
words and lastly isolated characters are needed to be separated. So, it
is needed to ensure that we have a classifier that can classify isolated
characters excellently. Isolated handwritten character recognition is
considered a hard image classification problem as writing style varies
human to human. However, despite Bangla being a popular language,
there is relatively less number of research works on Bangla HCR. This
thesis aims to find an effective and efficient technique to recognize
Bangla isolated characters and digits. After exploring several state-of-
arts machine learning technique it was found that a better approch is
needed for such a difficult task. Now-a-days, Deep Convolutional Neu-
ral Network (DCNN) is gaining popularity for better performances for
image classification but in the case of Bangla handwritten characters
it is relatively less explored. So, a DCNN model for Bangla handwrit-
ten basic characters, compound characters and digits recognition is
proposed. Rigorous experimentation on Bangla handwritten datasets
have showed superior accuracies than state-of-arts machine learning al-
gorithms. Using a 14 layered DCNN model containing 6 convolutional
layer, 6 max pooling layer and 2 fully connected layer with dropout
regularization, namely BanglaNet-14 a recognition accuracy of 95.62%
is obtained for a total of 231 classes of Bangla basic, compound char-
acters and digits. For evaluation three benchmark datasets namely
CMATERdb 3.1.2 (basic characters), CMATERdb 3.1.3.1 (compound
characters) and CMATERdb 3.1.1 (digits) are used. For CMATERdb
3.1.2, the proposed improved DCNN model gives a recognition ac-
curacy of 97.67% , for CMATERdb 3.1.3.1 a recognition accuracy of
96.33% and for CMATERdb 3.1.1 accuracy of 98.35% is obtained. For
building a complete Bangla HCR we need to have a classifire that can
detect any Bangla basic characters, compound characters and digits.
For this purpose CMATERdb 3.1.2, CMATERdb 3.1.3.1 and CMA-
TERdb 3.1.1 datasets were combined together that forms a total of 231
classes. Recognition accuracy for this dataset is 95.62%. Also two dif-
fernt datasets, namely, BanglaLekha isolated and Ekush dataset con-
taining 84 and 122 classes respectively are also considered for evaluat-
ing handwritten basic, compound characters and digits. The improved
DCNN model yields a recognition accuracy of 95.94% on BanglaLekha
isolated dataset and 93.86% accuracy on Ekush dataset. All these ex-
perimental evaluation is shown to give superior recognition accuracies
then existing prominent techniques.
