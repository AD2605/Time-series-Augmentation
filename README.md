# Time-series-Augmentation
A simple deep 10 layered convolutional network for augmenting time series data. The model is tested on EEG signals on motor imagery data where signals of different classes are very similar and hence extremely close reproduction is necessary. The use Prelu instead of other activation functions really helps due to the learnable parameter. 
The model achieves an average MSELoss of 0.04804 on val data. 

