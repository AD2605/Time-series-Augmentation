from CNNAE import Encoder
from Dataloader import get_dataloader


train_Data = get_dataloader('path to data', batch=1)
test_Data = get_dataloader('Path to data', batch=1)

model = Encoder().cuda()
model.train_model(model=model, train=train_Data, test=test_Data, epochs=600)
