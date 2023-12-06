import random
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# Define dataset loader
class MyDataset(Dataset):
    def __init__(self,input,label,transform=None):
        self.input=input
        self.label=label
        self.transform=transform

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,index):
        return self.input[index],self.label[index]


# Define model
class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1)
        self.activation1 = nn.Softsign()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.bn1 = nn.BatchNorm2d(3)

        self.conv2 = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.activation2 = nn.Softsign()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.bn2 = nn.BatchNorm2d(6)

        self.conv3 = nn.Conv2d(6, 12, kernel_size=3, stride=1)
        self.activation3 = nn.Softsign()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=0.2)
        self.bn3 = nn.BatchNorm2d(12)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(588, 11)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.activation3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = self.bn3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        
        return x


# Server
class Server(object):
    def __init__(self, model, eval_dataset, num_clients):
        
        self.global_model = model
        self.num_clients = num_clients

        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32)
	
    def model_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * (1/self.num_clients)   # average
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def model_eval(self):
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data = batch[0]
            target = torch.squeeze(batch[1]).int()
            target = torch.tensor(target, dtype=torch.int64)

            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            
            output = self.global_model(data)
            total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 *(float(correct) / float(dataset_size))
        loss = total_loss / dataset_size

        return acc, loss


# Client
class Client(object):
    def __init__(self, model, train_dataset, id = -1):
                self.local_model = model
                self.client_id = id
                self.train_dataset = train_dataset
                self.train_loader = torch.utils.data.DataLoader(self.train_dataset[id], batch_size=32)

    def local_train(self, model):
	    
            for name, param in model.state_dict().items():
                self.local_model.state_dict()[name].copy_(param.clone())
            optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.001, momentum=0.0001)
            self.local_model.train()
            for e in range(3):
                for batch_id, batch in enumerate(self.train_loader):
                    data = batch[0]
                    target = torch.squeeze(batch[1]).int()
                    target = torch.tensor(target, dtype=torch.int64)

                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()

                    optimizer.zero_grad()
                    output = self.local_model(data)
                    loss = nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                
            diff = dict()
            for name, data in self.local_model.state_dict().items():
                diff[name] = (data - model.state_dict()[name])
            
            return diff


# load data
train_data_path = 'FL-FD/Train_data.pkl'
test_data_path = 'FL-FD/Test_data.pkl'

train_data = pickle.load(open(train_data_path, 'rb'))
test_data = pickle.load(open(test_data_path, 'rb'))


# initialize model
mymodel = MyModel()
mymodel = mymodel.double()
mymodel = mymodel.cuda()


# hyperparameters
max_acc = 80    # thorshold of accuracy (80%), for saving best model
epoch = 200
total_client = 15   # total number of clients
num_clients = 12    # number of clients selected per round


# initialize server and clients
server = Server(mymodel, test_data, num_clients)
clients = []

for c in range(total_client):
	clients.append(Client(server.global_model, train_data, id = c))


# train
for e in range(epoch):
    candidates = random.sample(clients, num_clients)            # randomly select clients
    weight_accumulator = {}
    for name, params in server.global_model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(params)     # initialize weight_accumulator
    for c in candidates:
        diff = c.local_train(server.global_model)               # train local model
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name].add_(diff[name])           # update weight_accumulator
    server.model_aggregate(weight_accumulator)                  # aggregate global model
    acc, loss = server.model_eval()
    print("Epoch %d, global_acc: %f, global_loss: %f\n" % (e, acc, loss))
    if acc > max_acc:
        max_acc = acc
        torch.save(server.global_model.state_dict(), 'model.pth')
        print("save model")


# test
model = mymodel
model.load_state_dict(torch.load('model.pth'))
model.eval()
model = model.cuda()

y_test, y_predict = [], []

test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

for batch_id, batch in enumerate(test_loader):
    data = batch[0]
    target = torch.squeeze(batch[1]).int()
    target = torch.tensor(target, dtype=torch.int64)

    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()

    output = model(data)
    y_test.extend(target.cpu().numpy())
    y_predict.extend(torch.argmax(output, dim=1).cpu().numpy())


# classification_report
print(classification_report(y_test, y_predict,
      target_names = [f'A{i}' for i in range(1, 12)],digits=4))

# confusion matrix
plt.figure(dpi=150, figsize = (6,4))
classes = [f'A{i}' for i in range(1, 12)]
mat = confusion_matrix(y_test, y_predict)

df_cm = pd.DataFrame(mat, index = classes, columns = classes)
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True label', fontsize=15)
plt.xlabel('Predicted label', fontsize=15)                                          
plt.show()
