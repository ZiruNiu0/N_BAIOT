import torch
from torch.utils.data import Dataset, DataLoader, random_split
from os import path
from sklearn.preprocessing import StandardScaler
import pandas as pd
import fedml
from fedml import FedMLRunner
from fedml.data.MNIST.data_loader import download_mnist, load_partition_data_mnist

sc = StandardScaler()

class MyDataset(Dataset):
 
  def __init__(self,file_name):
    price_df=pd.read_csv(file_name)
    x=price_df.iloc[:,0:115].values
    x=sc.fit_transform(x)
    y=price_df.iloc[:,115].values
    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.long)

  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]

def load_data(args):
    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)
    train_data_num = 0
    test_data_num = 0
    train_data_global = None
    test_data_global = None
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    class_num = 2
    if args.rank==0:
        #server data
        data_path = args.test_data_url
        dataset = MyDataset(data_path)
        train_data_global = None
        test_data_global = [DataLoader(dataset, args.batch_size, shuffle=False)]
        test_data_num = len(test_data_global)
    else:
        #client data
        data_path = path.expanduser(args.data_cache_dir)
        dataset = MyDataset(data_path)
        Train_dataset = DataLoader(dataset, args.batch_size, shuffle=True)
        train_data_local_dict[args.rank-1] = Train_dataset
        test_data_local_dict[args.rank-1] = None
        train_data_local_num_dict[args.rank-1] = len(Train_dataset)
        train_data_num += len(Train_dataset)
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        import torch
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_features):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_features, 64, bias=False)
        self.fc2 = torch.nn.Linear(64, 16, bias=False)
        self.fc3 = torch.nn.Linear(16, 4, bias=False)
        self.fc4 = torch.nn.Linear(4, 2, bias=False)
        self.act = torch.nn.Sigmoid()
        #self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.act(x)
        #x = x.squeeze(-1)
        '''
        #calculating proximal term:
        n1 = torch.from_numpy(starting_point[0])
        n2 = torch.from_numpy(starting_point[1])
        n3 = torch.from_numpy(starting_point[2])
        n4 = torch.from_numpy(starting_point[3])
        norm1 = torch.sum(torch.pow(self.fc1.weight-n1, 2))
        norm2 = torch.sum(torch.pow(self.fc2.weight-n2, 2))
        norm3 = torch.sum(torch.pow(self.fc3.weight-n3, 2))
        norm4 = torch.sum(torch.pow(self.fc4.weight-n4, 2))
        proxy = norm1 + norm2 + norm3 + norm4
        '''
        return x


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model (the size of MNIST image is 28 x 28)
    model = NeuralNetwork(115)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
