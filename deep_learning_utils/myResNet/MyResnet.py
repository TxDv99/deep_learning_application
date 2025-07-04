from functools import reduce
import torch
import torch.nn as nn  
import torch.nn.functional as F 
from torch.utils.data import random_split 
from torch.utils.data.dataloader import DataLoader
from operator import mul





class MyResNet(nn.Module): 
    def __init__(self,layers_tuple_list = None,data_shape = None, skip_dict = None, debug = False):
        super().__init__() #chiamo costruttore           
        self.layers = nn.ModuleList()
        self.skips = skip_dict 
        self._previous_is_conv = False
        self._hidden_size = 0
        self.data_shape = data_shape
        self.flatten = nn.Flatten()
        self.debug = debug
        self.device = None #determined by trainer

        #layers tuple is a tuple list
        if layers_tuple_list is not None:
            for layer_number,layer_info in enumerate(layers_tuple_list):
                
                if layer_number == len(layers_tuple_list) -1:
                    last_layer = True
                else:
                    last_layer = False
                
                
                
                layer_type = layer_info[0]
                act_fun = layer_info[-1]
            
                if layer_type == "Linear":

                    self.addLinear(layer_info, act_fun, last_layer)
            
                elif layer_type == "Conv2d" :
                    self.addConv(layer_info, act_fun, last_layer)

                elif layer_type in ["BatchNorm2d", "MaxPool2d", "Dropout"]:
                    layer_cls = getattr(nn, layer_type)
                    
                    self.layers.append(layer_cls(layer_info[1]))


                else:
                    print(f'Error, unexpected layer type: {layer_type}, layer type must be one of: Linear, Conv2d, BatchNorm2d, Dropout, MaxPool2d')

    
    def addConv(self, layer_info,act_fun, last_layer = False):
        
        padding_m = layer_info[6] if len(layer_info) >= 6 else 'zeros'

                

        conv = nn.Conv2d(
                in_channels=layer_info[1],
                out_channels=layer_info[2],
                kernel_size=layer_info[3],
                stride=layer_info[4],
                padding=layer_info[5],             
                padding_mode=padding_m               
                )
                
        self.layers.append(conv)
                
        self._previous_is_conv = True

        if last_layer == False:

           if act_fun is not None:
               self.layers.append(getattr(nn,act_fun)())
           else:
               self.layers.append(nn.ReLU())



    
    def addLinear(self, layer_info, act_fun, last_layer = False):
        if self._previous_is_conv:

            ###adattatore se dopo voglio mettere un MLP###
            n_channels,H_in, W_in = self.data_shape
            with torch.no_grad():
                dummy = torch.zeros(1, n_channels, H_in, W_in)
                out = nn.Sequential(*self.layers)(dummy)
                self._hidden_size = reduce(mul,tuple(out.shape))


            self._build_adapter(layer_info)
            self._previous_is_conv = False              

        in_size = layer_info[1]
        width = layer_info[2]
        depth = layer_info[3]
        out_size = layer_info[4]
                
        dims = [in_size] + [width] * depth + [out_size]

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        
            if last_layer == False:
                if act_fun is not None:
                    self.layers.append(getattr(nn, act_fun)())
                else:
                    self.layers.append(nn.ReLU())

                
    def _build_adapter(self,layer_info):
        if layer_info[1] != self._hidden_size:
            adapter = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(self._hidden_size, layer_info[1])  
                    )
            self.layers.append(adapter)
        

    def _reproject(self,skip_saved,idx_from,arrival):
        
        to_proj = skip_saved[idx_from]
        
        if to_proj.size() != arrival.size(): 

            if to_proj.ndim == 4 and arrival.ndim == 4:

                (batch,n_ch_in, H_in,W_in) = tuple(to_proj.shape)
                (batch,n_ch_out, H_out,W_out) = tuple(arrival.shape)

                stride_h = max(1, H_in // H_out)
                stride_w = max(1, W_in // W_out)

                conv = nn.Conv2d(n_ch_in, n_ch_out, kernel_size=1,
                         stride=(stride_h, stride_w)).to(self.device)

                projected = conv(to_proj)

                del conv

                
                projected = F.interpolate(projected, size=(H_out, W_out), mode='bilinear', align_corners=False)

                return projected


            elif arrival.ndim == 2:
                
                to_proj = to_proj.view(-1)/to_proj.size()[0]
                projector = nn.Linear(to_proj.size()[0], arrival.size()[1]).to(self.device)
                
                
                projected = projector(to_proj)
                
                del projector
            
                return(projected)
        
        else:
            return(to_proj)
        



        
        #if to_proj.shape != arrival.shape:
        #    projector = nn.Linear(skip_from.shape[1],arrival.shape[1]).to(arrival.device)
        #    projection = projector(skip_from)
        #else:
        #    projection = skip_from

        #return(projection)
    
    
    def forward(self, x):
       
       skip_dict = self.skips
       skip_outputs = {}
       
       for index, layer in enumerate(self.layers):
          if layer.__class__.__name__ == "Linear" and len(x.shape) > 1:
                x = x.flatten(start_dim=1)
              
          x = layer(x)

          
          if skip_dict is not None:

            #initial_check
            for from_idx,to_idx in skip_dict.items():
                if from_idx not in range(len(self.layers)) or to_idx not in range(len(self.layers)):
                    raise IndexError('Can not find index of the layer in the skip connection dictionary')
                   

            if index in skip_dict.keys():
               skip_outputs[skip_dict[index]] = x  
            elif index in skip_outputs:
               x = x + self._reproject(skip_outputs,index,x)  
            
       return x
        
    
    def test(self, dataset, batch_size = 64, num_workers = 2, loss_fn = torch.nn.CrossEntropyLoss(), data_split = None):
        
        device = next(self.parameters()).device
        self.eval()  # modalità evaluation
        total_loss = 0.0
        correct = 0
        total = 0

        if data_split != None:
            dataset_size = len(dataset)
            val_size = int(data_split[0] * dataset_size)
            train_size = int(data_split[1] * dataset_size)
            test_size = dataset_size - val_size - train_size if len(data_split) == 2 else int(data_split[2]*dataset_size)

            train_dataset, val_dataset, test_dataset = random_split(dataset,[train_size, val_size, test_size],generator=torch.Generator())
        
            loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        with torch.no_grad():
            for xs, ys in loader:
                xs, ys = xs.to(device), ys.to(device)
                logits = self(xs)
                loss = loss_fn(logits, ys)
                total_loss += loss.item() * xs.size(0)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == ys).sum().item()
                total += ys.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"Test set  loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")
        self.train()  # torna in modalità training
    
        return (avg_loss,accuracy)
    
    def get_submodel(self, up_to_layer):
        
        submodel = MyResNet()
        submodel.layers = nn.ModuleList(self.layers[:up_to_layer])
        submodel._previous_is_conv = self._previous_is_conv
        submodel.data_shape = self.data_shape
        submodel._hidden_size = self._hidden_size
        submodel.device = self.device
        return(submodel)
        
    
    
    def show(self):
        for layer in self.layers:
            print(layer)

    
    def to(self,device):
        
        super().to(device)
        self.device = device
        
        return self
