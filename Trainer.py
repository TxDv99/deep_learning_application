import numpy as np
import matplotlib.pyplot as plt
import torch   
import torchvision.transforms as transforms
from torch.utils.data import random_split 
from torch.utils.data import ConcatDataset
from torch import nn
from types import SimpleNamespace
import random
from tqdm import tqdm
import warnings
import time
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import copy
from datetime import datetime
import wandb


class Trainer:
    @staticmethod
    def get_config_keys():
        
        return ("device", "num_workers", "seed")

    def __init__(self,config, model, dataset, lr,optimizer = None, loss_fn = torch.nn.CrossEntropyLoss()):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.history = None
        self.lr = lr
        self.validation_params = {"early stop times": 0, "check on val times": 0}

        config = SimpleNamespace(**self.config)
        
        if optimizer == None:
           self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        else:
           self.optimizer = optimizer
        
        self.dataset = dataset
        
        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
            

        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        

    def train(self, data_split, batch_size, num_epochs,iter_time = None, early_stopping = 5, val_check = 3, checks = True, use_wandb = True, project_name = None, augmentation = None, fgsm = None):
        
        self.config = SimpleNamespace(**self.config)
        #self.model.device = self.config.device
    
        #clean history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_epoch": []
            }
        self.validation_params = {"early stop times": early_stopping, "check on val times": val_check}
        
        #setup wandb project
        if use_wandb:
            if project_name == None:

               now = datetime.now()
               wandb.init(
                    project= now.strftime("%Y%m%d%H%M%S"),        
                    config={
                            "learning_rate": self.lr,
                            "batch_size": batch_size,
                            "epochs": num_epochs
                    }
                )
            else:
                wandb.init(
                    project=project_name,
                    config={
                            "learning_rate": self.lr,
                            "batch_size": batch_size,
                            "epochs": num_epochs
                    }  
                )

        #set random seeds
        seed = self.config.seed
        random.seed(seed)                     # seed per il modulo random di Python
        np.random.seed(seed)                  # seed per NumPy
        torch.manual_seed(seed)               # seed per PyTorch (CPU)
        torch.cuda.manual_seed(seed)          # seed per PyTorch (GPU)
        torch.cuda.manual_seed_all(seed)      # se usi multi-GPU
        
        #perform the dataset splitting and eventual augmentation
        dataset_size = len(self.dataset)
        val_size = int(data_split[0] * dataset_size)
        train_size = int(data_split[1] * dataset_size)
        test_size = dataset_size - val_size - train_size if len(data_split) == 2 else int(data_split[2]*dataset_size)

        train_dataset, val_dataset, test_dataset = random_split(self.dataset,[train_size, val_size, test_size],generator=torch.Generator())
        
        if augmentation != None:
            train_fraction = augmentation[0]
            aug_transform = augmentation[1]
            
            indices = random.sample(range(len(train_dataset)), int(train_fraction*len(train_dataset)))
            
            transformed_train = copy.copy(train_dataset)
            transformed_train.transform = aug_transform
            
            transformed_subset = Subset(transformed_train, indices)
            
            train_dataset = ConcatDataset([transformed_subset, train_dataset])

            


        # setup the dataloaders
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.config.num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.config.num_workers,
        )


        #initial checks
        if checks:
            self._initial_checks(train_loader, train_dataset)
        
        
        #training
        start_time = time.time()
        val_loss_history = []
        for epoch in range(num_epochs):
            self.model.train()  
            running_train_loss = 0.0
            running_val_loss = 0.0
        
            for (xs, ys) in tqdm(train_loader, desc=f'Training epoch {epoch}', leave=True):
                xs = xs.to(self.device)
                ys = ys.to(self.device)

                
                if fgsm != None:
                    
                    budget = fgsm[0]
                    if len(fgsm) == 1:
                        y_target = None
                    elif len(fgsm) == 2:
                        y_target = fgsm[1]
                       
                    augmented_list = []     
                    model_copy = copy.deepcopy(self.model).to(self.device)
                    
                    for index in range(xs.size()[0]):
                        
                        x = xs[index,:,:,:].unsqueeze(0)
                        y_true = ys[index].unsqueeze(0)
                        x_adv = FGSM(
                                    x=x,
                                    y_true=y_true,
                                    model=model_copy,
                                    budget=budget,
                                    y_target=y_target,
                                    loss_fun=self.loss_fn
                                    )

                        augmented_list.append(x_adv.squeeze(0))

                    xs = torch.stack(augmented_list, dim=0)
                    del model_copy
                    

                self.optimizer.zero_grad()    
                logits = self.model(xs)             
                loss = self.loss_fn(logits, ys)     
                loss.backward()               
                self.optimizer.step()         
                running_train_loss += loss.detach().item()
            
            avg_train_loss = running_train_loss / len(train_loader)
            print(f"Epoch {epoch}/{num_epochs}, Average training loss: {avg_train_loss:.4f}")
            train_acc = self._get_accuracy(train_loader)
            print(f"Epoch {epoch}/{num_epochs}, Training accuracy: {train_acc:.4f}")

            #performance su validation each val_check epoch
            if epoch % val_check == 0:

                for (xs,ys) in tqdm(val_loader, desc=f'Validation epoch {epoch}', leave=True):
                    xs = xs.to(self.device)
                    ys = ys.to(self.device)
    
                    self.optimizer.zero_grad()    
                    logits = self.model(xs)             
                    loss = self.loss_fn(logits, ys)    
                    running_val_loss += loss.detach().item()
                
                avg_val_loss = running_val_loss/ len(val_loader)
                val_loss_history.append(avg_val_loss)
                print(f"Epoch {epoch}/{num_epochs}, Average validation loss: {avg_val_loss:.4f}")
                val_acc  = self._get_accuracy(val_loader)
                print(f"Epoch {epoch}/{num_epochs}, Validation accuracy: {val_acc:.4f}")
                self.history["val_acc"].append(val_acc)
                self.history["val_loss"].append(avg_val_loss)
                self.history["val_epoch"].append(epoch)

                #check if other stop criteria are met
                if len(val_loss_history) >= early_stopping + 1:
                        recent_losses = val_loss_history[-(early_stopping + 1):-1]
                        no_improvement = all(val_loss_history[-1] >= loss for loss in recent_losses)
                        if no_improvement:
                            print("Early stopping triggered!")
                            if use_wandb:
                                wandb.finish()
                            return 
    
                # dopo ogni epoca
            self.history["train_loss"].append(avg_train_loss)
            self.history["train_acc"].append(train_acc)
            
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "training set accuracy": train_acc,
                    "validation set accuracy": val_acc
                })

            #check if other stop criteria are met
            if iter_time is not None:
                if time.time() - start_time > iter_time:
                    print("Training has reached it's time limit")
                    if use_wandb:
                       wandb.finish()
                    return 
                
            
        if use_wandb:
            wandb.finish()
        return 
    
    
    def fine_tune(self, layers_to_unfreeze, data_split, batch_size, num_epochs,iter_time = None, early_stopping = 5, val_check = 3, checks = True, use_wandb = True, project_name = None, augmentation = None, fgsm = None):
        
        for param in self.model.parameters():
            param.requires_grad = False

        for index, layer in enumerate(self.model.layers):
            if index in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
        
        print(f"[Fine-tune] Unfreezing layer indices: {layers_to_unfreeze}")

        self.train(data_split, batch_size, num_epochs,iter_time, early_stopping , val_check , checks , use_wandb, project_name,augmentation,fgsm)

        return



    

    def _initial_checks(self, train_loader, train_dataset):
        #instanziate model and optimizer
        device = self.device
        model_test = copy.deepcopy(self.model)
        model_test = model_test.to(device)


        optimizer_test = torch.optim.Adam(model_test.parameters(), lr=1e-2)

        (xs,ys) = next(iter(train_loader))
        xs, ys = xs.to(device), ys.to(device)
        logits = model_test(xs)
        mean_logits = logits.mean(dim = 0)
        diff = mean_logits.max() - mean_logits.min()
        if diff > 0.3:
            warnings.warn("Not uniform initial logits distribution!")
            
        small_train_dataset = Subset(train_dataset, list(range(int(len(train_dataset)*0.05)))) #take 5% of the trainingset
        small_loader = DataLoader(small_train_dataset, batch_size=4, shuffle=True)
            
    
        for epoch in range(300):
            for xs, ys in small_loader:
                xs, ys = xs.to(device), ys.to(device)

                optimizer_test.zero_grad()
                logits = model_test(xs)
                loss = self.loss_fn(logits, ys)
                loss.backward()
                optimizer_test.step()
        if loss.item() > 0.05:
            warnings.warn("Couldn't overfit on small dataset!")
        if diff < 0.3 and loss.item() < 0.5:
            print("Initial checks passed succesfully")

        return
    
    def _get_accuracy(self, data_loader):

        self.model.eval()  
        correct = 0
        total = 0
        
        with torch.no_grad():  
            for xs, ys in data_loader:
                xs, ys = xs.to(self.device), ys.to(self.device)
                outputs = self.model(xs)  
                preds = torch.argmax(outputs, dim=1)  

                correct += (preds == ys).sum().item() 
                total += ys.size(0)

        accuracy = correct / total
        
        self.model.train()
        
        return accuracy
    
    
    def plot_curves(self):

        epochs = [i[0] for i  in enumerate(self.history['train_loss'])]
        val_epochs = self.history["val_epoch"]

        plt.figure(figsize=(16, 8))

        plt.subplot(1, 2, 1)
        plt.plot(epochs,self.history['train_loss'], color="red", marker="o", linestyle='-', label='Training Loss (batch-wise avg)')
        plt.plot(val_epochs,self.history['val_loss'], color="green", marker="o", linestyle='--', label='Validation Loss (epoch)')
        plt.xticks(range(len(epochs)), [str(x) for x in epochs])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Average Loss per Epoch')
        plt.legend() 
    
        plt.subplot(1, 2, 2)
        plt.plot(epochs,self.history['train_acc'], color="red", marker="o", linestyle='-', label='Training Accuracy (batch-wise avg)')
        plt.plot(val_epochs,self.history['val_acc'], color="green", marker="o", linestyle='--', label='Validation Accuracy (epoch)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Epoch')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
        return
    

    def get_summary(self):

        loss_train = self.history['train_loss']
        loss_val = self.history['val_loss']
        accuracy_train = self.history['train_acc']
        accuracy_val = self.history['val_acc']

        report = {"training loss": loss_train[-1], "validation loss": loss_val[-1], "training accuracy": accuracy_train[:-1], "validation accuracy": accuracy_val[-1]}

        return(report)
        
        



def FGSM(x, model, budget=0.1,y_true = None, y_target=None, loss_fun=nn.CrossEntropyLoss()):
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    output = model(x)

    classes = list(range(output.size(1)))

    if y_target is not None:
        if y_target not in classes:
            raise ValueError(f"Target label out of range! Valid: {classes}")
        elif y_target == y_true:
            raise ValueError(f"Target label and true label are the same!")
        y = torch.tensor([y_target], device=x.device)
    else:
        y = y_true if torch.is_tensor(y_true) else torch.tensor([y_true], device=x.device)

    loss = loss_fun(output, y)

    model.zero_grad()
    loss.backward()

    perturbation = budget * torch.sign(x.grad)

    if y_target is not None:
        xadv = x - perturbation 
    else:
        xadv = x + perturbation  

    xadv = torch.clamp(xadv, -1, 1).detach()
    return xadv
    
