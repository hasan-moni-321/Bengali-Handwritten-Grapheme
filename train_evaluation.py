from tqdm import tqdm 

from sklearn.metrics import accuracy_score

import torch 
import torch.nn as nn 



def loss_fn(outputs, targets):
    op1, op2, op3 = outputs
    target1, target2, target3 = targets

    layer1 = nn.BCEWithLogitsLoss()(op1, target1)
    layer2 = nn.BCEWithLogitsLoss()(op2, target2)
    layer3 = nn.BCEWithLogitsLoss()(op3, target3) 
    return (layer1 + layer2 + layer3) / 3 




def train(dataset, data_loader, model, device, optimizer, scheduler):
    model.train()
    #train_accuracy = []
    #acc = 0
    #train_losses = []
    #train_loss = 0
    #counter = 0
    total = 0
    train_loss = 0
    correct = 0
    

    for bi, data in tqdm(enumerate(data_loader), total=int(len(dataset)/ data_loader.batch_size)):
        #counter += 1 
        image = data['image']
        grapheme_root = data['grapheme_root']
        vowel_diacritic = data['vowel_diacritic']
        consonant_diacritic = data['consonant_diacritic']

        image = image.to(device, dtype=torch.float)
        grapheme_root = grapheme_root.to(device, dtype=torch.float) 
        vowel_diacritic = vowel_diacritic.to(device, dtype=torch.float)
        consonant_diacritic = consonant_diacritic.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()
        #scheduler.step()

        ##########
        #outputs = torch.sigmoid(outputs)
        #outputs[outputs >= 0.5] = 1
        #accuracy = accuracy_score(targets, outputs) 
        #acc += accuracy
        ##########
        #train_loss += loss.item()
        
        
        train_loss += loss.item()
        _, predicted = outputs.max(1) # it will take only out. But model will give 3 output. correct it 
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item() 
        
        

    #train_loss = train_loss/ len(train_data.sampler)
    #train_losses.append(train_loss)
    
    train_acc = correct / total 
    train_loss = train_loss/total

    print("Epoch: {}  \tTraining Acc: {:.6f}  \tTraining Loss: {:.6f}".format(epoch+1, train_acc, train_loss)) 
    return train_acc, train_loss




def evaluation(dataset, data_loader, model, device):
    model.eval()
    #valid_accuracy = []
    #valid_losses = []
    #valid_loss = 0
    #acc = 0
    #counter = 0
    total = 0
    valid_loss = 0
    correct = 0
    

    for bi, data in tqdm(enumerate(data_loader), total=int(len(dataset)/ data_loader.batch_size)):
        #counter = counter + 1
        image = data['image']
        grapheme_root = data['grapheme_root']
        vowel_diacritic = data['vowel_diacritic']
        consonant_diacritic = data['consonant_diacritic']

        image = image.to(device, dtype=torch.float)
        grapheme_root = grapheme_root.to(device, dtype=torch.float) 
        vowel_diacritic = vowel_diacritic.to(device, dtype=torch.float)
        consonant_diacritic = consonant_diacritic.to(device, dtype=torch.float)

        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        ###############
        #outputs = torch.sigmoid(outputs)
        #outputs[outputs >= 0.5] = 1   
        #accuracy = accuracy_score(targets, outputs)
        #acc += accuracy
        ###############
        #valid_loss += loss.item()
        
        valid_loss += loss.item()
        _, predicted = outputs.max(1) # It will take only out output. But model will give 3 output, so you need to correction. 
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        

    #valid_acc = acc/ counter
    #valid_accuracy.append(valid_acc)
    
    valid_acc = correct/ total
    valid_loss = valid_loss / total


    print("Epoch: {} \tValidation Acc: {:.6f}  \tValidation Loss: {:.6f}".format(epoch+1, valid_acc, valid_loss))
    return valid_acc, valid_loss

