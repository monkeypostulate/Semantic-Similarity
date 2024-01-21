import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.utils.data as data_utils

from tqdm import tqdm_notebook




def train_model(model,
                train_tensor,
                validation_tensor,
                n_epochs,
               loss_function,
               optimizer):
    # Initialize empty lists to store training and validation loss and accuracy
    train_loss = []
    validation_loss = []
    train_accuracy = []
    validation_accuracy = []

    # Calculate the total number of training and validation samples
    n_tra = len(train_tensor) * train_tensor.batch_size
    n_val = len(validation_tensor)*validation_tensor.batch_size

    # Iterate over epochs
    for epoch in tqdm_notebook(range(n_epochs)):  # loop over the dataset `N_EPOCHS` times
        model.train()  # Set the model to training mode
        # 1. Training
        epoch_loss = 0
        epoch_correct = 0
        for i, data in enumerate(train_tensor):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).detach().cpu().numpy()

        train_loss.append(epoch_loss / (i + 1))
        train_accuracy.append(epoch_correct / n_tra)

        model.eval()  # Set the model to evaluation mode

    # 2. Validation
        epoch_loss = 0
        epoch_correct = 0
        for i, data in enumerate(validation_tensor):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            with torch.no_grad(): # disable gradient calculation during evaluation
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            epoch_loss += loss.item()
            epoch_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).detach().cpu().numpy()

        validation_loss.append(epoch_loss / (i + 1))
        validation_accuracy.append(epoch_correct / n_val)
        
    return {'train_loss': train_loss,
                'validation_loss': validation_loss,
                'train_accuracy': train_accuracy,
                'validation_accuracy': validation_accuracy}






def train_test_data(business_data,
                    X,
                    y,
                    p):
    number_documents = X.shape[0]
    sample_size = int( np.round(number_documents*p) )
    id_train = np.random.choice(range(number_documents), sample_size, replace = False)
    id_test = set(range(number_documents))- set(id_train)
    id_test =  list(id_test)

    number_categories = len(np.unique(y))


    X_train =  X[id_train,:]
    y_train =  y[id_train]

    X_validation =  X[id_test,:]    
    y_validation =  y[id_test]

    batch_size = 50

    business_data_train = business_data.iloc[id_train]
    business_data_test = business_data.iloc[id_test]


    X_train =  torch.tensor(X_train).to()
    y_train =  torch.tensor(y_train).to()
    X_validation =  torch.tensor(X_validation).to()
    y_validation =  torch.tensor(y_validation).to()


    train_tensor = data_utils.TensorDataset(X_train, y_train)
    validation_tensor = data_utils.TensorDataset(X_validation, y_validation)


    train_tensor = data_utils.DataLoader(dataset = train_tensor,
                                   batch_size = batch_size,
                                   shuffle = True,
                                   drop_last = True)

    validation_tensor = data_utils.DataLoader(dataset = validation_tensor,
                                   batch_size = batch_size,
                                   shuffle = False,
                                   drop_last = False)
    
    return X_train, y_train, X_validation, y_validation, train_tensor, validation_tensor, id_train, id_test, business_data_train , business_data_test








class ClassificationModel(nn.Module):
    def __init__(self, inpud_dim = 768,
                 layer1_number_neurons = 100,
                 layer2_number_neurons = 50,
                output_dim = 2,
                activation_f = nn.functional.relu):
        super(ClassificationModel, self).__init__()

    # single layer model
        self.l1 = nn.Linear(inpud_dim, layer1_number_neurons)
        self.l2 = nn.Linear(layer1_number_neurons, layer2_number_neurons)
        self.l3 = nn.Linear(layer2_number_neurons, output_dim)
        self.activation_f = activation_f

    def forward(self, x):
        y = x

        # output - just logits. for optimized trainign - without activaiton here
        y = self.activation_f(self.l1(y))
        y = self.activation_f(self.l2(y))
        y = self.l3(y)  # returns logits

        return y
    
    
    

    
    

    
def train_classificaiton(inpud_dim,
                         number_categories,
                         learning_rate,
                         train_tensor,
                         validation_tensor,
                         n_epochs):
    
    model_classification = ClassificationModel(inpud_dim = inpud_dim,
                                 output_dim = number_categories)


    # Define the loss function (Cross Entropy Loss)
    loss_function = nn.CrossEntropyLoss()

    # Define the optimizer (Stochastic Gradient Descent)
    optimizer = optim.SGD(model_classification.parameters(),
                      lr = learning_rate,
                      momentum = 0.9)



    # ###########
    # Train Model
    output = train_model(model_classification,
                     train_tensor,
                     validation_tensor,
                     n_epochs,
                     loss_function,
                     optimizer)
    return model_classification, output






def predictions(model_classification, X_validation, y_validation, business_data_test):
    data = torch.tensor(X_validation)

    result = model_classification( data )


    prediction = result.argmax(axis = 1)
    prediction = prediction.numpy()
    y_validation = y_validation.numpy()


    business_data_test =  business_data_test.copy()

    business_data_test['prediction'] = prediction
    business_data_test['actual'] = y_validation
    business_data_test['correct_prediction'] = prediction == y_validation
    return business_data_test