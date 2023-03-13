import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13]
# 3 say 12 uthayeiga
y=dataset.iloc[:,13]

geography=pd.get_dummies(x['Geography'],drop_first=True)
gender=pd.get_dummies(x['Gender'],drop_first=True)

x=pd.concat([x,geography,gender],axis=1)
# concat kardou columns

x=x.drop(["Geography",'Gender'],axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
# scale down so magnitude less less computation power all feats in same scale range
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit(x_test)

#hyperparam optimization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
from keras.layers import Activation,Embedding,Flatten,LeakyReLU,BatchNormalization
from keras.activations import relu,sigmoid

def create_model(layers,activation):
    model=Sequential()
    for i,nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=x_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
    model.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'] )
    return model
model=KerasClassifier(build_fn=create_model,verbose=0)

#     0 hidden   1 hidden      2 hidden layers
layers=[[20],     [40,20],      [45,30,15]]
activations=['sigmoid','relu']
param_grid=dict(layers=layers,activation=activations,batch_size=[128,256],epochs=[30])
grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=5)

grid_result=grid.fit(x_train,y_train)
print(grid_result.best_score_,grid_result.best_params_)

pred_y=grid.predict(x_test)
y_predictions=(pred_y > 0.5)
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_predictions,y_test)
accuracy_score(y_predictions,y_test)














classifier=Sequential()

classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=11))

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'he_uniform',activation='relu',input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'he_uniform',activation='relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.compile(optimer='Adamax')
# Fitting the ANN to the Training set
model_history=classifier.fit(x_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 100)

# list all data in history

print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)


























