#import data set 
dataSet=dlmread("backprop.txt");

#remove unwanted rows and columns
dataSet = dataSet(2:end,3:end)
#dataset(1:5,1:end)

#create vector with 8143 integers 
fullDataSet = [1:8143]

#get random 5700 integers from data set for the training data
trainingData = randperm(8143,5700)

#we need to get different intergers from the training data to test data. we use 'setdiff' function to do that. 
testData = setdiff(fullDataSet,trainingData)

#training data set 
for k = 1:size(trainingData,2)
  row = trainingData(k)
  if k == 1
    trainingDataSet = [dataSet(row,:)];
  else
    trainingDataSet = [trainingDataSet;dataSet(row,:)];
  endif
endfor  
#trainingDataSet(1:5,:)

#test data set
for k = 1:size(testData,2)
  row = testData(k)
  if k == 1
    testDataSet = [dataSet(row,:)];
  else
    testDataSet = [testDataSet;dataSet(row,:)];
  endif
endfor 
#testDataSet(1:5,:)

#create bias training data set 
bTraining = ones(1,size(trainingData,2));

#create bias test data set 
bTest = ones(1,size(testData,2));

#final training data set 
trainingData = [bTraining',trainingDataSet];
#trainingData(1:20,:)

#final test data set 
testDataSet = [bTest',testDataSet];
#testDataSet(1:20,:)

#weights
Th1 = 2 * rand(4,6)-1;
Th2 = 2 * rand(1,4)-1;

#learning rate
LR = 0.01;

#loops
L = 1000;

#define sigmoid function
function [result] = sigmoid(x)
  result = 1.0 ./(1.0+exp(-x));
endfunction  

#back propagation algorithm for online learning 
function [Th1,Th2,M2] = OL(trainingData,Th1,Th2,LR,L,M2=[])
  for i=1:L
    m=rows(trainingData);
    D=0;
    for i=1:rows(trainingData)
      
      P1 = [trainingData(i,1:6)'];
      R1 = Th1*P1;
      #call sigmoid function 
      P2 = sigmoid(R1);
      R2 = Th2*P2;
      #call sigmoid function
      Z = sigmoid(R2);     
      B = trainingData(i,7);
      error = B-Z;
      Q = ((1/2)*(B-Z)^2);
      D=D+Q;      
      F = error*Z*(1-Z);
      E = (((Th2')*F ).*(P2.*(1-P2)));      
      Th1 = Th1-(LR*(E*[P1']));
      Th2 = Th2-(LR*(F*[P2']));
      
    endfor
      D=D/m;
      M2=[M2,D];
  endfor
endfunction

#back propagation algorithm for batch learning 
function [Th1,Th2,M1] = BL(trainingData,Th1,Th2,LR,L,M1=[])
  for i=1:L
    
      m=rows(trainingData);
      T2_Delta=0;
      T1_Delta=0;      
      D=0;
      
    for i=1:rows(trainingData)
      
      P1 = [trainingData(i,1:6)'];
      R1 = Th1*P1;
      #call sigmoid function
      P2 = sigmoid(R1);
      R2 = Th2*P2;
      #call sigmoid function
      Z = sigmoid(R2);     
      B = trainingData(i,7);
      error = B-Z;
      Q = ((1/2)*(B-Z)^2);
      D=D+Q;     
      F = error*Z*(1-Z);
      E = (((Th2')*F).*(P2.*(1-P2)));   
      T2_Delta=T2_Delta+[F*P2'];
      T1_Delta=T1_Delta+[E*P1'];
             
    endfor
    
      D=D/m;
      M1=[M1,D];
      Th1 = Th1-(LR*(T1_Delta/m));
      Th2 = Th2-(LR*(T2_Delta/m));
      
  endfor
  
endfunction

#Get the accuracy of the two models from the test data
function test_function(testData,Th1,Th2)
  
  for i=1:rows(testData)
    
    P1 = [testData(i,1:6)'];
    R1 = Th1*P1;
    P2 = sigmoid(R1);
    R2 = Th2*P2;
    Z = sigmoid(R2);
    #check threshold value for classification
    if Z>0.5
      Z=1;
    else
      Z=0;      
    endif
    Z
    B = testData(i,7);
    disp("end 1 row")
  endfor
endfunction

x = [1:1000]
plot(x,M1)
hold on
plot(x,M2)



[T11,T22,M2] = OL(trainingData,Th1,Th2,LR,L)
[T11,T22,M1] = BL(trainingData,Th1,Th2,LR,L)
test_function(testData,T11,T22)





