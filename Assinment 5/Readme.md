Team Members-

1:Saurabh Jain
2:Darshan Jani
3:Jaiveer Singh
4:Rushiraj

##Assignment:

1:Your new target is:
     
     1.99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
     
     2.Less than or equal to 15 Epochs
     
     3.Less than 10000 Parameters (additional points for doing this in less than 8000 pts

2:Do this in exactly 3 steps

3:Each File must have "target, result, analysis" TEXT block (either at the start or the end)

4:You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct. 

5:Evaluation is highly subjective, and if you target anything out of the order, marks will be deducted. 

6:Explain your 3 steps using these target, results, and analysis with links to your GitHub files (Colab files moved to GitHub). 

7:Keep Receptive field calculations handy for each of your models. 



## Code1 - Setup
**Target**

. Set up the skeleton with dropout, batch Norm etc

**Results** 

. Parameters - 8064 

. Best training accuracy- 98.13

. Best test accuracy - 99.17

**Analysis**

. Model is under-fitting

. The gap between test and train is high

. Capacity can be increased

## Code2
**Target**

. Added 1 more convolution layer and Gap Layer

. Changed the number of kernels in some conv layers

. Increased the capacity

**Results**

. Parameters - 9634

. Best training accuracy- 98.67

. Best test accuracy - 99.4


**Analysis**

. Target is reached, but only once.

. Model is still under-fitting


## Code 3

**Target**

. Increased the dropout from 0.1 to 0.12

**Results**

. Parameters - 9872

. Best training accuracy- 98.64

. Best test accuracy - 99.32

**Analysis**

. Model is still under-fitting

. The accuracy of both is reduced by increasing dropout

## Code 4

**Target**

. Decreased the dropout from 0.12 to 0.05

**Results**

. Parameters - 9872

. Best training accuracy- 98.91

. Best test accuracy - 99.53

**Analysis**

. Reached the desired accuracy at 8th epoch

. Should see how it works introducing LR and reducing no of parameters

## Code 5

**Target**

. Decreased the dropout from 0.05 to 0.02

**Results**

Parameters - 8016

Best training accuracy- 98.93

Best test accuracy - 99.42

**Analysis**

. Reached the desired accuracy at 11th epoch with less parameters
