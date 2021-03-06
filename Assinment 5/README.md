**Team Members**

1:Saurabh Jain

2:Darshan Jani

3:Jaiveer Singh

4:Rushirajsinh Parmar

## Assignment

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


## **ANALYSIS**


|        | Target                                  | Parameters | Train Acc | Val Acc   |
| :----- | --------------------------------------- | ---------- | --------- | --------- |
| Approach 1 | Increased the dropout from 0.1 to 0.12  | 9872       | 98.64     | 99.32     |
| Approach 2 | Decreased the dropout from 0.12 to 0.05 | 9872       | 98.91     | 99.53     |
| Approach 3 | Decreased the dropout from 0.05 to 0.02 | **8016**   | **98.93** | **99.42** |



## Approach 1

**Target**

. Increased the dropout from 0.1 to 0.12

**Results**

. Parameters - 9872

. Best training accuracy- 98.64

. Best test accuracy - 99.32

**Analysis**

. Model is still under-fitting

. The accuracy of both is reduced by increasing dropout

## Approach 2

**Target**

. Decreased the dropout from 0.12 to 0.05

**Results**

. Parameters - 9872

. Best training accuracy- 98.91

. Best test accuracy - 99.53

**Analysis**

. Reached the desired accuracy at 8th epoch

. Should see how it works introducing LR and reducing no of parameters

## Approach 3

**Target**

. Decreased the dropout from 0.05 to 0.02

**Results**

Parameters - 8016

Best training accuracy- 98.93

Best test accuracy - 99.42

**Analysis**

. Reached the desired accuracy at 11th epoch with less parameters



![Curves](https://user-images.githubusercontent.com/34182074/120846778-3cbe5080-c590-11eb-99c5-4253a23e9859.png)




