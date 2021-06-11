<h1>Problem statement</h1>
</br>You are making 3 versions of your 5th assignment's best model (or pick one from best assignments):
</br>        Network with Group Normalization
</br>        Network with Layer Normalization
</br>        Network with L1 + BN
</br>You MUST:
</br>        Write a single model.py file that includes GN/LN/BN and takes an argument to decide which normalization to include
</br>        Write a single notebook file to run all the 3 models above for 20 epochs each
</br>        Create these graphs:
</br>            Graph 1: Test/Validation Loss for all 3 models together
</br>            Graph 2: Test/Validation Accuracy for 3 models together
</br>            graphs must have proper annotation
</br>        Find 10 misclassified images for each of the 3 models, and show them as a 5x2 image matrix in 3 separately annotated images. 
</br>        write an explanatory README file that explains:
</br>            what is your code all about,
</br>            how to perform the 3 covered normalization (cannot use values from the excel sheet shared)
</br>            your findings for normalization techniques,
</br>            add all your graphs
</br>            your 3 collection-of-misclassified-images 
</br>        Upload your complete assignment on GitHub and share the link on LMS
</br>        Mention the name of your partners who are doing the assignment. If someone submits separately, then we will see who scored lowest, and we will assign the lowest to all. If there are NO partners, then mention there are NO parters and make sure NO one mentions your name. 

<h3> Approach</h3>
We have created a separate model.py which is responsible for the initialization of model. where we have to pass Type of normalization we want (BN/GN/LN) to use.</br>
Session_6_BatchNorm_GroupNorm_LayerNorm.ipynb file contains all model stats along with performance of each Normalization Technique.

<h2> Group Normalization Neural Architecture:- </h2>

![Screenshot from 2021-06-11 22-52-48](https://user-images.githubusercontent.com/74832766/121726158-ced1d600-cb07-11eb-9f8f-01bcd25704e8.png)

<h3> Miss Classified Images in Group Normalization</h3>

![Screenshot from 2021-06-11 23-15-00](https://user-images.githubusercontent.com/74832766/121728715-42291700-cb0b-11eb-9a93-35e8f117b382.png)

<h2> Batch Normalization  Architecture :- </h2>

![Screenshot from 2021-06-11 22-55-03](https://user-images.githubusercontent.com/74832766/121726556-54ee1c80-cb08-11eb-8c5a-f8508030284b.png)

<h3> Miss Classified Images in Batch Normalization</h3>

![Screenshot from 2021-06-11 23-15-21](https://user-images.githubusercontent.com/74832766/121728749-4d7c4280-cb0b-11eb-9957-59aa20e434a1.png)

<h2> Layer Normalization Architecture :-</h2>

![Screenshot from 2021-06-11 22-55-25](https://user-images.githubusercontent.com/74832766/121726602-6800ec80-cb08-11eb-8cba-c18af70e735e.png)

<h3> Miss Classified Images in Layer Normalization</h3>

![Screenshot from 2021-06-11 23-15-45](https://user-images.githubusercontent.com/74832766/121728776-540aba00-cb0b-11eb-83fa-d23112f5c0ad.png)


<h2>All Normalizatin technique used model performance comparison </h2>

![Screenshot from 2021-06-11 22-56-02](https://user-images.githubusercontent.com/74832766/121726683-849d2480-cb08-11eb-8eb0-d51f9b1c7da1.png)

![Screenshot from 2021-06-11 22-56-21](https://user-images.githubusercontent.com/74832766/121726702-89fa6f00-cb08-11eb-9b80-06804601fab1.png)


Team Members :- 
1) Darshan Jani
2) Rushiraj
3) Jaiveer
4) Saurabh Jain

