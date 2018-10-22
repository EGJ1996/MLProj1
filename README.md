# MLProj1
I tried to implement a simple linear regression with the full matrix (all the columns) and the reduced one (without the correlated columns): I tested different polynomial degrees and checked the rmse error on the train and test set (5 folds cross validation).
--> maybe we can try with different degrees for each feature.
Next I want to try with gradient descent and SGD for the polynomial degree (=2) I think is best to avoid overfitting, and then with ridge regression (using cross validation and maybe a grid search to find the best combination lambda/polynomial degree), if it's not too time consuming.

I think that Emiljano can try fitting different models with all the methods according to the PRI_jet_num value (since he found different sets of significant features for each group) and then we can compare our two approaches.
Moreover in the beginning we can study a bit more the features, and try to do some plots about the correlations to explain our choices (maybe not all of them, as I tried in the code); Paul, could you do this part, please? And maybe try to implement a PCA to understand the real dimensions of the problem (I am thinking of a graph with the cumulated std of y and the percentage that is explained by n, n+1, n+2... features), if you want.
--> do you think it's better to keep the primitive features or the derived one, if they are correlated? 

Finally, if you want to do prettier plots and improve comments and explanations, feel completely free to do it, your help is welcomed!
