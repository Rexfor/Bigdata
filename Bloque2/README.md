## Introduction
Machine learning or machine learning in Spanish is a branch of data science / artificial intelligence that is used to generate systems that learn by themselves, that is, they are programmed to learn automatically, this is done by identifying patterns in millions of data that can process, with this it is expected that the created system can predict future behaviors in millions of data, at present machine learning has been in trend together with big data in large companies since they can provide tools that are very useful to these companies, although we could say that machine learning is something very old because since the 50s people have looked for a way to generate intelligent systems that could perform tasks autonomously, for example in 1952 where a system was generated that I could play Chinese checkers and I was getting better game after game, but it wasn't until the 1980s that e systems were generated with rules to discard data from a data group, these being well received by corporations being a milestone in 1981 Gerald Dejong introduces the concept of “Explanation Based Learning”, where the system analyzes training data and generates rules that allow you to discard unnecessary data; for a time machine learning was stagnant until in 2006 a great interest in it was generated, given that the computing power in the hardware was increasing and with the great abundance of data that there was machine learning would bring again to the game, with the great advances in the power of computers, machine learning was gaining a lot of ground in the industries, having giants that use it to analyze large amounts of data such as Google that bought a deep learning startup or its own development with GoogleBrain to autonomously analyze YouTube videos and detect those that contain cats.
In this way we can know that machine learning is essential for our daily lives, since with this we can have recommender systems like those of Netflix that, based on our preferences, can recommend movies or series that may interest us, according to this, in machine learning there are classifier and regression algorithms; This document will seek to delve into the different algorithms in the world of machine learning, mainly the classification and regression algorithms together with their scala code using the spark framework, but before we begin, we must know what we mean by each algorithm type.
The classification algorithms try to identify and classify objects on a set of predefined classes, for example, classify if a news is sports, entertainment, politics, etc. If only 2 possible classes are allowed, then it is called binary classification; but if more than 2 classes are allowed, we are talking about multiclass classification, this type of algorithms is in the supervised learning category since they learn based on the relationships that associate the input values ​​with the information output values ​​according to the type of data output you can have the category to which the algorithm belongs, it can be classification or regression where we would find out if it is regression because the output value would be continuous data, thus being a line with the prediction.

## Classification algorithms

The classification algorithms is a concept that is often used in the world of machine learning, which, as we will see, is used to predict a category in which an object can be classified based on the input data, for example we can use this type algorithms to predict if an email is "spam" or is a "real", in this type of algorithms there are two main classifications. we have the:

## Binary classification.
  - In this case we can use the example given above, with the mail that must be categorized as "spam" or "true" mail, since we only have 2 final data taking as 0 or 1 any of the two states in which the mail, if an email enters and must be classified in any of these 2 classes, if it is "spam" it would be "0" and if it is "true" it would be "1" taking into account the characteristics of the new mail that arrives.

## Multinomial Classification.
  - In this case, the multinomial type is one that has input data but has a large number of output values, an example is the classification of a set of images, taking into account that we are asked to classify an image in categories such as:
    - Animals.
    - People.
    - Plants.

  - If it only had two classes, it would be a binary classification.

Next, some algorithms and their explanation in code using scala will be explained in a detailed way with the help of the spark framework and with the help of the MLlib library which is used in scala to handle Machine Learning in a simple way, which gives us many tool for the use of machine learning, with a few simple lines of code we can observe in more detail the operation of some algorithms of the classification type.












## Logistic regression
Logistic regression is a statistical tool developed in the 60's, it is a simple but very useful algorithm and it can be adapted for a multiclass classification, the basis of regression systems is to predict qualitative characteristics of an object based on The data that we could have, which must be known variables, these can be qualitative or qualitative values that must act as dependent variables. An example of this could be classifying the sex of a black widow by her size.

![logistica](https://user-images.githubusercontent.com/60914099/104147363-cde31f80-5382-11eb-8e80-0a8acea3be7a.PNG)
### Logistic regression

The logistic regression is divided into 2 types:

- Binary logistic regression.
This makes use of known variables, whether qualitative or quantitative, to act as the variables that will be used to determine the characteristics that we need to predict a third variable that can only have an answer within parameters 0 or 1.
- Multinomial logistic regression.
Multinomial regression also makes use of qualitative or quantitative variables to predict data, but in this case we not only have 2 classes 0 or 1, in this case we can have more of these classes.

Through logistic regression, the aim is to obtain the probability that an event occurs based on variables that we determine are important to predict the new one, these must be relevant to be able, for example, if we have a human being and we want to classify it in a biological gender, We can do it if we have information about the reproductive system, its complexion, etc; with these data we could predict their sex.




## Binomial logistic regression

The binary logistic regression algorithm is a subcategory of the logistic regression algorithms, it refers to binary because it is one in which you can only have two final values, 0 or 1, false or true, its use can be given when we are interested in knowing the influence of a set of variables to predict an output variable, its use is very useful for when we want to know a third variable.

If we have a variable in which we only have two possible events in which a person lives or dies, these will be our dependent variables and we want to know the effect of other independent variables such as age or smoking, the logistic regression model will help us to calculate the next:

- Given the values of the independent variables, estimate the probability that the event of interest occurs (for example, becoming ill).
- We can evaluate the influence that each independent variable has on the response, in the form of OR.
- An OR greater than one indicates an increase in the probability of the event and an OR less than one indicates a decrease.

But to build a binary logistic regression model we need:
- A set of independent variables.
- A response variable that only has two values 0 or 1. Here it differs from the multiple regression model, where the response variable is numeric.

Steps to follow to encode the variables:

- The dependent variable must be coded as 1, which will indicate that the event can happen and its absence as 0.
- The independent variables can be several, the following being the most used:
    -Dichotomic:
      -The case that is believed favors the occurrence of the event is coded as 1. The opposite case is encoded as 0.
    -Categorical:
      -When the independent variable can take more than two possible values, unlike the dichotomous variable, we can code them using indicator variables.
    -Numerical
      -It is to make use of cutoff values that suit us to be able to create categories according to the chosen percentiles.

In order to obtain a correct binary logistic regression model, we must take into account that the parameters must be congruent with each other, therefore we must have a sufficient amount of observation data for each variable independently, likewise, unnecessary data must not be had in the observations we must correctly identify all the relevant variables for the model.

This section will explain how the code of the binomial logistic implementation works in scala, using spark, in the following code the steps are performed to perform this regression.

```
// We must load the class to use the function
import org.apache.spark.ml.classification.LogisticRegression

// We load the data into the training model, using a training variable for storage, the information is loaded from the address set.
val training =spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// The function will be loaded into a variable, here the information on the number of iterations that must be given and the regularization of parameters is placed, likewise the linear combination must be regularized
val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

// We fit the model to establish the training data.
val lrModel = lr.fit(training)

// The coefficients of the training model are printed.
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
// Next, the training model must be called
val trainingSummary = lrModel.binarySummary
// We must obtain the objectives by iteration, in this part a foreach is used to perform the count.
val objectiveHistory = trainingSummary.objectiveHistory
println("objectiveHistory:")
objectiveHistory.foreach(loss => println(loss))

// We are going to obtain the area under the ROC, with this we refer to the section of the training data frame that interests us and we are going to print it.
val roc = trainingSummary.roc
roc.show()
println(s"areaUnderROC: ${trainingSummary.areaUnderROC}")

// we are going to establish the maximum area of the model for its implementation from the data of the training data frame to verify its level of precision from the data of the header.
val fMeasure = trainingSummary.fMeasureByThreshold
val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
  .select("threshold").head().getDouble(0)
lrModel.setThreshold(bestThreshold)
```


/*
Output:
+---+--------------------+
|FPR| TPR|
+---+--------------------+
|0.0| 0.0|
|0.0|0.017543859649122806|
|0.0| 0.03508771929824561|
|0.0| 0.05263157894736842|
|0.0| 0.07017543859649122|
|0.0| 0.08771929824561403|
|0.0| 0.10526315789473684|
|0.0| 0.12280701754385964|
|0.0| 0.14035087719298245|
|0.0| 0.15789473684210525|
|0.0| 0.17543859649122806|
|0.0| 0.19298245614035087|
|0.0| 0.21052631578947367|
|0.0| 0.22807017543859648|
|0.0| 0.24561403508771928|
|0.0| 0.2631578947368421|
|0.0| 0.2807017543859649|
|0.0| 0.2982456140350877|
|0.0| 0.3157894736842105|
|0.0| 0.3333333333333333|
+---+--------------------+
only showing top 20 rows
areaUnderROC: 1.0



## Multinomial logistic regression
The multinomial logistic regression algorithm is used in models with dependent variables of the numerical type that can be divided into more than two categories, while the binary can only be done in 2 categories, being polyatomic, it is an extension of the binary version of logistic regression , the variables can be predictive with which we will help us to predict the result.
Throughout the years, polycotomic variables have been modeled by discriminant analysis to determine if a variable using labels belongs to a category.
It is based on the same principles as simple logistic regression but expanding the number of predictors. Predictors can be both continuous and categorical.
When evaluating the validity and quality of a multiple logistic regression model, both the model as a whole and the predictors that comprise it are analyzed. The model is considered useful if it is capable of showing an improvement over the null model, the model without predictors. There are 3 statistical tests that quantify this improvement by comparing the residuals: likelihood ratio, score and Wald test. There is no guarantee that the 3 will reach the same conclusion, when this occurs it seems to be advisable to rely on the likelihood ratio.
This is useful for situations in which you want to be able to classify subjects based on the values ​​of a set of predictor variables.
Some considerations to take into account when performing this type of regression are:

- Data.
  - The dependent variable must be categorical. The independent variables can be factors or covariates. In general, the factors must be categorical variables and the covariates must be continuous variables.
- Assumptions.
  - The likelihood ratio of any pair of categories is assumed to be independent of the other response categories. Under this assumption, for example, if a new product is introduced into a market, the market shares of all other products will be affected equally proportionally. Similarly, given a pattern in the covariates, the responses are assumed to be independent multinomial variables.

This section will explain how the multinomial logistics implementation code works in scala, using spark. The following code performs the steps to perform this regression.

```
// We must load the library that will help us with this classification algorithm
import org.apache.spark.ml.classification.LogisticRegression

// We must load the data for the training model to later use it in the model.
val training = spark
  .read
  .format("libsvm")
  .load("data/mllib/sample_multiclass_classification_data.txt")

// The function will be loaded into a variable, here the information on the number of iterations to be given and the regularization of parameters is placed, likewise the linear combination must be regularized
val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

// We must fit the data in the training model
val lrModel = lr.fit(training)

// We print the coefficients that will serve as covariates that represent the change in the function by the predictor.
println(s"Coefficients: \n${lrModel.coefficientMatrix}")
println(s"Intercepts: \n${lrModel.interceptVector}")

// of the model we assign a variable where a summary of data for the training model will be inserted.
val trainingSummary = lrModel.summary

// We must obtain the objectives by iteration, in this part a foreach is used to perform the count.
val objectiveHistory = trainingSummary.objectiveHistory
println("objectiveHistory:")
objectiveHistory.foreach(println)

// In this section we are going to print the metrics of our interest using a for each, this will be shown at the end.
println("False positive rate by label:")
trainingSummary.falsePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
  println(s"label $label: $rate")
}

println("True positive rate by label:")
trainingSummary.truePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
  println(s"label $label: $rate")
}

println("Precision by label:")
trainingSummary.precisionByLabel.zipWithIndex.foreach { case (prec, label) =>
  println(s"label $label: $prec")
}

println("Recall by label:")
trainingSummary.recallByLabel.zipWithIndex.foreach { case (rec, label) =>
  println(s"label $label: $rec")
}


println("F-measure by label:")
trainingSummary.fMeasureByLabel.zipWithIndex.foreach { case (f, label) =>
  println(s"label $label: $f")
}
//These are the precision data, the false positive ratio to determine the values, as well as the positive to determine the recall values
val accuracy = trainingSummary.accuracy
val falsePositiveRate = trainingSummary.weightedFalsePositiveRate
val truePositiveRate = trainingSummary.weightedTruePositiveRate
val fMeasure = trainingSummary.weightedFMeasure
val precision = trainingSummary.weightedPrecision
val recall = trainingSummary.weightedRecall
println(s"Accuracy: $accuracy\nFPR: $falsePositiveRate\nTPR: $truePositiveRate\n" +
  s"F-measure: $fMeasure\nPrecision: $precision\nRecall: $recall")
```

/*
Output:
Coefficients:
3 x 4 CSCMatrix
(1,2) -0.7803943459681859
(0,3) 0.3176483191238039
(1,3) -0.3769611423403096
Intercepts:
[0.05165231659832854,-0.12391224990853622,0.07225993331020768]
objectiveHistory:
1.098612288668108
1.087602085441699
1.0341156572156232
1.0289859520256006
1.0300389657358995
1.0239965158223991
1.0236097451839508
1.0231082121970012
1.023022220302788
1.0230018151780262
1.0229963739557606
False positive rate by label:
label 0: 0.22
label 1: 0.05
label 2: 0.0
True positive rate by label:
label 0: 1.0
label 1: 1.0
label 2: 0.46
Precision by label:
label 0: 0.6944444444444444
label 1: 0.9090909090909091
label 2: 1.0
Recall by label:
label 0: 1.0
label 1: 1.0
label 2: 0.46
F-measure by label:
label 0: 0.819672131147541
label 1: 0.9523809523809523
label 2: 0.6301369863013699
Accuracy: 0.82
FPR: 0.09
TPR: 0.82
F-measure: 0.8007300232766211
Precision: 0.8678451178451179
Recall: 0.82


## Decision tree classifier
Tree-based learning algorithms are considered one of the best and most widely used supervised learning methods. Tree-based methods empower predictive models with high precision, stability, and ease of interpretation.
Unlike linear models, they map nonlinear relationships quite well. They are adaptable to solve any type of problem (classification or regression).

Methods like decision trees, random forest, gradient augmentation are popularly used in all kinds of data science problems.
Decision or classification trees are algorithms for classifying using successive partitions. They are appropriate when there is a large number of data, one of their advantages being their descriptive nature that allows to easily understand and interpret the decisions made by the model, revealing complex forms in the data structure that cannot be detected with conventional regression methods.

The decision trees contain the following components:
- Nodes.
  - The nodes are the input variables
- Branches
  - The branches represent the possible values ​​of the input variables
- Leaves.
  - The leaves are the possible values ​​of the output variable.

As the first element of a decision tree, we have the root node that is going to represent the most relevant variable in the classification process. All decision tree learning algorithms obtain models that are more or less complex and consistent with respect to the evidence, but if the data contains inconsistencies, the model will adjust to these inconsistencies and harm its overall behavior in prediction. known as overfitting. To solve this problem, it is necessary to limit the growth of the tree by modifying the learning algorithms to achieve more general models. This is what is known as decision tree pruning.
Taking into account the above we can see the following disadvantages:

- The overfitting.
- Loss of information when categorizing continuous variables.
- Its level of precision compared to other methods such as SVM tends to have an error rate of less than 30% approximately.
- Its instability to small changes in the data can modify the entire structure of the tree.
But just as it has disadvantages, it also has many advantages that help us with our problems.
- It's easy to understand.
- It is useful in exploring data.
- The data type does not represent a constraint.

This section will explain how the decision tree implementation code works in scala, using spark. The following code performs the steps to perform this regression.

```
// We must load the following libraries that will help us with this algorithm of // classification
// The pipeline library helps us with the estimation of the data, while the decision
// libraries help us with the model and its evaluation, while the feature library helps us
// with the extraction and transformation of characteristics in the data.
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// We load the data for the model and it becomes a dataframe.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// We must assign an index to the labels to convert the string into a column with its respective index.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)

// We must assign an index to convert the created columns into a vector.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
// In this part we assign a category limit to avoid the overfitting error
 .setMaxCategories(4)   
.fit(data)

// We divide the data into the training and test sets leaving only 30% of the data for testing
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// The model for the classification tree will be created with the test data.
val dt = new DecisionTreeClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")

// We return the indexed values to the original values for use within the model.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labelsArray(0))
// With the help of Pipeline the estimator will be called in the input data set to fit a model, this through the indices and thus have the metadata that we require for the operation of the decision tree.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// With this part of the code, the model is "trained" with the indices that were previously taken within the training data.
val model = pipeline.fit(trainingData)

// Once we train the model we can perform the tests with the data from the test set and make the predictions that will be saved in a variable predictions
val predictions = model.transform(testData)

// We print the rows of the predictions variable, in this case we only select 5.
predictions.select("predictedLabel", "label", "features").show(5)

// With the help of the evaluator library, we are going to select data from the prediction columns that have an index and we are going to calculate the error in the test data according to the prediction.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
```

/*
Output:
+--------------+-----+--------------------+
|predictedLabel|label| features|
+--------------+-----+--------------------+
| 0.0| 0.0|(692,[121,122,123...|
| 0.0| 0.0|(692,[123,124,125...|
| 0.0| 0.0|(692,[124,125,126...|
| 0.0| 0.0|(692,[124,125,126...|
| 0.0| 0.0|(692,[125,126,127...|
+--------------+-----+--------------------+
only showing top 5 rows
Test Error = 0.030303030303030276
Learned classification tree model:
DecisionTreeClassificationModel (uid=dtc_a286075ebc4c) of depth 2 with 5 nodes
If (feature 406 <= 22.0)
If (feature 99 in {2.0})
Predict: 0.0
Else (feature 99 not in {2.0})
Predict: 1.0
Else (feature 406 > 22.0)
Predict: 0.0


## Random forest classifier
The Random Forest is a supervised learning algorithm. The "forest" that is built is a set of decision trees, generally trained with the packaging method. The general idea of ​​the bagging method is that a combination of learning models increases the overall result.
Bottom line: Random Forest creates multiple decision trees and merges them together to get a more accurate and stable prediction.
A great advantage of the random forest is that it can be used for both
classification as regression, which make up most of today's machine learning systems. Let's look at the random forest in classification, as classification is sometimes considered the building block of machine learning.
One of the biggest advantages of the random forest is its versatility. It can be used for classification and regression tasks, and it's also easy to see the relative importance you assign to input characteristics.
The random forest is also a very useful algorithm because the hyperparameters
Often used defaults produce a good prediction result. Understanding hyperparameters is pretty straightforward, and there aren't many of them either.
One of the biggest problems with machine learning is overfitting, but most of the time this won't happen thanks to the random forest classifier. If there are enough trees in the forest, the classifier will not fit the model too closely.

The main limitation of the random forest is that a large number of trees can make the algorithm too slow and ineffective for real-time predictions. In general, these algorithms are quick to train, but quite slow to create predictions once trained. A more accurate prediction requires more trees, which results in a slower model. In most real-world applications, the random forest algorithm is fast enough, but there certainly can be situations where runtime performance is important and other approaches are preferred.

And of course, the random forest is a predictive modeling tool and not a
descriptive tool, which means if you are looking for a description of the relationships in your data, other approaches would be better.

The way the random forest works is as follows:
- We select k features (columns) from the total m (where k is less than m) and we create a decision tree with those k features.
- We create n trees always varying the amount of k features and we could also vary the number of samples that we pass to those trees (this is known as a “bootstrap sample”)
- We take each of the n trees and ask them to make the same classification. We save the result of each tree obtaining n outputs.
- We calculate the votes obtained for each selected “class” and we will consider the one with the most votes as the final classification of our “forest”.

This section will explain how the decision tree implementation code works in scala, using spark. The following code performs the steps to perform this regression.

```
// We must load the following libraries that will help us with this random forest algorithm. In this case, as the algorithm is a more complex variant of the decision tree algorithm, the same libraries will be used as with the tree, but adding the decision tree model library.
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// We must load the data and convert it into a dataframe for use.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// We must assign an index to the labels to convert the string into a column with its respective index, with this we add the metadata for the operation of the algorithm.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)

// We must assign an index to convert the created columns into a vector.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
// In this part we assign a category limit to avoid the overfitting error
  .setMaxCategories(4)
  .fit(data)

// We divide the data into the training and test sets leaving only 30% of the data for testing
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// The model for the classification tree will be created with the test data.
val rf = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(10)

// We return the indexed values to the original values for use within the model.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// With the help of Pipeline, the estimator will be called in the input data set to fit a model, this through the indices and thus have the metadata that we require for the operation of the algorithm.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// With this part of the code, the model is "trained" with the indices that were previously taken within the training data.
val model = pipeline.fit(trainingData)

// Once we train the model we can perform the tests with the data from the test set and make the predictions that will be saved in a variable predictions
val predictions = model.transform(testData)

// We print the rows of the predictions variable, in this case we only select 5.
predictions.select("predictedLabel", "label", "features").show(5)

// With the help of the evaluator library, we are going to select data from the prediction columns that have an index and we are going to calculate the error in the test data according to the prediction.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println("Learned classification forest model:\n" + rfModel.toDebugString)
```

/*
Output:
+--------------+-----+--------------------+
|predictedLabel|label| features|
+--------------+-----+--------------------+
| 0.0| 0.0|(692,[95,96,97,12...|
| 0.0| 0.0|(692,[98,99,100,1...|
| 0.0| 0.0|(692,[123,124,125...|
| 0.0| 0.0|(692,[124,125,126...|
| 0.0| 0.0|(692,[124,125,126...|
+--------------+-----+--------------------+
only showing top 5 rows
Test Error = 0.0
Learned classification forest model:
RandomForestClassificationModel (uid=rfc_b10be44ab730) with 10 trees
Tree 0 (weight 1.0):
If (feature 510 <= 2.5)
If (feature 495 <= 111.0)
Predict: 0.0
Else (feature 495 > 111.0)
Predict: 1.0
Else (feature 510 > 2.5)
Predict: 1.0
Tree 1 (weight 1.0):
If (feature 567 <= 8.0)
If (feature 456 <= 31.5)
If (feature 373 <= 11.5)
Predict: 0.0
Else (feature 373 > 11.5)
Predict: 1.0
Else (feature 456 > 31.5)
Predict: 1.0
Else (feature 567 > 8.0)
If (feature 317 <= 9.5)
Predict: 0.0
Else (feature 317 > 9.5)
If (feature 491 <= 49.5)
Predict: 1.0
Else (feature 491 > 49.5)
Predict: 0.0
Tree 2 (weight 1.0):
If (feature 540 <= 87.0)
If (feature 576 <= 221.5)
Predict: 0.0
Else (feature 576 > 221.5)
If (feature 490 <= 15.5)
Predict: 1.0
Else (feature 490 > 15.5)
Predict: 0.0
Else (feature 540 > 87.0)
Predict: 1.0
Tree 3 (weight 1.0):
If (feature 518 <= 18.0)
If (feature 350 <= 97.5)
Predict: 1.0
Else (feature 350 > 97.5)
If (feature 356 <= 16.0)
Predict: 0.0
Else (feature 356 > 16.0)
Predict: 1.0
Else (feature 518 > 18.0)
Predict: 0.0
Tree 4 (weight 1.0):
If (feature 429 <= 11.5)
If (feature 358 <= 12.0)
Predict: 0.0
Else (feature 358 > 12.0)
Predict: 1.0
Else (feature 429 > 11.5)
Predict: 1.0
Tree 5 (weight 1.0):
If (feature 462 <= 62.5)
If (feature 240 <= 253.5)
Predict: 1.0
Else (feature 240 > 253.5)
Predict: 0.0
Else (feature 462 > 62.5)
Predict: 0.0
Tree 6 (weight 1.0):
If (feature 385 <= 4.0)
If (feature 545 <= 3.0)
If (feature 346 <= 2.0)
Predict: 0.0
Else (feature 346 > 2.0)
Predict: 1.0
Else (feature 545 > 3.0)
Predict: 0.0
Else (feature 385 > 4.0)
Predict: 1.0
Tree 7 (weight 1.0):
If (feature 512 <= 8.0)
If (feature 350 <= 7.0)
If (feature 298 <= 152.5)
Predict: 1.0
Else (feature 298 > 152.5)
Predict: 0.0
Else (feature 350 > 7.0)
Predict: 0.0
Else (feature 512 > 8.0)
Predict: 1.0
Tree 8 (weight 1.0):
If (feature 462 <= 62.5)
If (feature 324 <= 253.5)
Predict: 1.0
Else (feature 324 > 253.5)
Predict: 0.0
Else (feature 462 > 62.5)
Predict: 0.0
Tree 9 (weight 1.0):
If (feature 301 <= 30.0)
If (feature 517 <= 20.5)
If (feature 630 <= 5.0)
Predict: 0.0
Else (feature 630 > 5.0)
Predict: 1.0
Else (feature 517 > 20.5)
Predict: 0.0
Else (feature 301 > 30.0)
Predict: 1.0

## Gradient-boosted tree classifier

Gradient Boosting models are made up of a set of individual decision trees, trained sequentially, so that each new tree tries to improve on the errors of the previous trees. The prediction of a new observation is obtained by adding the predictions of all the individual trees that make up the model.
Tree-based methods have become one of the benchmarks in the predictive field due to the good results they generate in very diverse problems.
It is a generalization of the AdaBoost algorithm that allows us to use any cost function, as long as it is differentiable. The flexibility of this algorithm has made it possible to apply boosting to a multitude of problems (regression, multiple classification ...), making it one of the most successful machine learning methods. Although there are several adaptations, the general idea of ​​all of them is the same: train models sequentially, so that each model adjusts the residuals (errors) of the previous models.
One of the parameters of this type of argument is the learning rate, which controls the degree of improvement of a tree with respect to the previous one. A small learning rate means a slower improvement but better adapting to the data, which generally translates into improvements in the result at the cost of greater consumption of resources.
In addition to the size of the trees that constitute a Boosting procedure, another important parameter of the “gradient Boosting” is the number of iterations 𝑀. Normally, in each iteration the training error is reduced, so that for a sufficiently large 𝑀 this error can be made very small. However, fitting the training data so precisely can lead to overfitting, which is not good for future predictions. Therefore, there is an optimal number 𝑀 ∗ that minimizes the error of future predictions. A convenient way to estimate 𝑀 ∗ is to calculate the risk or prediction error as a function of 𝑀 in a validation sample. The value of 𝑀 that minimizes this risk is taken as an estimate of 𝑀 ∗.


This section will explain how the code for the implementation of the boosted gradient in scala works, using spark, the following code performs the steps to perform this regression.

```
// We must load the following libraries that will help us with this algorithm, since the booste gradient is a variant of the decision trees, the same libraries are used for the evaluation and indexing metadata.
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// We load the data for the model and it becomes a dataframe.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// We must assign an index to the labels to convert the string into a column with its respective index.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)
// We must assign an index to convert the created columns into a vector.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
// In this part we assign a category limit to avoid the overfitting error
  .setMaxCategories(4)
  .fit(data)

// We divide the data into the training and test sets leaving only 30% of the data for testing
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// The model for the gradient with the test data will be created.
val gbt = new GBTClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setMaxIter(10)

// We return the indexed values to the original values for use within the model.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// With the help of Pipeline, the estimator will be called in the input data set to fit a model, this through the indices and thus have the metadata that we require for the operation of the algorithm.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// With this part of the code, the model is "trained" with the indices that were previously taken within the training data.
val model = pipeline.fit(trainingData)

// Once we train the model we can perform the tests with the data from the test set and make the predictions that will be saved in a variable predictions
val predictions = model.transform(testData)

// We print the rows of the predictions variable, in this case we only select 5.
predictions.select("predictedLabel", "label", "features").show(5)

// With the help of the evaluator library, we are going to select data from the prediction columns that have an index and we are going to calculate the error in the test data according to the prediction.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println("Learned classification GBT model:\n" + gbtModel.toDebugString)
```

/*
Output:
+--------------+-----+--------------------+
|predictedLabel|label| features|
+--------------+-----+--------------------+
| 0.0| 0.0|(692,[123,124,125...|
| 0.0| 0.0|(692,[124,125,126...|
| 0.0| 0.0|(692,[124,125,126...|
| 0.0| 0.0|(692,[125,126,127...|
| 0.0| 0.0|(692,[126,127,128...|
+--------------+-----+--------------------+
only showing top 5 rows
Test Error = 0.0
Learned classification GBT model:
GBTClassificationModel (uid=gbtc_ef6c4e8f1ddc) with 10 trees
Tree 0 (weight 1.0):
If (feature 434 <= 88.5)
If (feature 99 in {2.0})
Predict: -1.0
Else (feature 99 not in {2.0})
Predict: 1.0
Else (feature 434 > 88.5)
Predict: -1.0
Tree 1 (weight 0.1):
If (feature 434 <= 88.5)
If (feature 549 <= 253.5)
If (feature 400 <= 159.5)
Predict: 0.4768116880884702
Else (feature 400 > 159.5)
Predict: 0.4768116880884703
Else (feature 549 > 253.5)
Predict: -0.4768116880884694
Else (feature 434 > 88.5)
If (feature 267 <= 254.5)
Predict: -0.47681168808847024
Else (feature 267 > 254.5)
Predict: -0.4768116880884712
Tree 2 (weight 0.1):
If (feature 434 <= 88.5)
If (feature 243 <= 25.0)
Predict: -0.4381935810427206
Else (feature 243 > 25.0)
If (feature 182 <= 32.0)
Predict: 0.4381935810427206
Else (feature 182 > 32.0)
If (feature 154 <= 9.5)
Predict: 0.4381935810427206
Else (feature 154 > 9.5)
Predict: 0.43819358104272066
Else (feature 434 > 88.5)
If (feature 461 <= 66.5)
Predict: -0.4381935810427206
Else (feature 461 > 66.5)
Predict: -0.43819358104272066
Tree 3 (weight 0.1):
If (feature 462 <= 62.5)
If (feature 549 <= 253.5)
Predict: 0.4051496802845983
Else (feature 549 > 253.5)
Predict: -0.4051496802845982
Else (feature 462 > 62.5)
If (feature 433 <= 244.0)
Predict: -0.4051496802845983
Else (feature 433 > 244.0)
Predict: -0.40514968028459836
Tree 4 (weight 0.1):
If (feature 462 <= 62.5)
If (feature 100 <= 193.5)
If (feature 235 <= 80.5)
If (feature 183 <= 88.5)
Predict: 0.3765841318352991
Else (feature 183 > 88.5)
If (feature 239 <= 9.0)
Predict: 0.3765841318352991
Else (feature 239 > 9.0)
Predict: 0.37658413183529915
Else (feature 235 > 80.5)
Predict: 0.3765841318352994
Else (feature 100 > 193.5)
Predict: -0.3765841318352994
Else (feature 462 > 62.5)
If (feature 129 <= 58.0)
If (feature 515 <= 88.0)
Predict: -0.37658413183529915
Else (feature 515 > 88.0)
Predict: -0.3765841318352994
Else (feature 129 > 58.0)
Predict: -0.3765841318352994
Tree 5 (weight 0.1):
If (feature 462 <= 62.5)
If (feature 293 <= 253.5)
Predict: 0.35166478958101
Else (feature 293 > 253.5)
Predict: -0.3516647895810099
Else (feature 462 > 62.5)
If (feature 433 <= 244.0)
Predict: -0.35166478958101005
Else (feature 433 > 244.0)
Predict: -0.3516647895810101
Tree 6 (weight 0.1):
If (feature 434 <= 88.5)
If (feature 548 <= 253.5)
If (feature 154 <= 24.0)
Predict: 0.32974984655529926
Else (feature 154 > 24.0)
Predict: 0.3297498465552994
Else (feature 548 > 253.5)
Predict: -0.32974984655530015
Else (feature 434 > 88.5)
If (feature 349 <= 2.0)
Predict: -0.32974984655529926
Else (feature 349 > 2.0)
Predict: -0.3297498465552994
Tree 7 (weight 0.1):
If (feature 434 <= 88.5)
If (feature 568 <= 253.5)
If (feature 658 <= 252.5)
If (feature 631 <= 27.0)
Predict: 0.3103372455197956
Else (feature 631 > 27.0)
If (feature 209 <= 62.5)
Predict: 0.3103372455197956
Else (feature 209 > 62.5)
Predict: 0.3103372455197957
Else (feature 658 > 252.5)
Predict: 0.3103372455197958
Else (feature 568 > 253.5)
Predict: -0.31033724551979525
Else (feature 434 > 88.5)
If (feature 294 <= 31.5)
If (feature 184 <= 110.0)
Predict: -0.3103372455197956
Else (feature 184 > 110.0)
Predict: -0.3103372455197957
Else (feature 294 > 31.5)
If (feature 350 <= 172.5)
Predict: -0.3103372455197956
Else (feature 350 > 172.5)
Predict: -0.31033724551979563
Tree 8 (weight 0.1):
If (feature 434 <= 88.5)
If (feature 627 <= 2.5)
Predict: -0.2930291649125432
Else (feature 627 > 2.5)
Predict: 0.2930291649125433
Else (feature 434 > 88.5)
If (feature 379 <= 11.5)
Predict: -0.2930291649125433
Else (feature 379 > 11.5)
Predict: -0.2930291649125434
Tree 9 (weight 0.1):
If (feature 434 <= 88.5)
If (feature 243 <= 25.0)
Predict: -0.27750666438358235
Else (feature 243 > 25.0)
If (feature 244 <= 10.5)
Predict: 0.27750666438358246
Else (feature 244 > 10.5)
If (feature 263 <= 237.5)
If (feature 159 <= 10.0)
Predict: 0.27750666438358246
Else (feature 159 > 10.0)
Predict: 0.2775066643835826
Else (feature 263 > 237.5)
Predict: 0.27750666438358257
Else (feature 434 > 88.5)
Predict: -0.2775066643835825

## Multilayer perceptron classifier
Later work with multilayer perceptrons has shown that they are able to approximate an XOR operator, as well as many other non-linear functions.
Just as Rosenblatt based the perceptron on a McCulloch-Pitts neuron, conceived in 1943, perceptrons themselves are building blocks that only prove useful in functions as large as multilayer perceptrons; The multilayer perceptron is the hello world of deep learning - a good place to start when you're learning about deep learning.
A multilayer perceptron (MLP) is a deep artificial neural network. It is made up of more than one perceptron. They are composed of an input layer to receive the signal, an output layer that makes a decision or prediction about the input, and between those two, an arbitrary number of hidden layers that are the true computational engine of the MLP. MLPs with a hidden layer are able to approximate any continuous function.
Feedback networks like MLPs are like tennis or ping pong. They are mainly involved in two movements, a constant coming and going. You can think of this guess-and-answer ping pong as a kind of fast-paced science, since every guess is a test of what we think we know, and every answer is feedback that lets us know how wrong we are.
The architecture of the multilayer Perceptron is characterized by having its neurons grouped in layers of different levels. Each of the layers is made up of a set of neurons and there are three different types of layers: the input layer, the hidden layers and the output layer.
The neurons of the input layer do not act as neurons themselves, but are only responsible for receiving signals or patterns from outside and propagating these signals to all neurons in the next layer. The last layer acts as the network's output, providing the network's response to each of the input patterns to the outside. Neurons in hidden layers perform non-linear processing of received patterns.
The multilayer perceptron evolves the simple perceptron and for this it incorporates layers of hidden neurons, with this it manages to represent non-linear functions.
The multilayer perceptron is made up of an input layer, an output layer, and n hidden layers in between; It is characterized by having disjoint but related outputs, in such a way that the output of one neuron is the input of the next.
In the multilayer perceptron, about 2 phases can be differentiated:

- Spread.
In which the output result of the network is calculated from the input values forward.
- Learning.
In which the errors obtained at the output of the perceptron are propagated backwards (backpropagation) in order to modify the weights of the connections so that the estimated value of the network increasingly resembles the real one, this approximation is carried out by the gradient function of the error.

This section will explain how the code of the multilayer perceptron implementation in scala works, using spark, in the following code the steps are performed to be able to perform this regression.

```
// We must load the following libraries that will help us with this algorithm, for the classification of the perceptron type and its evaluation.
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// We must load the data and convert it into a dataframe for use.
val data = spark.read.format("libsvm")
  .load("data/mllib/sample_multiclass_classification_data.txt")

// We divide the data into the training and test sets, leaving 40% for the test data and leaving the seed for the data.
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

// We must specify the layers of the neural network, in this case they will be 4, the following data are the intermediate values in the characteristics and leaving the last value for the outputs of the classes in this case 3.
val layers = Array[Int](4, 5, 4, 3)

// We must create the model with the training data so that the model will have the data, next to this we must place the seed for the data and have a maximum of 100.
val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)

// We enter the data into the model for your training work.
val model = trainer.fit(train)

// Once the training of the model is done, we proceed to enter the test data and calculate the precision of the model and print it, having a precision of 90% in the model.
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))
```

/*
Output:
Test set accuracy = 0.9019607843137255

## Linea der Support Vector Machine
Support vector machines were widely used before the deep learning era. For many applications the use of SVM was preferred over neural networks. The reason was that the math of SVMs was very well understood and the property of obtaining the maximum separation margin was very attractive. Neural networks could perform the classification in the wrong way as we have seen in the previous examples. Some success stories of support vector machines are:
optical character recognition face detection for digital cameras to target correctly spam filters for email image recognition on board satellites (knowing which parts of an image have
clouds, land, water, ice, etc.)
Currently, deep neural networks have a greater capacity for learning and
generalization than SVMs; Support Vector Machines allow finding the optimal way to classify between various classes. Optimal classification is done by maximizing the separation margin between classes. The vectors that define the edge of this separation are the support vectors. In the case that the classes are not linearly separable, we can use the kernel trick to add a new dimension where they are.
SVMs (or Support Vector Machines) are a type of Learning Machines. In
They are particularly one of those that first need to train with situations in which they are told the correct answer over many examples, and once it has been trained, it enters the use phase; and it simply becomes a box that returns the answer to a new case (in short, it is a supervised learning method).
Those who invented the SVs were Vladimir Vapnik (a statistically oriented person) and his colleagues at AT&T. The method is based on the use of mathematical programming, formulated in such a way that the statistical interpretation of the model is particularly appropriate. The model is rigorously supported by the statistical learning theories proposed by Vapnik.
SVM models will help us to predict data, as long as we have trained the machine. This prediction can be of several types:

- Binary classification prediction
- Multi-category ranking prediction
- General regression prediction.

```
// We must load the following library that will help us with this SVC algorithm and its operation.
import org.apache.spark.ml.classification.LinearSVC

// We load the data for the model and it becomes a dataframe.
val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// The model for the classification tree will be created with the test data.
val lsvc = new LinearSVC()
  .setMaxIter(10)
  .setRegParam(0.1)

// We will fit the model according to the parameters established for the data
val lsvcModel = lsvc.fit(training)

// We print the coefficients of the SVC model
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
```

/*
Output:
Coefficients: [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-5.170630317473439E-4,-1.172288654973735E-4,-8.882754836918948E-5,8.522360710187464E-5,0.0,0.0,-1.3436361263314267E-5,3.729569801338091E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.888949552633658E-4,2.9864059761812683E-4,3.793378816193159E-4,-1.762328898254081E-4,0.0,1.5028489269747836E-6,1.8056041144946687E-6,1.8028763260398597E-6,-3.3843713506473646E-6,-4.041580184807502E-6,2.0965017727015125E-6,8.536111642989494E-5,2.2064177429604464E-4,2.1677599940575452E-4,-5.472401396558763E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.21415502407147E-4,3.1351066886882195E-4,2.481984318412822E-4,0.0,-4.147738197636148E-5,-3.6832150384497175E-5,0.0,-3.9652366184583814E-6,-5.1569169804965594E-5,-6.624697287084958E-5,-2.182148650424713E-5,1.163442969067449E-5,-1.1535211416971104E-6,3.8138960488857075E-5,1.5823711634321492E-6,-4.784013432336632E-5,-9.386493224111833E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.3174897827077767E-4,1.7055492867397665E-4,0.0,-2.7978204136148868E-5,-5.88745220385208E-5,-4.1858794529775E-5,-3.740692964881002E-5,-3.9787939304887E-5,-5.545881895011037E-5,-4.505015598421474E-5,-3.214002494749943E-6,-1.6561868808274739E-6,-4.416063987619447E-6,-7.9986183315327E-6,-4.729962112535003E-5,-2.516595625914463E-5,-3.6407809279248066E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.4719098130614967E-4,0.0,-3.270637431382939E-5,-5.5703407875748054E-5,-5.2336892125702286E-5,-7.829604482365818E-5,-7.60385448387619E-5,-8.371051301348216E-5,-1.8669558753795108E-5,0.0,1.2045309486213725E-5,-2.3374084977016397E-5,-1.0788641688879534E-5,-5.5731194431606874E-5,-7.952979033591137E-5,-1.4529196775456057E-5,8.737948348132623E-6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0012589360772978808,-1.816228630214369E-4,-1.0650711664557365E-4,-6.040355527710781E-5,-4.856392973921569E-5,-8.973895954652451E-5,-8.78131677062384E-5,-5.68487774673792E-5,-3.780926734276347E-5,1.3834897036553787E-5,7.585485129441565E-5,5.5017411816753975E-5,-1.5430755398169695E-5,-1.834928703625931E-5,-1.0354008265646844E-4,-1.3527847721351194E-4,-1.1245007647684532E-4,-2.9373916056750564E-5,-7.311217847336934E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.858228613863785E-4,-1.2998173971449976E-4,-1.478408021316135E-4,-8.203374605865772E-5,-6.556685320008032E-5,-5.6392660386580244E-5,-6.995571627330911E-5,-4.664348159856693E-5,-2.3026593698824318E-5,7.398833979172035E-5,1.4817176130099997E-4,1.0938317435545486E-4,7.940425167011364E-5,-6.743294804348106E-7,-1.2623302721464762E-4,-1.9110387355357616E-4,-1.8611622108961136E-4,-1.2776766254736952E-4,-8.935302806524433E-5,-1.239417230441996E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.829530831354112E-4,-1.3912189600461263E-4,-1.2593136464577562E-4,-5.964745187930992E-5,-5.360328152341982E-5,-1.0517880662090183E-4,-1.3856124131005022E-4,-7.181032974125911E-5,2.3249038865093483E-6,1.566964269571967E-4,2.3261206954040812E-4,1.7261638232256968E-4,1.3857530960270466E-4,-1.396299028868332E-5,-1.5765773982418597E-4,-2.0728798812007546E-4,-1.9106441272002828E-4,-1.2744834161431415E-4,-1.2755611630280015E-4,-5.1885591560478935E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.59081567023441E-4,-1.216531230287931E-4,-5.623851079809818E-5,-3.877987126382982E-5,-7.550900509956966E-5,-1.0703140005463545E-4,-1.4720428138106226E-4,-8.781423374509368E-5,7.941655609421792E-5,2.3206354986219992E-4,2.7506982343672394E-4,2.546722233188043E-4,1.810821666388498E-4,-1.3069916689929984E-5,-1.842374220886751E-4,-1.977540482445517E-4,-1.7722074063670741E-4,-1.487987014723575E-4,-1.1879021431288621E-4,-9.755283887790393E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.302740311359312E-4,-5.3683030235535024E-5,-1.7631200013656873E-5,-7.846611034608254E-5,-1.22100767283256E-4,-1.7281968533449702E-4,-1.5592346128894157E-4,-5.239579492910452E-5,1.680719343542442E-4,2.8930086786548053E-4,3.629921493231646E-4,2.958223512266975E-4,2.1770466955449064E-4,-6.40884808188951E-5,-1.9058225556007997E-4,-2.0425138564600712E-4,-1.711994903702119E-4,-1.3853486798341369E-4,-1.3018592950855062E-4,-1.1887779512760102E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-7.021411112285498E-5,-1.694500843168125E-5,-7.189722824172193E-5,-1.4560828004346436E-4,-1.4935497340563198E-4,-1.9496419340776972E-4,-1.7383743417254187E-4,-3.3438825792010694E-5,2.866538327947017E-4,2.9812321570739803E-4,3.77250607691119E-4,3.211702827486386E-4,2.577995115175486E-4,-1.6627385656703205E-4,-1.8037105851523224E-4,-2.0419356344211325E-4,-1.7962237203420184E-4,-1.3726488083579862E-4,-1.3461014473741762E-4,-1.2264216469164138E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0015239752514658556,-5.472330865993813E-5,-9.65684394936216E-5,-1.3424729853486994E-4,-1.4727467799568E-4,-1.616270978824712E-4,-1.8458259010029364E-4,-1.9699647135089726E-4,1.3085261294290817E-4,2.943178857107149E-4,3.097773692834126E-4,4.112834769312103E-4,3.4113620757035025E-4,1.6529945924367265E-4,-2.1065410862650534E-4,-1.883924081539624E-4,-1.979586414569358E-4,-1.762131187223702E-4,-1.272343622678854E-4,-1.2708161719220297E-4,-1.4812221011889967E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.001140680600536578,-1.323467421269896E-4,-1.2904607854274846E-4,-1.4104748544921958E-4,-1.5194605434027872E-4,-2.1104539389774283E-4,-1.7911827582001795E-4,-1.8952948277194435E-4,2.1767571552539842E-4,3.0201791656326465E-4,4.002863274397723E-4,4.0322806756364006E-4,4.118077382608461E-4,3.7917405252859545E-6,-1.9886290660234838E-4,-1.9547443112937263E-4,-1.9857348218680872E-4,-1.3336892200703206E-4,-1.2830129292910815E-4,-1.1855916317355505E-4,-1.765597203760205E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0010938769592297973,-1.2785475305234688E-4,-1.3424699777466666E-4,-1.505200652479287E-4,-1.9333287822872713E-4,-2.0385160086594937E-4,-1.7422470698847553E-4,4.63598443910652E-5,2.0617623087127652E-4,2.862882891134514E-4,4.074830988361515E-4,3.726357785147985E-4,3.507520190729629E-4,-1.516485494364312E-4,-1.7053751921469217E-4,-1.9638964654350848E-4,-1.9962586265806435E-4,-1.3612312664311173E-4,-1.218285533892454E-4,-1.1166712081624676E-4,-1.377283888177579E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.044386260118809E-4,-1.240836643202059E-4,-1.335317492716633E-4,-1.5783442604618277E-4,-1.9168434243384107E-4,-1.8710322733892716E-4,-1.1283989231463139E-4,1.1136504453105364E-4,1.8707244892705632E-4,2.8654279528966305E-4,4.0032117544983536E-4,3.169637536305377E-4,2.0158994278679014E-4,-1.3139392844616033E-4,-1.5181070482383948E-4,-1.825431845981843E-4,-1.602539928567571E-4,-1.3230404795396355E-4,-1.1669138691257469E-4,-1.0532154964150405E-4,-1.3709037042366007E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-4.0287410145021705E-4,-1.3563987950912995E-4,-1.3225887084018914E-4,-1.6523502389794188E-4,-2.0175074284706945E-4,-1.572459106394481E-4,2.577536501278673E-6,1.312463663419457E-4,2.0707422291927531E-4,3.9081065544314936E-4,3.3487058329898135E-4,2.5790441367156086E-4,2.6881819648016494E-5,-1.511383586714907E-4,-1.605428139328567E-4,-1.7267287462873575E-4,-1.1938943768052963E-4,-1.0505245038633314E-4,-1.1109385509034013E-4,-1.3469914274864725E-4,-2.0735223736035555E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-5.034374233912422E-4,-1.5961213688405883E-4,-1.274222123810994E-4,-1.582821104884909E-4,-2.1301220616286252E-4,-1.2933366375029613E-4,1.6802673102179614E-5,1.1020918082727098E-4,2.1160795272688753E-4,3.4873421050827716E-4,2.6487211944380384E-4,1.151606835026639E-4,-5.4682731396851946E-5,-1.3632001630934325E-4,-1.4340405857651405E-4,-1.248695773821634E-4,-8.462873247977974E-5,-9.580708414770257E-5,-1.0749166605399431E-4,-1.4618038459197777E-4,-3.7556446296204636E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-5.124342611878493E-4,-2.0369734099093433E-4,-1.3626985098328694E-4,-1.3313768183302705E-4,-1.871555537819396E-4,-1.188817315789655E-4,-1.8774817595622694E-5,5.7108412194993384E-5,1.2728161056121406E-4,1.9021458214915667E-4,1.2177397895874969E-4,-1.2461153574281128E-5,-7.553961810487739E-5,-1.0242174559410404E-4,-4.44873554195981E-5,-9.058561577961895E-5,-6.837347198855518E-5,-8.084409304255458E-5,-1.3316868299585082E-4,-2.0335916397646626E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.966510928472775E-4,-1.3738983629066386E-4,-3.7971221409699866E-5,-6.431763035574533E-5,-1.1857739882295322E-4,-9.359520863114822E-5,-5.0878371516215046E-5,-8.269367595092908E-8,0.0,1.3434539131099211E-5,-1.9601690213728576E-6,-2.8527045990494954E-5,-7.410332699310603E-5,-7.132130570080122E-5,-4.9780961185536E-5,-6.641505361384578E-5,-6.962005514093816E-5,-7.752898158331023E-5,-1.7393609499225025E-4,-0.0012529479255443958,0.0,0.0,2.0682521269893754E-4,0.0,0.0,0.0,0.0,0.0,-4.6702467383631055E-4,-1.0318036388792008E-4,1.2004408785841247E-5,0.0,-2.5158639357650687E-5,-1.2095240910793449E-5,-5.19052816902203E-6,-4.916790639558058E-6,-8.48395853563783E-6,-9.362757097074547E-6,-2.0959335712838412E-5,-4.7790091043859085E-5,-7.92797600958695E-5,-4.462687041778011E-5,-4.182992428577707E-5,-3.7547996285851254E-5,-4.52754480225615E-5,-1.8553562561513456E-5,-2.4763037962085644E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.4886180455242474E-4,-5.687523659359091E-6,7.380040279654313E-5,4.395860636703821E-5,7.145198242379862E-5,6.181248343370637E-6,0.0,-6.0855538083486296E-5,-4.8563908323274725E-5,-4.117920588930435E-5,-4.359283623112936E-5,-6.608754161500044E-5,-5.443032251266018E-5,-2.7782637880987207E-5,0.0,0.0,2.879461393464088E-4,-0.0028955529777851255,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.2312114837837392E-4,-1.9526747917254753E-5,-1.6999506829961688E-5,5.4835294148085086E-5,1.523441632762399E-5,-5.8365604525328614E-5,-1.2378194216521848E-4,-1.1750704953254656E-4,-6.19711523061306E-5,-5.042009645812091E-5,-1.4055260223565886E-4,-1.410330942465528E-4,-1.9272308238929396E-4,-4.802489964676616E-4] Intercept: 0.012911305214513969

## Naive Bayes
In a broad sense, Naive Bayes models are a special class of algorithms for
classification of Machine Learning, or Machine Learning, as we will refer to from now on. They are based on a statistical classification technique called "Bayes's theorem." These models are called "Naive" algorithms, or "Innocents" in Spanish. They assume that the predictor variables are independent of each other. In other words, that the presence of a certain feature in a data set is not at all related to the presence of any other feature. They provide an easy way to build very well behaved models due to their simplicity.
Naive Bayes strengths and weaknesses

The main strengths are:
- A fast and easy way to predict classes, for binary and multiclass classification problems.
- In cases where a presumption of independence is appropriate, the algorithm performs better than other classification models, even with less training data.
- The decouples of the class conditional characteristic distributions mean that each distribution can be estimated independently as having only one dimension. This helps with dimensionality issues and improves performance.

The main weak points are:
- Although they are pretty good classifiers, Naive Bayes algorithms are notorious for being poor estimators. Therefore, the probabilities obtained should not be taken very seriously.
- The Naive independence presumption most likely does not reflect what data is like in the real world.
- When the test data set has a characteristic that has not been observed in the training set, the model will assign it a probability of zero and it will be useless to make predictions. One of the main methods to avoid this is the smoothing technique, the Laplace estimation being one of the most popular.

Naive Bayes classifier types:
- Naive multinomial Bayes:
  - This is mainly used for document classification problems, that is, if a document belongs to the category of sports, politics, technology, etc. The characteristics / predictors used by the classifier are the frequency of the words present in the document.
- Naive Bernoulli Bayes:
  - This is similar to multinomial naive bayes, but the predictors are Boolean variables. The parameters we use to predict the class variable take only yes or no values, for example whether a word appears in the text or not.
- Naive Gaussian Bayes:
  - When the predictors take a continuous value and are not discrete, we assume that these values are sampled from a Gaussian distribution.

This section will explain how the code of the Naive bayes implementation in scala works, using spark, the following code performs the steps to perform this regression.

```
// We must load the following libraries that will help us with this algorithm of // classification
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// We load the data for the model and it becomes a dataframe.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// We divide the data into the training and test sets leaving only 30% of the data for testing
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

// The model for the classification tree will be created with the test data.
val model = new NaiveBayes()
  .fit(trainingData)

// We print the rows of the predictions variable from the model with the test data and we will adjust the model according to the parameters established for the data
val predictions = model.transform(testData)
predictions.show()

//Con ayuda de la librería evaluator vamos a seleccionar datos de las columnas prediction y que tenga índice y vamos a realizar el cálculo del error en los datos de prueba según la predicción.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test set accuracy = " + accuracy)
```

/*
Output:
+-----+--------------------+--------------------+-----------+----------+
|label| features| rawPrediction|probability|prediction|
+-----+--------------------+--------------------+-----------+----------+
| 0.0|(692,[95,96,97,12...|[-173678.60946628...| [1.0,0.0]| 0.0|
| 0.0|(692,[98,99,100,1...|[-178107.24302988...| [1.0,0.0]| 0.0|
| 0.0|(692,[100,101,102...|[-100020.80519087...| [1.0,0.0]| 0.0|
| 0.0|(692,[124,125,126...|[-183521.85526462...| [1.0,0.0]| 0.0|
| 0.0|(692,[127,128,129...|[-183004.12461660...| [1.0,0.0]| 0.0|
| 0.0|(692,[128,129,130...|[-246722.96394714...| [1.0,0.0]| 0.0|
| 0.0|(692,[152,153,154...|[-208696.01108598...| [1.0,0.0]| 0.0|
| 0.0|(692,[153,154,155...|[-261509.59951302...| [1.0,0.0]| 0.0|
| 0.0|(692,[154,155,156...|[-217654.71748256...| [1.0,0.0]| 0.0|
| 0.0|(692,[181,182,183...|[-155287.07585335...| [1.0,0.0]| 0.0|
| 1.0|(692,[99,100,101,....|[-145981.83877498...| [0.0,1.0]| 1.0|
| 1.0|(692,[100,101,102...|[-147685.13694275...| [0.0,1.0]| 1.0|
| 1.0|(692,[123,124,125...|[-139521.98499849...| [0.0,1.0]| 1.0|
| 1.0|(692,[124,125,126...|[-129375.46702012...| [0.0,1.0]| 1.0|
| 1.0|(692,[126,127,128...|[-145809.08230799...| [0.0,1.0]| 1.0|
| 1.0|(692,[127,128,129...|[-132670.15737290...| [0.0,1.0]| 1.0|
| 1.0|(692,[128,129,130...|[-100206.72054749...| [0.0,1.0]| 1.0|
| 1.0|(692,[129,130,131...|[-129639.09694930...| [0.0,1.0]| 1.0|
| 1.0|(692,[129,130,131...|[-143628.65574273...| [0.0,1.0]| 1.0|
| 1.0|(692,[129,130,131...|[-129238.74023248...| [0.0,1.0]| 1.0|
+-----+--------------------+--------------------+-----------+----------+
only showing top 20 rows
Test set accuracy = 1.0




















# Algoritmos de Regresión

Unlike the classification algorithms that give us data that can be from a category, the regression gives us a numerical value, this will be a value within an infinite set of values.
- Some simple regression examples are:
- Predict the fluctuations of virtual currencies.
- Estimate how long it will take a car to reach its destination.
- Predict how many products are sold in a certain category.

In the world of machine learning there are many techniques that we can use for regression problems, some examples are:

- Linear and nonlinear regression.
- Support vector machines.
- Decision tree.
- Random forests.
- Deep learning.

Within the world of machine learning there are algorithms that work in both categories, that is why we can have trees in classification algorithms and in regression.
Regression analysis concentrates on setting a variable as dependent and seeing its behavior with another series of independent and / or changing variables. With these models we can build a machine learning process that facilitates the prediction of results and forecasts.




















## Linear regression
Linear regression is a field of study that emphasizes the statistical relationship between two continuous variables known as predictor and response variables. (Note: when there is more than one predictor variable, it becomes multiple linear regression.)
- The predictor variable is most often denoted x and is also known as the independent variable.
- The response variable is most often denoted as and and is also known as the dependent variable.

A regression model is a model that allows us to describe how one variable x influences another variable y. Being the variable x an independent variable and the variable y, a dependent. Aims to establish estimates of y for different values ​​of X
The simple linear regression forecast is an optimal model for demand patterns with a trend (increasing or decreasing), that is, patterns that present a linear relationship between demand and time.
This form of analysis estimates the coefficients of the linear equation, involving one or more independent variables that best predict the value of the dependent variable. Linear regression fits a straight line or surface that minimizes discrepancies between the expected and actual output values. There are simple linear regression calculators that use the "least squares" method to determine the best fit line for a set of paired data. Next, the value of X (dependent variable) with respect to Y (independent variable) is calculated.
Linear regression models are relatively straightforward and provide an easy-to-interpret mathematical formula that can generate predictions. Linear regression can be applied to various areas of business and academic studies.

Assumptions that must be taken into account to be successful with linear regression analysis:

- For each variable: Consider the number of valid cases, the mean, and the standard deviation.
- For each model: Consider regression coefficients, correlation matrix, partial and semi-partial correlations, multiple R, R2, adjusted R2, change in R2, standard error of estimate, analysis of variance table, predicted values, and residuals. Also, consider 95 percent confidence intervals for each regression coefficient, variance-covariance matrix, variance inflation factor, tolerance, Durbin-Watson test, distance measures (Mahalanobis, Cook, and leverage values), DfBeta, DfFit, prediction intervals and diagnostic information on each case.
- Graphs: Consider scatter plots, partial and normal probability graphs, and histograms.
- Data: The dependent and independent variables must be quantitative. Categorical variables, such as religion, main field of study, or region of residence, must be recoded to be binary (dummy) variables or other types of contrast variables.
- Other hypotheses: For each value of the independent variable, the distribution of the dependent variable must be normal. The variance of the distribution of the dependent variable must be constant for all values ​​of the independent variable. The relationship between the dependent variable and each independent variable must be linear; furthermore, all observations must be independent.

This section will explain how the code of the linear regression implementation works in scala, using spark, the following code performs the steps to perform this regression.

```
// We must import the following library that will help us with this linear regression algorithm.
import org.apache.spark.ml.regression.LinearRegression

// We load the data for the model and it becomes a dataframe. to use as training data.
val training = spark.read.format("libsvm")
  .load("data/mllib/sample_linear_regression_data.txt")

// With the help of the imported library for the algorithm, we must place the information of the parameters.
val lr = new LinearRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

// we fit the training data to the model for its operation.
val lrModel = lr.fit(training)

// We print the coefficients to visualize a little more graph next to the interceptions.
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// From the created model we must obtain a summary of the data that comes from the training set and we are going to print some metrics with which we can determine the residuals and the r2
val trainingSummary = lrModel.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")
```

/*
Output:
Coefficients: [0.0,0.32292516677405936,-0.3438548034562218,1.9156017023458414,0.05288058680386263,0.765962720459771,0.0,-0.15105392669186682,-0.21587930360904642,0.22025369188813426] Intercept: 0.1598936844239736
numIterations: 7
objectiveHistory: [0.49999999999999994,0.4967620357443381,0.4936361664340463,0.4936351537897608,0.4936351214177871,0.49363512062528014,0.4936351206216114]
+--------------------+
| residuals|
+--------------------+
| -9.889232683103197|
| 0.5533794340053554|
| -5.204019455758823|
| -20.566686715507508|
| -9.4497405180564|
| -6.909112502719486|
| -10.00431602969873|
| 2.062397807050484|
| 3.1117508432954772|
| -15.893608229419382|
| -5.036284254673026|
| 6.483215876994333|
| 12.429497299109002|
| -20.32003219007654|
| -2.0049838218725005|
| -17.867901734183793|
| 7.646455887420495|
| -2.2653482182417406|
|-0.10308920436195645|
| -1.380034070385301|
+--------------------+
only showing top 20 rows
RMSE: 10.189077167598475
r2: 0.022861466913958184


## Generalized linear regression
Generalized linear regression algorithms
It is a flexible generalization of ordinary linear regression that allows response variables that have error distribution patterns other than a normal distribution. The GLM generalizes linear regression by allowing the linear model to be related to the response variable through a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.
John Nelder and Robert Wedderburn formulated generalized linear models as a way to unify other statistical models, such as linear regression, logistic regression, and Poisson regression. They proposed an iteratively weighted least squares method for the maximum likelihood estimation of the model parameters. Maximum likelihood estimation remains popular and is the default method in many statistical computing packages. Other approaches have been developed, including Bayesian approaches and least squares fits to stabilized responses of variance.
In a generalized linear model (GLM), each Y outcome of the dependent variables is assumed to be generated from a particular distribution in the exponential family, a large class of probability distributions that includes the normal, binomial, Poisson, and gamma distributions. , among others.
In contrast to linear regression where the output is assumed to follow a Gaussian distribution, generalized linear models (GLM) are specifications of linear models where the response variable “Yi” follows some distribution of the exponential family of distributions.

Available families
 ![familys](https://user-images.githubusercontent.com/60914099/104147308-8f4d6500-5382-11eb-96b1-7279b554b1ad.PNG)


This section will explain how the code for the implementation of generalized linear regression through gauss in scala works, using spark, the following code performs the steps to perform this regression.

```
// We must import the following library that will help us with this generalized linear regression algorithm.
import org.apache.spark.ml.regression.GeneralizedLinearRegression

// We load the data for the model and it becomes a dataframe.
val dataset = spark.read.format("libsvm")
  .load("data/mllib/sample_linear_regression_data.txt")

// We initialize a variable to load the generalized linear regression model, within this we must place the parameters, in this case the family parameter would be Gaussian but here we can choose or place another “family” which can be binomial, poisson, gamma or tweedie but by default it will always be Gaussian, in the link parameter we use it to name the function that gives the relationship between the predictor and the function, we place the number of maximum iterations that in this case would be 10 and the regularization of the data.
val glr = new GeneralizedLinearRegression()
  .setFamily("gaussian")
  .setLink("identity")
  .setMaxIter(10)
  .setRegParam(0.3)

// we fit the model to receive the data.
val model = glr.fit(dataset)


// We can print the coefficients of the calculation and the interceptions of the regression model.
println(s"Coefficients: ${model.coefficients}")
println(s"Intercept: ${model.intercept}")

// We get a summary of the training data and can print some data of the metrics.
val summary = model.summary
println(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
println(s"T Values: ${summary.tValues.mkString(",")}")
println(s"P Values: ${summary.pValues.mkString(",")}")
println(s"Dispersion: ${summary.dispersion}")
println(s"Null Deviance: ${summary.nullDeviance}")
println(s"Residual Degree Of Freedom Null: ${summary.residualDegreeOfFreedomNull}")
println(s"Deviance: ${summary.deviance}")
println(s"Residual Degree Of Freedom: ${summary.residualDegreeOfFreedom}")
println(s"AIC: ${summary.aic}")
println("Deviance Residuals: ")
summary.residuals().show()
```

/*
Output:
Coefficients: [0.010541828081257216,0.8003253100560949,-0.7845165541420371,2.3679887171421914,0.5010002089857577,1.1222351159753026,-0.2926824398623296,-0.49837174323213035,-0.6035797180675657,0.6725550067187461]
Intercept: 0.14592176145232041
Coefficient Standard Errors: 0.7950428434287478,0.8049713176546897,0.7975916824772489,0.8312649247659919,0.7945436200517938,0.8118992572197593,0.7919506385542777,0.7973378214726764,0.8300714999626418,0.7771333489686802,0.463930109648428
T Values: 0.013259446542269243,0.9942283563442594,-0.9836067393599172,2.848657084633759,0.6305509179635714,1.382234441029355,-0.3695715687490668,-0.6250446546128238,-0.7271418403049983,0.8654306337661122,0.31453393176593286
P Values: 0.989426199114056,0.32060241580811044,0.3257943227369877,0.004575078538306521,0.5286281628105467,0.16752945248679119,0.7118614002322872,0.5322327097421431,0.467486325282384,0.3872259825794293,0.753249430501097
Dispersion: 105.60988356821714
Null Deviance: 53229.3654338832
Residual Degree Of Freedom Null: 500
Deviance: 51748.8429484264
Residual Degree Of Freedom: 490
AIC: 3769.1895871765314
Deviance Residuals:
+-------------------+
| devianceResiduals|
+-------------------+
|-10.974359174246889|
| 0.8872320138420559|
| -4.596541837478908|
|-20.411667435019638|
|-10.270419345342642|
|-6.0156058956799905|
|-10.663939415849267|
| 2.1153960525024713|
| 3.9807132379137675|
|-17.225218272069533|
| -4.611647633532147|
| 6.4176669407698546|
| 11.407137945300537|
| -20.70176540467664|
| -2.683748540510967|
|-16.755494794232536|
| 8.154668342638725|
|-1.4355057987358848|
|-0.6435058688185704|
| -1.13802589316832|
+-------------------+
only showing top 20 rows


## Decision tree regression
A decision tree is a flowchart like tree structure, where each internal node denotes a test on an attribute, each branch represents a test result
and each terminal node contains a class label.
Typically starts with a single node, branching off into possible outcomes. Each of those results leads to additional nodes, which branch out into other possibilities. This gives it an arboreal shape.
There are three different types of nodes:

- Opportunity nodes.
  - An opportunity node shows the probabilities of certain outcomes.
- decision nodes.
  - A decision node shows a decision to be made.
- end nodes.
  - An end node shows the end result of a decision path.


Decision trees and their sets are popular methods for machine learning regression and classification tasks. Decision trees are widely used because they are easy to interpret, handle categorical features, extend to multiclass classification settings, do not require feature scaling, and can capture nonlinearities and feature interactions. Tree set algorithms, such as random forests and momentum, are among the best for classification and regression tasks.

The decision tree is a greedy algorithm that performs recursive binary partitioning of feature space. The tree predicts the same label for each lower partition (leaf). Each partition is avidly chosen by selecting the best division from a set of possible divisions, in order to maximize the information gain in a tree node.

This section will explain how the code of the implementation of the regression decision tree in scala works, using spark, in the following code the steps are performed to perform this regression.

```
// We must load the following libraries that will help us with this algorithm of // classification
// The pipeline library helps us with the estimation of the data, while the decision // libraries help us with the model and its evaluation, while the feature library helps us // with the extraction and transformation of characteristics in the data.

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor

// We load the data for the model and it becomes a dataframe.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// We must assign an index to convert the created columns into a vector.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
// In this part we assign a category limit to avoid the overfitting error
  .setMaxCategories(4)
  .fit(data)

// We divide the data into the training and test sets leaving only 30% of the data for testing
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// The model for the classification tree will be created with the test data.
val dt = new DecisionTreeRegressor()
  .setLabelCol("label")
  .setFeaturesCol("indexedFeatures")

// With the help of Pipeline the estimator will be called in the input data set to fit a model, this through the indexes and thus have the metadata
val pipeline = new Pipeline()
  .setStages(Array(featureIndexer, dt))

// With this part of the code, the model is "trained" with the indices that were previously taken within the training data.
val model = pipeline.fit(trainingData)

// Once we train the model we can perform the tests with the data from the test set and make the predictions that will be saved in a variable predictions
val predictions = model.transform(testData)

// We print the rows of the predictions variable, in this case we only select 5.
predictions.select("prediction", "label", "features").show(5)

// With the help of the evaluator library, we are going to select data from the prediction columns that have an index and we are going to calculate the error in the test data according to the prediction.
val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
println("Learned regression tree model:\n" + treeModel.toDebugString)
```

/*
Output:
+----------+-----+--------------------+
|prediction|label| features|
+----------+-----+--------------------+
| 0.0| 0.0|(692,[123,124,125...|
| 0.0| 0.0|(692,[124,125,126...|
| 0.0| 0.0|(692,[124,125,126...|
| 0.0| 0.0|(692,[126,127,128...|
| 0.0| 0.0|(692,[126,127,128...|
+----------+-----+--------------------+
only showing top 5 rows
Root Mean Squared Error (RMSE) on test data = 0.19611613513818404
Learned regression tree model:
DecisionTreeRegressionModel (uid=dtr_f30a452bc6d9) of depth 2 with 5 nodes
If (feature 406 <= 126.5)
If (feature 99 in {0.0,3.0})
Predict: 0.0
Else (feature 99 not in {0.0,3.0})
Predict: 1.0
Else (feature 406 > 126.5)
Predict: 1.0


## Conclusión.
In the branch of machine learning, algorithms are the cornerstone of supervised learning, in the categories of classification and regression, with the help of these algorithms companies have been able to predict hundreds of movements and what could happen with these movements, classifying algorithms have been used in areas of health, construction, climate, its usefulness is enough because with the help of these we can know the probabilities of data that we want, an example could be the probability in which a patient may suffer from a disease based on other data, such as We can see throughout this document algorithms and their application in scala are described and it can be said that they all work for something specific and in which some have an advantage over others, so it is necessary to know what is required to be done with the information you have.

























Bibliografía.

- Gandhi, R. (2018, 26 junio). Naive Bayes Classifier - Towards Data Science. Medium. https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
- Acerca de la regresión lineal. (s. f.). México | IBM. Recuperado 19 de diciembre de 2020, de https://www.ibm.com/mx-es/analytics/learn/linear-regression
- Spark. (s. f.) | Apache. Recuperado 19 de diciembre de 2020, de https://spark.apache.org/docs/1.5.2/api/scala/index.html#org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm
- Spark. (s. f.).  | Apache. Recuperado 19 de diciembre de 2020, de https://spark.apache.org/docs/1.5.2/api/scala/index.html#org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm
- Heras, J. M. (2020, 29 septiembre). ¿Clasificación o Regresión? IArtificial.net. https://www.iartificial.net/clasificacion-o-regresion/#Regresion
- Data, S. B. (2019a, diciembre 7). ¿Qué es la regresión lineal? Parte 1. sitiobigdata.com. https://sitiobigdata.com/2019/10/25/que-es-la-regresion-lineal/
- Koehrsen, W. (2020, 18 agosto). Random Forest Simple Explanation - Will Koehrsen. Medium. https://williamkoehrsen.medium.com/random-forest-simple-explanation-377895a60d2d
- Honkela, A. (2001, 30 mayo). Multilayer perceptrons. aalto. http://users.ics.aalto.fi/ahonkela/dippa/node41.html
- Nicholson, C. (s. f.). A Beginner’s Guide to Multilayer Perceptrons (MLP). Pathmind. Recuperado 19 de diciembre de 2020, de https://wiki.pathmind.com/multilayer-perceptron
- Heras, J. M. (2019, 28 mayo). Máquinas de Vectores de Soporte (SVM). IArtificial.net. https://www.iartificial.net/maquinas-de-vectores-de-soporte-svm/
- Máquina de vectores de soporte (SVM). (s. f.). MATLAB & Simulink. Recuperado 18 de diciembre de 2020, de https://la.mathworks.com/discovery/support-vector-machine.html
- L. (2019, 24 septiembre). Naive Bayes – Teoría. Ligdi González. https://ligdigonzalez.com/naive-bayes-teoria-machine-learning/#:%7E:text=Na%C3%AFve%20Bayes%20o%20el%20Ingenuo,de%20independencia%20entre%20los%20predictores.&text=Esta%20suposici%C3%B3n%20se%20denomina%20independencia%20condicional%20de%20clase.+
- Caparrini, F. S. (2020, 14 diciembre). Aprendizaje Supervisado y No Supervisado - Fernando Sancho Caparrini. cs. http://www.cs.us.es/%7Efsancho/?e=77
- Caparrini, F. S. (2017, 23 septiembre). Introducción al Aprendizaje Automático - Fernando Sancho Caparrini. cs. http://www.cs.us.es/%7Efsancho/?e=75
- N. (2018, 13 mayo). Regresión Lineal –. Aprende Machine Learning. https://www.aprendemachinelearning.com/tag/regresion-lineal/#:%7E:text=La%20regresi%C3%B3n%20lineal%20es%20un,Machine%20Learning%20y%20en%20estad%C3%ADstica.&text=En%20estad%C3%ADsticas%2C%20regresi%C3%B3n%20lineal%20es,explicativas%20nombradas%20con%20%E2%80%9CX%E2%80%9D.
- González, B. A. (s. f.). Conceptos básicos de Machine Learning – Cleverdata. cleverdata. Recuperado 16 de diciembre de 2020, de https://cleverdata.io/conceptos-basicos-machine-learning/
- Roman, V. (2019, 1 abril). Aprendizaje Supervisado: Introducción a la Clasificación y Principales Algoritmos. Medium. https://medium.com/datos-y-ciencia/aprendizaje-supervisado-introducci%C3%B3n-a-la-clasificaci%C3%B3n-y-principales-algoritmos-dadee99c9407
- Perez, A. J., Kizys, R., & Manzanedo Del Hoyo, L. M. (2010, enero). Regresion logistica Binaria. MECD. https://econometriai.files.wordpress.com/2010/01/reg-logistica.pdf
- Instituto de Economía y geografía, & Rojo, J. M. (2007, febrero). Regresión con variable dependiente cualitativa. Laboratorio de estadística. http://humanidades.cchs.csic.es/cchs/web_UAE/tutoriales/PDF/Regresion_variable_dependiente_dicotomica_3.pdf
- Rodrigo, J. A. (2016, agosto). Regresión logística simple y múltiple. cienciadedatos. https://www.cienciadedatos.net/documentos/27_regresion_logistica_simple_y_multiple.html
- IBM Knowledge Center. (s. f.). IBM. Recuperado 17 de diciembre de 2020, de https://www.ibm.com/support/knowledgecenter/es/SSLVMB_sub/statistics_mainhelp_ddita/spss/regression/idh_mnlr_modl.html
- Data, S. B. (2019, 14 diciembre). Árbol de decisión en Machine Learning (Parte 1). sitiobigdata.com. https://sitiobigdata.com/2019/12/14/arbol-de-decision-en-machine-learning-parte-1/
- Johanna Orellana Alvear - johanna.orellana@ucuenca.edu.ec. (s. f.). Arboles de decision y Random Forest. bookdown. Recuperado 17 de diciembre de 2020, de https://bookdown.org/content/2031/ensambladores-random-forest-parte-ii.html
- Rodrigo, J. A. (s. f.). Arboles de decision, Random Forest, Gradient Boosting y C5.0. cienciadedatos. Recuperado 18 de diciembre de 2020, de https://www.cienciadedatos.net/documentos/33_arboles_de_prediccion_bagging_random_forest_boosting#Gradient_Boosting
- N.A. (s. f.). Apuntes de estadistica. bioestadistica. Recuperado 18 de diciembre de 2020, de https://www.bioestadistica.uma.es/apuntesMaster/regresion-logistica-binaria.html
- Roman, V. (2019b, abril 29). Algoritmos Naive Bayes: Fundamentos e Implementación. Medium. https://medium.com/datos-y-ciencia/algoritmos-naive-bayes-fudamentos-e-implementaci%C3%B3n-4bcb24b307f

### Collaborator
* **Rexfor** - [Github] (https://github.com/Rexfor)
