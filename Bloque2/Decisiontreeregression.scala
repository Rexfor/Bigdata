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
