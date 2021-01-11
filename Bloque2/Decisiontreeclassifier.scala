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
