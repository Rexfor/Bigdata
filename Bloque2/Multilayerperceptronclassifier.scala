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
