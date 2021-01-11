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
