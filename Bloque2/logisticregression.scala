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
