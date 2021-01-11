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
