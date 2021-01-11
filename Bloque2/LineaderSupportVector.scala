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
