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