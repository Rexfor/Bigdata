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
