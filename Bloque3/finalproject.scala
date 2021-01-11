//Contenido del proyecto
//1.- Objectivo: Comparacion del rendimiento siguientes algoritmos de machine learning
// - SVM
// - Decision Three
// - Logistic Regresion
// - Multilayer perceptron
//Con el siguiente data set: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing 

// Contenido del documento de proyecto final
// 1. Portada
// 2. Indice
// 3. Introduccion
// 4. Marco teorico de los algoritmos
// 5. Implementacion (Que herramientas usaron y porque (en este caso spark con scala))
// 6. Resultados (Un tabular con los datos por cada algoritmo para ver su preformance)
//    y su respectiva explicacion.
// 7. Conclusiones
// 8. Referencias (No wikipedia por ningun motivo, traten que sean de articulos cientificos)
//    El documento debe estar referenciado 

// Nota: si el documento no es presentado , no revisare su desarrollo del proyecto

//-------  S V M  -------//

def svm():Unit={ 
// we import the libraries to be able to perform the SVM
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.LinearSVC

// Error log
Logger.getLogger("org").setLevel(Level.ERROR)

// We create our spark session to start using spark
val spark = SparkSession.builder().getOrCreate()

// We must load the data from the dataset "bank-full.csv"
val data = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

// We make a vector to store the columns that will be used.
val assembler = new VectorAssembler().setInputCols(Array("age","balance","day","duration","campaign","pdays","previous")).setOutputCol("features")

// We will use stringindexer to make the data in column Y binary.
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")

// We divide the data into the training and test sets leaving only 30% of the data for testing
val Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed = 11L)

// We are going to create the linearSVM model with the label and features columns of the dataset, as well as for the prediction, with 10 iterations
val lsvc = new LinearSVC()
.setLabelCol("label")
.setFeaturesCol("features")
.setPredictionCol("prediction")
.setMaxIter(10)
.setRegParam(0.1)

// With the help of Pipeline, the estimator will be called in the input data set to fit a model, this through the indices and thus have the metadata that we require for the operation of the algorithm.
val pipeline = new Pipeline().setStages(Array(labelIndexer, assembler, lsvc))

// we adjust the model
val model = pipeline.fit(training)

// The results of the test set are taken with transform
val result = model.transform(test)

// Result of the test data set in an RDD for the prediction data
val predictionAndLabelsrdd = result.select($"prediction", $"label").as[(Double, Double)].rdd

// a MulticlassMetrics object is initialized with the previous results for some metrics
val metrics = new MulticlassMetrics(predictionAndLabelsrdd)

println("-SVM-")

// We use the metrics data to print the precision of the algorithm.
println(s"Presicion = ${(metrics.accuracy)}")

// We calculate the execution time of the algorithm.
val time = System.nanoTime
val duration = (System.nanoTime - time) / 1e9d
println("Tiempo de ejecución: " + duration)
}
svm()

//-------   D E C I S I O N   T R E E  -------//
def dt():Unit={
// we import the libraries to be able to make the decision tree
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.IndexToString
import org.apache.log4j._
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler

// Error log
Logger.getLogger("org").setLevel(Level.ERROR)

// We create our spark session to start using spark
val spark = SparkSession.builder().getOrCreate()

// We must load the data from the dataset "bank-full.csv"
val data = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

// We create the label indexer to assign column Y to the indexes.
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(data)

// We make a vector to store the columns that will be used.
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val features = assembler.transform(data)

// the dataset is identified by feature in a vector to be able to use it.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)

// We divide the data into the training and test sets leaving only 30% of the data for testing
val Array(trainingD, testD) = features.randomSplit(Array(0.7, 0.3))

// We create a DecisionTree object
val ds = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

// we create the branch to create the prediction.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// With the help of Pipeline, the estimator will be called in the input data set to fit a model, this through the indices and thus have the metadata that we require for the operation of the algorithm.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, ds, labelConverter))

// we create a training model with the training data.
val model = pipeline.fit(trainingD)

// Data transformation in the model with the test data
val predictions = model.transform(testD)


// we print the predictions.
predictions.select("predictedLabel", "y", "features").show(5)

println("- Decision Tree -")

// the tree model is generated
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"tree model:\n ${treeModel.toDebugString}")

// we calculate the precision of the model with the evaluator data
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Precision = ${(accuracy)}")

//Execution time
val time1 = System.nanoTime
val duration = (System.nanoTime - time1) / 1e9d
println("Tiempo de ejecución: " + duration)
}
dt()

//-------   L O G I S T I C    R E G R E SS I O N   -------//

def Logis():Unit={
// we import the libraries to be able to perform the logistic regression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.Pipeline
    
// Error log
Logger.getLogger("org").setLevel(Level.ERROR)
    
// We create our spark session to start using spark
val spark = SparkSession.builder().getOrCreate()
    
// We must load the data from the "bank-full.csv" dataset and make it dataframe
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
 


// We make a vector to store the columns that will be used.
val assembler = new VectorAssembler().setInputCols(Array("age","balance","day","duration","campaign","pdays","previous")).setOutputCol("features")
    
// We will use stringindexer to make the data in column Y binary.
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
val dataIndexed = labelIndexer.fit(df).transform(df)
    
// We divide the data into the training and test sets leaving only 30% of the data for testing
val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 12345)

// we instantiate the new logistic regression
val lr = new LogisticRegression()

// With the help of Pipeline, the estimator will be called in the input data set to fit a model, this through the indices and thus have the metadata that we require for the operation of the algorithm.
val pipeline = new Pipeline().setStages(Array(assembler,lr))

// we fit the model with the training data.
val model = pipeline.fit(training)

// we generate the results
val results = model.transform(test)

// we calculate the predictions of the model.
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

// We create a metric variable to have the data of the prediction evaluator
val metrics = new MulticlassMetrics(predictionAndLabels)

println("-Logistic Regression -")

// We create the confusion matrix with the predictions of the columns.
println(metrics.confusionMatrix)

// We calculate the precision of the model.
println(s"Precision = ${(metrics.accuracy)}")

// We calculate the execution time of the model.
val time2 = System.nanoTime
val duration = (System.nanoTime - time2) / 1e9d
println("Tiempo de ejecución: " + duration)
}
logis()


//-------   M U L T I L A Y E R    P E R C E P T R O N  -------//

def mulpe():Unit={
// we import the libraries to be able to perform the perceptron multilayer
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
    
// Error log
Logger.getLogger("org").setLevel(Level.ERROR)
    
// We create our spark session to start using spark
val spark = SparkSession.builder().getOrCreate()
    
// We must load the data from the "bank-full.csv" dataset and make it a dataframe.
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

// We make a vector to store the columns that are used.
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val features = assembler.transform(df)
  
// We will use stringindexer to make the data in column Y binary.
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
val dataIndexed = labelIndexer.fit(features).transform(features)
  
// We divide the data into the training and test sets leaving only 30% of the data for testing
val split(training,test) = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 1234L)

// We assign the value to the layers of our model
val layers = Array[Int](5, 2, 3, 2)

// we create the model with the given parameters.
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// We fit the model with the training data.
val model = trainer.fit(training)

// we create a variable to insert the results of our model
val result = model.transform(test)
    
// we create predictions with prediction and label columns
val predictionAndLabels = result.select("prediction", "label")
//mostramos los datos calculados
predictionAndLabels.show(10)

println("- Multilayer perceptron -")

// We calculate the precision of our model.
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Accuracy test = ${evaluator.evaluate(predictionAndLabels)}")

// we calculate the execution time
val time3 = System.nanoTime
val duration = (System.nanoTime - time3) / 1e9d
println("Tiempo de ejecución: " + duration)
}
mulpe()
