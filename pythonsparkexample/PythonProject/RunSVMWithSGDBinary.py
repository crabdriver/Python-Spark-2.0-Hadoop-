# -*- coding: UTF-8 -*-
import sys
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.feature import StandardScaler


def SetLogger( sc ):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)    

def SetPath(sc):
    global Path
    if sc.master[0:5]=="local" :
        Path="file:/home/hduser/pythonsparkexample/PythonProject/"
    else:   
        Path="hdfs://master:9000/user/hduser/"
#如果要在cluster模式运行(hadoop yarn 或Spark Stand alone)，请按照书上的说明，先把文件上传到HDFS目录
        
def get_mapping(rdd, idx):
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()

def extract_label(record):
    label=(record[-1])
    return float(label)

def extract_features(field,categoriesMap,featureEnd):
    categoryIdx = categoriesMap[field[3]]
    categoryFeatures = np.zeros(len(categoriesMap))
    categoryFeatures[categoryIdx] = 1
    numericalFeatures=[convert_float(field)  for  field in field[4: featureEnd]]    
    return  np.concatenate(( categoryFeatures, numericalFeatures))

def convert_float(x):
    return (0 if x=="?" else float(x))

def PrepareData(sc): 
    #----------------------1.导入并转换数据-------------
    print("开始导入数据...")
    rawDataWithHeader = sc.textFile(Path+"data/train.tsv")
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)    
    rData=rawData.map(lambda x: x.replace("\"", ""))    
    lines = rData.map(lambda x: x.split("\t"))
    print("共计：" + str(lines.count()) + "项")
    #----------------------2.建立训练评估所需数据 RDD[LabeledPoint]-------------
    print "标准化之前：",
    categoriesMap = lines.map(lambda fields: fields[3]). \
                                        distinct().zipWithIndex().collectAsMap()
    labelRDD = lines.map(lambda r:  extract_label(r))
    featureRDD = lines.map(lambda r:  extract_features(r,categoriesMap,len(r) - 1))

    for i in featureRDD.first():
        print (str(i)+","),
    print ""       
    
    stdScaler = StandardScaler(withMean=True, withStd=True).fit(featureRDD)
    ScalerFeatureRDD=stdScaler.transform(featureRDD)
    print "标准化之后：",
    for i in ScalerFeatureRDD.first():
        print (str(i)+","),        
    labelpoint=labelRDD.zip(ScalerFeatureRDD)
    labelpointRDD=labelpoint.map(lambda r: LabeledPoint(r[0], r[1]))

    
    #----------------------3.以随机方式将数据分为3个部分并且返回-------------
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print("将数据分trainData:" + str(trainData.count()) + 
              "   validationData:" + str(validationData.count()) +
              "   testData:" + str(testData.count()))
    return (trainData, validationData, testData, categoriesMap) #返回数据

    
def PredictData(sc,model,categoriesMap): 
    print("开始导入数据...")
    rawDataWithHeader = sc.textFile(Path+"data/test.tsv")
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)    
    rData=rawData.map(lambda x: x.replace("\"", ""))    
    lines = rData.map(lambda x: x.split("\t"))
    print("共计：" + str(lines.count()) + "项")
    dataRDD = lines.map(lambda r:  ( r[0]  ,
                            extract_features(r,categoriesMap,len(r) )))
    DescDict = {
           0: "暂时性网页(ephemeral)",
           1: "长青网页(evergreen)"
     }
    for data in dataRDD.take(10):
        predictResult = model.predict(data[1])
        print " 网址：  " +str(data[0])+"\n" +\
                  "             ==>预测:"+ str(predictResult)+ \
                  " 说明:"+DescDict[predictResult] +"\n"

def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels=score.zip(validationData \
                                   .map(lambda p: p.label))  \
                                   .map(lambda (x,y): (float(x),float(y)) )
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUC=metrics.areaUnderROC
    return( AUC)


def trainEvaluateModel(trainData,validationData,
                                        numIterations, stepSize, regParam):
    startTime = time()
    model = SVMWithSGD.train(trainData, numIterations, stepSize, regParam)
    AUC = evaluateModel(model, validationData)
    duration = time() - startTime
    print    "训练评估：使用参数" + \
                " numIterations="+str(numIterations) +\
                " stepSize="+str(stepSize) + \
                " regParam="+str(regParam) +\
                 " 所需时间="+str(duration) + \
                 " 结果AUC = " + str(AUC) 
    return (AUC,duration, numIterations, stepSize, regParam,model)


def evalParameter(trainData, validationData, evalparm,
                  numIterationsList, stepSizeList, regParamList):
    
    metrics = [trainEvaluateModel(trainData, validationData,  
                                numIterations,stepSize,  regParam  ) 
                       for numIterations in numIterationsList
                       for stepSize in stepSizeList  
                       for regParam in regParamList ]
    
    if evalparm=="numIterations":
        IndexList=numIterationsList[:]
    elif evalparm=="stepSize":
        IndexList=stepSizeList[:]
    elif evalparm=="regParam":
        IndexList=regParamList[:]
    
    df = pd.DataFrame(metrics,index=IndexList,
            columns=['AUC', 'duration','numIterations', 'stepSize', 'regParam','model'])
    showchart(df,evalparm,'AUC','duration',0.5,0.7 )
    
def showchart(df,evalparm ,barData,lineData,yMin,yMax):
    ax = df[barData].plot(kind='bar', title =evalparm,figsize=(10,6),legend=True, fontsize=12)
    ax.set_xlabel(evalparm,fontsize=12)
    ax.set_ylim([yMin,yMax])
    ax.set_ylabel(barData,fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[[lineData ]].values, linestyle='-', marker='o', linewidth=2.0,color='r')
    plt.show()
    
def evalAllParameter(trainData, validationData, 
                     numIterationsList, stepSizeList, regParamList):    
    metrics = [trainEvaluateModel(trainData, validationData,  
                            numIterations,stepSize,  regParam  ) 
                      for numIterations in numIterationsList 
                      for stepSize in stepSizeList  
                      for  regParam in regParamList ]
    
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter=Smetrics[0]
    
    print("调校后最佳参数：numIterations:" + str(bestParameter[2]) + 
                                      "  ,stepSize:" + str(bestParameter[3]) + 
                                     "  ,regParam:" + str(bestParameter[4])   + 
                                      "  ,结果AUC = " + str(bestParameter[0]))
    
    return bestParameter[5]

def  parametersEval(trainData, validationData):
    print("----- 评估numIterations参数使用 ---------")
    evalParameter(trainData, validationData,"numIterations", 
                              numIterationsList= [1, 3, 5, 15, 25],   
                              stepSizeList=[100],  
                              regParamList=[1 ])  
    print("----- 评估stepSize参数使用 ---------")
    evalParameter(trainData, validationData,"stepSize", 
                              numIterationsList=[25],                    
                              stepSizeList= [10, 50, 100, 200],    
                              regParamList=[1])   
    print("----- 评估regParam参数使用 ---------")
    evalParameter(trainData, validationData,"regParam", 
                              numIterationsList=[25],      
                              stepSizeList =[100],        
                              regParamList=[0.01, 0.1, 1 ])

def CreateSparkContext():
    sparkConf = SparkConf()                                                       \
                         .setAppName("LogisticRegressionWithSGD")                         \
                         .set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print ("master="+sc.master)    
    SetLogger(sc)
    SetPath(sc)
    return (sc)

if __name__ == "__main__":
    print("RunSVMWithSGDBinary")
    sc=CreateSparkContext()
    print("==========数据准备阶段===============")
    (trainData, validationData, testData, categoriesMap) =PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    print("==========训练评估阶段===============")
    (AUC,duration, numIterations, stepSize, regParam,model)= \
          trainEvaluateModel(trainData, validationData, 3, 50, 1)
    if (len(sys.argv) == 2) and (sys.argv[1]=="-e"):
        parametersEval(trainData, validationData)
    elif   (len(sys.argv) == 2) and (sys.argv[1]=="-a"): 
        print("-----所有参数训练评估找出最好的参数组合---------")  
        model=evalAllParameter(trainData, validationData,
                        [1, 3, 5, 15, 25], 
                        [10, 50, 100, 200],
                        [0.01, 0.1, 1 ])
    print("==========测试阶段===============")
    auc = evaluateModel(model, testData)
    print("使用test Data测试最佳模型,结果 AUC:" + str(auc))
    print("==========预测数据===============")
    PredictData(sc, model, categoriesMap)
    
