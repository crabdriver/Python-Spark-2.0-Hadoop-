# -*- coding: UTF-8 -*-
import sys
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import NaiveBayes
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
    ret=(0 if x=="?" else float(x))
    return(0 if ret<0 else ret)


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
    
    print "标准化之后：",    
    stdScaler = StandardScaler(withMean=False, withStd=True).fit(featureRDD)
    ScalerFeatureRDD=stdScaler.transform(featureRDD)
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


def trainEvaluateModel(trainData,validationData,lambdaParam):
    startTime = time()
    model = NaiveBayes.train(trainData,   lambdaParam)
    AUC = evaluateModel(model, validationData)
    duration = time() - startTime
    print    "训练评估：使用参数" + \
                " lambda="+str( lambdaParam) +\
                 " 所需时间="+str(duration) + \
                 " 结果AUC = " + str(AUC) 
    return (AUC,duration,  lambdaParam,model)


def evalParameter(trainData, validationData, evalparm,
                  lambdaParamList):
    
    metrics = [trainEvaluateModel(trainData, validationData,regParam ) 
                                  for regParam in  lambdaParamList]
    
    evalparm="lambdaParam"
    IndexList=lambdaParamList
    
    df = pd.DataFrame(metrics,index=IndexList,
            columns=['AUC', 'duration',' lambdaParam','model'])
    showchart(df,evalparm,'AUC','duration',0.5,0.7 )
    
def showchart(df,evalparm ,barData,lineData,yMin,yMax):
    ax = df[barData].plot(kind='bar', title =evalparm,figsize=(10,6),legend=True, fontsize=12)
    ax.set_xlabel(evalparm,fontsize=12)
    ax.set_ylim([yMin,yMax])
    ax.set_ylabel(barData,fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[[lineData ]].values, linestyle='-', marker='o', linewidth=2.0,color='r')
    plt.show()
def evalAllParameter(training_RDD, validation_RDD, lambdaParamList):    
    metrics = [trainEvaluateModel(trainData, validationData,  lambdaParam  ) 
                        for lambdaParam in lambdaParamList  ]
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter=Smetrics[0]
    
    print("调校后最佳参数：lambdaParam:" + str(bestParameter[2]) +  
             "  ,结果AUC = " + str(bestParameter[0]))
    return bestParameter[3]

    
def  parametersEval(trainData, validationData):
    print("----- 评估lambda参数使用 ---------")
    evalParameter(trainData, validationData,"lambdaParam", 
            lambdaParamList=[1.0, 3.0, 5.0, 15.0, 25.0,30.0,35.0,40.0,45.0,50.0,60.0]) 
         


def CreateSparkContext():
    sparkConf = SparkConf()                                                       \
                         .setAppName("RunNaiveBayesBinary")                         \
                         .set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print ("master="+sc.master)    
    SetLogger(sc)
    SetPath(sc)
    return (sc)

if __name__ == "__main__":
    print("RunNaiveBayesBinary")
    sc=CreateSparkContext()
    print("==========数据准备阶段===============")
    (trainData, validationData, testData, categoriesMap) =PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    print("==========训练评估阶段===============")
    
    (AUC,duration,  lambdaParam,model)= \
            trainEvaluateModel(trainData, validationData, 60.0)
          
    if (len(sys.argv) == 2) and (sys.argv[1]=="-e"):
        parametersEval(trainData, validationData)
    elif   (len(sys.argv) == 2) and (sys.argv[1]=="-a"): 
        print("-----所有参数训练评估找出最好的参数组合---------")  
        model=evalAllParameter(trainData, validationData, 
                           [1.0, 3.0, 5.0, 15.0, 25.0,30.0,35.0,40.0,45.0,50.0,60.0])

              
    print("==========测试阶段===============")
    auc = evaluateModel(model, testData)
    print("使用test Data测试最佳模型,结果 AUC:" + str(auc))
    print("==========预测数据===============")
    PredictData(sc, model, categoriesMap)
    
