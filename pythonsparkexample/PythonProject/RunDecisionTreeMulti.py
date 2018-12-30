# -*- coding: UTF-8 -*-
import sys
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics


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
    return float(label)-1

def convert_float(x):
    return (0 if x=="?" else float(x))

def extract_features(record,featureEnd):
    numericalFeatures=[convert_float(field)  for  field in record[0: featureEnd]]
    return  numericalFeatures



def PrepareData(sc): 
    #----------------------1.导入并转换数据-------------
    print("开始导入数据...")
    rawData = sc.textFile(Path+"data/covtype.data")
    print("共计：" + str(rawData.count()) + "项")
    lines = rawData.map(lambda x: x.split(","))
    #----------------------2.建立训练评估所需数据 RDD[LabeledPoint]-------------
    labelpointRDD = lines.map(lambda r: LabeledPoint(
                                                     extract_label(r), 
                                                     extract_features(r,len(r) - 1)))
    #----------------------3.以随机方式将数据分为3个部分并且返回-------------
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print("将数据分trainData:" + str(trainData.count())+\
             "   validationData:" + str(validationData.count()) +\
             "   testData:" + str(testData.count()))
    print labelpointRDD.first()
    return (trainData, validationData, testData) 

def PredictData(sc,model): 
    #----------------------1.导入并转换数据-------------
    rawData = sc.textFile(Path+"data/covtype.data")
    print("共计：" + str(rawData.count()) + "项")
    print("建立训练评估所需数据 RDD...")
    lines = rawData.map(lambda x: x.split(","))
    #----------------------2.建立预测所需数据 RDD[LabeledPoint]-------------
    labelpointRDD = lines.map(lambda r: LabeledPoint(
                              extract_label(r), extract_features(r,len(r) - 1)))
    #----------------------3.进行预测并显示结果-------------
    for lp in labelpointRDD.take(100):
        predict = model.predict(lp.features)
        label=lp.label
        features=lp.features
        result = ("正确" if  (label == predict) else "错误")
        print("土地条件：海拔:" + str(features[0]) + 
                 " 方位:" + str(features[1]) + 
                 " 斜率:" + str(features[2]) + 
                 " 水源垂直距离:" + str(features[3]) + 
                 " 水源水平距离:" + str(features[4]) + 
                 " 9点时阴影:" + str(features[5]) + 
                 "....==>预测:" + str(predict) +
                 " 实际:" + str(label) + "结果:" + result)
 


def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels=score.zip(validationData.map(lambda p: p.label))
    metrics = MulticlassMetrics(scoreAndLabels)
    accuracy = metrics.accuracy
    return( accuracy)
 

def trainEvaluateModel(trainData,validationData,
                                          impurityParm, maxDepthParm, maxBinsParm):
    startTime = time()
    model = DecisionTree.trainClassifier(trainData,\
                                    numClasses=7, categoricalFeaturesInfo={}, \
                                    impurity=impurityParm, 
                                    maxDepth=maxDepthParm, 
                                    maxBins=maxBinsParm)
    accuracy = evaluateModel(model, validationData)
    duration = time() - startTime
    print    "训练评估：使用参数" + \
                " impurityParm= %s"%impurityParm+ \
                " maxDepthParm= %s"%maxDepthParm+ \
                " maxBinsParm = %d."%maxBinsParm + \
                 " 所需时间=%d"%duration + \
                 " 结果accuracy = %f " % accuracy 
    return (accuracy,duration, impurityParm, maxDepthParm, maxBinsParm,model)




def evalParameter(trainData, validationData, evaparm,impurityList, maxDepthList, maxBinsList):
    metrics = [trainEvaluateModel(trainData, validationData,  impurity,numIter,  maxBins  ) 
               for impurity in impurityList for numIter in maxDepthList  for maxBins in maxBinsList ]
    if evaparm=="impurity":
        IndexList=impurityList[:]
    elif evaparm=="maxDepth":
        IndexList=maxDepthList[:]
    elif evaparm=="maxBins":
        IndexList=maxBinsList[:]
    df = pd.DataFrame(metrics,index=IndexList,
               columns=['accuracy', 'duration','impurity', 'maxDepth', 'maxBins','model'])
    
    showchart(df,evaparm,'accuracy','duration',0.6,1.0 )
    
    
def showchart(df,evalparm ,barData,lineData,yMin,yMax):
    ax = df[barData].plot(kind='bar', titl =evalparm,figsize=(10,6),legend=True, fontsize=12)
    ax.set_xlabel(evalparm,fontsize=12)
    ax.set_ylim([yMin,yMax])
    ax.set_ylabel(barData,fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[[lineData ]].values, linestyle='-', marker='o', linewidth=2.0,color='r')
    plt.show()
    
    
def evalAllParameter(training_RDD, validation_RDD, impurityList, maxDepthList, maxBinsList):    
    metrics = [trainEvaluateModel(trainData, validationData,  impurity,numIter,  maxBins  ) 
                        for impurity in impurityList for numIter in maxDepthList  for maxBins in maxBinsList ]
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter=Smetrics[0]
    print("调校后最佳参数：impurity:" + str(bestParameter[2]) + 
             "  ,maxDepth:" + str(bestParameter[3]) + 
            "  ,maxBins:" + str(bestParameter[4])   + 
            "  ,结果accuracy = " + str(bestParameter[0]))
    return bestParameter[5]

def  parametersEval(training_RDD, validation_RDD):
    print("----- 评估impurity参数使用 ---------")
    evalParameter(trainData, validationData,"impurity", 
                              impurityList=["gini", "entropy"],   
                              maxDepthList=[10],  
                              maxBinsList=[10 ])  

    print("----- 评估maxDepth参数使用 ---------")
    evalParameter(trainData, validationData,"maxDepth", 
                              impurityList=["gini"],                    
                              maxDepthList=[3, 5, 10, 15, 20, 25],    
                              maxBinsList=[10])   

    print("----- 评估maxBins参数使用 ---------")
    evalParameter(trainData, validationData,"maxBins", 
                              impurityList=["gini"],      
                              maxDepthList =[10],        
                              maxBinsList=[3, 5, 10, 50, 100, 200 ])
    


def CreateSparkContext():
    sparkConf = SparkConf()                                                       \
                         .setAppName("RunDecisionTreeBinary")                         \
                         .set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print ("master="+sc.master)    
    SetLogger(sc)
    SetPath(sc)
    return (sc)

if __name__ == "__main__":
    print("RunDecisionTreeMulti")
    sc=CreateSparkContext()
    print("==========数据准备阶段===============")
    (trainData, validationData, testData) =PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    print("==========训练评估阶段===============")
    (AUC,duration, impurityParm, maxDepthParm, maxBinsParm,model)= \
        trainEvaluateModel(trainData, validationData, "entropy", 15,50)
    if (len(sys.argv) == 2) and (sys.argv[1]=="-e"):
        parametersEval(trainData, validationData)
    elif   (len(sys.argv) == 2) and (sys.argv[1]=="-a"): 
        print("-----所有参数训练评估找出最好的参数组合---------")  
        model=evalAllParameter(trainData, validationData,
                          ["gini", "entropy"],
                          [3, 5, 10, 15],
                          [3, 5, 10, 50 ])
                
    print("==========测试阶段===============")
    accuracy = evaluateModel(model, testData)
    print("使用test Data测试最佳模型,结果 accuracy:" + str(accuracy))
    print("==========预测数据===============")
    PredictData(sc, model)
    #print   model.toDebugString()
    
