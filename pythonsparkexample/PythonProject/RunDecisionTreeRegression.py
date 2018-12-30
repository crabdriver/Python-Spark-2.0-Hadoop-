# -*- coding: UTF-8 -*-
import sys
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.evaluation import RegressionMetrics
import math 
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

def extract_label(record):
    label=(record[-1])
    return float(label)

def convert_float(x):
    return (0 if x=="?" else float(x))

def extract_features(record,featureEnd):
    featureSeason=[convert_float(field)  for  field in record[2]] 
    features=[convert_float(field)  for  field in record[4: featureEnd-2]]
    return  np.concatenate( (featureSeason, features))

def PrepareData(sc): 
    #----------------------1.导入并转换数据-------------
    print("开始导入数据...")
    rawDataWithHeader = sc.textFile(Path+"data/hour.csv")
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)    
    lines = rawData.map(lambda x: x.split(","))
    print (lines.first())
    print("共计：" + str(lines.count()) + "项")
    #----------------------2.建立训练评估所需数据 RDD[LabeledPoint]-------------
    labelpointRDD = lines.map(lambda r:LabeledPoint(
                                                    extract_label(r), 
                                                    extract_features(r,len(r) - 1)))

    print labelpointRDD.first()
    #----------------------3.以随机方式将数据分为3个部分并且返回-------------
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print("将数据分trainData:" + str(trainData.count()) + 
             "   validationData:" + str(validationData.count()) +
             "   testData:" + str(testData.count()))
    #print labelpointRDD.first()
    return (trainData, validationData, testData) #返回数据

def PredictData(sc,model): 
    #----------------------1.导入并转换数据-------------
    print("开始导入数据...")
    rawDataWithHeader = sc.textFile(Path+"data/hour.csv")
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)    
    lines = rawData.map(lambda x: x.split(","))
    #print (lines.first())
    print("共计：" + str(lines.count()) + "项")
    #----------------------2.建立训练评估所需数据 LabeledPoint RDD-------------
    labelpointRDD = lines.map(lambda r: LabeledPoint(
                                                     extract_label(r), 
                                                     extract_features(r,len(r) - 1)))
    #----------------------3.定义字典----------------
    SeasonDict = { 1 : "春",  2 : "夏",  3 :"秋",  4 : "冬"   }
    HoildayDict={  0 : "非假日", 1 : "假日"  }  
    WeekDict = {0:"一",1:"二",2:"三",3:"四",4 :"五",5:"六",6:"日"}
    WorkDayDict={ 1 : "工作日",  0 : "非工作日"  }
    WeatherDict={ 1 : "晴",  2 : "阴",  3 : "小雨", 4 : "大雨" }
    #----------------------4.进行预测并显示结果--------------
    for lp in labelpointRDD.take(100):
        predict = int(model.predict(lp.features))
        label=lp.label
        features=lp.features
        result = ("正确" if  (label == predict) else "错误")
        error = math.fabs(label - predict)
        dataDesc="  特征: "+SeasonDict[features[0]] +"季,"+\
                            str(features[1]) + "月," +\
                            str(features[2]) +  "时,"+ \
                            HoildayDict[features[3]] +","+\
                            "星期"+WeekDict[features[4]]+","+ \
                            WorkDayDict[features[5]]+","+\
                            WeatherDict[features[6]]+","+\
                            str(features[7] * 41)+ "度,"+\
                            "体感" + str(features[8] * 50) + "度," +\
                            "湿度" + str(features[9] * 100) + ","+\
                            "风速" + str(features[10] * 67) +\
                            " ==> 预测结果:" + str(predict )+\
                            "  , 实际:" + str(label) + result +",  误差:" + str(error)
        print dataDesc
        

    

def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels=score.zip(validationData.map(lambda p: p.label))
    metrics = RegressionMetrics(scoreAndLabels)
    RMSE=metrics.rootMeanSquaredError
    return( RMSE)
 

def trainEvaluateModel(trainData,validationData,
                                           impurityParm, maxDepthParm, maxBinsParm):
    startTime = time()
    model = DecisionTree.trainRegressor(trainData, 
                                  categoricalFeaturesInfo={}, \
                                  impurity=impurityParm, 
                                  maxDepth=maxDepthParm, 
                                  maxBins=maxBinsParm)
    RMSE = evaluateModel(model, validationData)
    duration = time() - startTime
    print    "训练评估：使用参数" + \
                " impurityParm= %s"%impurityParm+ \
                "  maxDepthParm= %s"%maxDepthParm+ \
                "  maxBinsParm = %d."%maxBinsParm + \
                 "  所需时间=%d"%duration + \
                 "  结果RMSE = %f " % RMSE 
    return (RMSE,duration, impurityParm, maxDepthParm, maxBinsParm,model)


def evalParameter(trainData, validationData, evaparm,impurityList, maxDepthList, maxBinsList):
    metrics = [trainEvaluateModel(trainData, validationData,  impurity,maxdepth,  maxBins  ) 
                            for impurity in impurityList 
                            for maxdepth in maxDepthList  
                            for maxBins in maxBinsList ]
    if evaparm=="impurity":
        IndexList=impurityList[:]
    elif evaparm=="maxDepth":
        IndexList=maxDepthList[:]
    elif evaparm=="maxBins":
        IndexList=maxBinsList[:]

    df = pd.DataFrame(metrics,index=IndexList,
                      columns=['RMSE', 'duration','impurityParm', 'maxDepthParm', 'maxBinsParm','model'])
    
    showchart(df,evaparm,'RMSE','duration',0,200 )
    
    
def showchart(df,evalparm ,barData,lineData,yMin,yMax):
    ax = df[barData].plot(kind='bar', title =evalparm,figsize=(10,6),legend=True, fontsize=12)
    ax.set_xlabel(evalparm,fontsize=12)
    ax.set_ylim([yMin,yMax])
    ax.set_ylabel(barData,fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[[lineData ]].values, linestyle='-', marker='o', linewidth=2.0,color='r')
    plt.show()
    
def evalAllParameter(training_RDD, validation_RDD, impurityList, maxDepthList, maxBinsList):    
    metrics = [trainEvaluateModel(trainData, validationData,  impurity,maxdepth,  maxBins  ) 
                        for impurity in impurityList 
                        for maxdepth in maxDepthList  
                        for maxBins in maxBinsList ]
    Smetrics = sorted(metrics, key=lambda k: k[0])
    bestParameter=Smetrics[0]
    
    print("调校后最佳参数：impurity:" + str(bestParameter[2]) + 
            "  ,maxDepth:" + str(bestParameter[3]) + 
            "  ,maxBins:" + str(bestParameter[4])   + 
            "  ,结果RMSE = " + str(bestParameter[0]))
    
    return bestParameter[5]

def  parametersEval(training_RDD, validation_RDD):
    
    print("----- 评估maxDepth参数使用 ---------")
    evalParameter(training_RDD, validation_RDD,"maxDepth", 
                              impurityList=["variance"],       
                              maxDepthList =[3, 5, 10, 15, 20, 25]  ,
                              maxBinsList=[10])
    print("----- 评估maxBins参数使用 ---------")
    evalParameter(training_RDD, validation_RDD,"maxBins", 
                              impurityList=["variance"],        
                              maxDepthList=[10],                   
                              maxBinsList=[3, 5, 10, 50, 100, 200 ])  



def CreateSparkContext():
    sparkConf = SparkConf()                                            \
                         .setAppName("RunDecisionTreeRegression")           \
                         .set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print ("master="+sc.master)    
    SetLogger(sc)
    SetPath(sc)
    return (sc)

if __name__ == "__main__":
    print("RunDecisionTreeRegression")
    sc=CreateSparkContext()
    print("==========数据准备阶段===============")
    (trainData, validationData, testData) =PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    print("==========训练评估阶段===============")
    (AUC,duration, impurityParm, maxDepthParm, maxBinsParm,model)= \
             trainEvaluateModel(trainData, validationData, "variance", 10, 100)
    if (len(sys.argv) == 2) and (sys.argv[1]=="-e"):
        parametersEval(trainData, validationData)
    elif   (len(sys.argv) == 2) and (sys.argv[1]=="-a"): 
        print("-----所有参数训练评估找出最好的参数组合---------")  
        model=evalAllParameter(trainData, validationData,
                          ["variance"],
                          [3, 5, 10, 15, 20, 25], 
                          [3, 5, 10, 50, 100, 200 ])
    print("==========测试阶段===============")
    RMSE = evaluateModel(model, testData)
    print("使用test Data测试最佳模型,结果 RMSE:" + str(RMSE))
    print("==========预测数据===============")
    PredictData(sc, model)
    #print   model.toDebugString()
    
    
    
