# -*- coding: UTF-8 -*-
from math import sqrt
from operator import add
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.mllib.recommendation import ALS
from pyspark import SparkConf, SparkContext


#import logging

def SetLogger( sc ):
    logger = sc._jvm.org.apache.log4j
    sc.setLogLevel("FATAL")
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
    
def PrepareData(sc): 
    #----------------------1. 建立用户评价数据-------------
    rawUserData = sc.textFile(Path+"data/u.data")
    print "数据项数"+str(rawUserData.count())
    rawRatings = rawUserData.map(lambda line: line.split("\t")[:3] )
    ratingsRDD = rawRatings.map(lambda fields: (fields[0],fields[1],fields[2]))
    #----------------------2. 以随机方式将数据分为3 个分并且返回-------------
    trainData, validationData, testRDD =ratingsRDD.randomSplit([8, 1, 1], seed=0L)
    print(" trainData:" + str(trainData.count()) +  
           " validationData:" + str(validationData.count()) + 
           " testData:" + str(testRDD.count()))
    return(trainData, validationData, testRDD)

def computeRMSE(alsmodel, RatingRDD):
    n=RatingRDD.count()
    predictRDD= alsmodel.predictAll(RatingRDD.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = \
        predictRDD.map(lambda x: ((x[0], x[1]), x[2])) \
            .join(RatingRDD.map(lambda x: ((float(x[0]), float(x[1])), float(x[2]))) )\
            .values()
        
    return sqrt(predictionsAndRatings \
            .map(lambda x: (x[0] - x[1]) ** 2) \
            .reduce(add)  / float(n))

         

def trainModel(trainData, validationData, rank, iterations, lambdaParm):
    startTime = time()
    model = ALS.train(trainData, rank, iterations, lambdaParm)
    Rmse = computeRMSE(model, validationData)
    duration = time() - startTime

    print    "训练评估：使用参数" + \
                "rank="+str(rank)+ \
                "lambda="+str(lambdaParm) +\
                "iterations="+ str(iterations)  + \
                "所需时间="+str(duration) + \
               "结果Rmse " +str(Rmse)
                       
    return (Rmse,duration, rank, iterations, lambdaParm,model)


def evalParameter(trainData, validationData, evaparm,
                  rankList, numIterationsList, lambdaList):
    metrics = [trainModel(trainData, validationData,  rank,numIter,  lambas ) 
                       for rank in rankList 
                       for numIter in numIterationsList  
                       for lambas in lambdaList ]
    if evaparm=="rank":
        IndexList=rankList[:]
    elif evaparm=="numIterations":
        IndexList=numIterationsList[:]
    elif evaparm=="lambda":
        IndexList=lambdaList[:]
    df = pd.DataFrame(metrics,index=IndexList,columns=
         ['RMSE', 'duration' , 'rank', 'iterations', 'lambdaParm','model'])
    showchart(df,evaparm,'RMSE','duration',0.8,5)

def showchart(df,evalparm ,barData,lineData,yMin,yMax):
    ax = df[barData].plot(kind='bar', title =evalparm,figsize=(10,6),legend=True, fontsize=12)
    ax.set_xlabel(evalparm,fontsize=12)
    ax.set_ylim([yMin,yMax])
    ax.set_ylabel(barData,fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[[lineData ]].values, linestyle='-', marker='o', linewidth=2.0,color='r')
    plt.show()
    
def evalAllParameter(trainData, validationData, rankList, numIterationsList, lambdaList):    
    metrics = [trainModel(trainData, validationData,  rank,numIter,  lambas  )  
                      for rank in rankList for numIter in numIterationsList  for lambas in lambdaList ]
    Smetrics = sorted(metrics, key=lambda k: k[0])
    print 'Best:'
    bestParameter=Smetrics[0]
    print bestParameter
       
    print("调校后最佳参数：rank:" + str(bestParameter[2]) + 
              "  ,numIterations:" + str(bestParameter[3]) + 
              "  ,lambda:" + str(bestParameter[4])   + 
              "  ,结果RMSE = " + str(bestParameter[0]))
    
    return bestParameter[5]
     

def  parametersTunning(trainData, validationData):

    print("----- 评估rank参数使用 5,10,15,20,50,100---------")
    evalParameter(trainData, validationData,"rank", 
            rankList=[5,10,15,20,50,100], 
            numIterationsList=[10],        
            lambdaList=[1.0 ])      

    print("----- 评估numIterations参数使用 5,10,15,20,25---------")
    evalParameter(trainData, validationData,"numIterations", 
            rankList=[8],                     
            numIterationsList=[5,10,15,20,25],     
            lambdaList=[1.0])       
    
    print("----- 评估lambda参数使用 0.05,0.1,1.0,5.0,10.0---------")
    evalParameter(trainData, validationData,"lambda", 
            rankList=[8],                 
            numIterationsList=[10],     
            lambdaList=[0.05,0.1,1.0,5.0,10.0 ])   

    print("-----所有参数训练评估找出最好的参数组合---------")   
    bestModel=evalAllParameter(
         trainData, validationData,
         rankList=[5,10,15,20],
         numIterationsList=[5, 10, 15, 20,30],
         lambdaList=[0.05,0.1,1.0,5.0 ])


    return bestModel

def SaveModel(model,sc): 
    try:        
        model.save(sc,Path+"ALSmodel")
        print("已存储 Model 在ALSmodel")
    except Exception :
        print "Model已经存在,先删除."      

def CreateSparkContext():
    sparkConf = SparkConf()                                                       \
                         .setAppName("RecommendTrainEval")                         \
                         .set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print ("master="+sc.master)    
    SetLogger(sc)
    SetPath(sc)
    return (sc)
          

if __name__ == "__main__":
    sc=CreateSparkContext()
    print("========== 数据准备阶段===============")
    (trainData, validationData, testRDD)=PrepareData(sc)
    trainData.persist() ;    validationData.persist();    testRDD.persist()
    print("========== 训练评估阶段===============")
    bestModel = parametersTunning(trainData, validationData)
    print("========== 测试阶段===============")
    testRmse = computeRMSE(bestModel, testRDD)
    print(" 使用test Data 测试bestModel," + " 结果rmse = " + str(testRmse))
    print("========== 存储Model========== ==")
    SaveModel(bestModel ,sc)
    print("saved")
    trainData.unpersist(); validationData.unpersist(); testRDD.unpersist()
    print("persisted")
    
