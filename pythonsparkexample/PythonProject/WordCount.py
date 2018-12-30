# -*- coding: UTF-8 -*-
from pyspark import SparkContext
from pyspark import SparkConf

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

def CreateSparkContext():
    sparkConf = SparkConf()                                                       \
                         .setAppName("WordCounts")                         \
                         .set("spark.ui.showConsoleProgress", "false") \
              
    sc = SparkContext(conf = sparkConf)
    print("master="+sc.master)
    SetLogger(sc)
    SetPath(sc)
    return (sc)

    

if __name__ == "__main__":
    print("开始运行RunWordCount")
    sc=CreateSparkContext()
 
    print("开始读取文本文件...")
    textFile = sc.textFile(Path+"data/README.md")
    print("文本文件共"+str(textFile.count())+"行")
     
    countsRDD = textFile                                     \
                  .flatMap(lambda line: line.split(' ')) \
                  .map(lambda x: (x, 1))                    \
                  .reduceByKey(lambda x,y :x+y)
                  
    print("文字统计共"+str(countsRDD.count())+"项数据")                  
    print("开始存储到文本文件...")
    try:
        countsRDD.saveAsTextFile(Path+ "data/output")
        
    except Exception as e:
        print("输出目录已经存在,请先删除原有目录")
    sc.stop()
