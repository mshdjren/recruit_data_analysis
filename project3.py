import cluster
import random, pylab, numpy
import os
import matplotlib.pyplot as plt

#get currnet path
currentpath = os.getcwd()
print(currentpath)
#change path
os.chdir("c:/Users/msh10/Desktop/학교/4학년 1학기/센서빅데이터/강의노트(코드)/Lecture12/Lecture12/")

class Patient(cluster.Example):
    pass

def scaleAttrs(vals):
    vals = pylab.array(vals)
    mean = sum(vals)/len(vals)
    sd = numpy.std(vals)
    vals = vals - mean
    return vals/sd

def getData(toScale = True):
    #read in data
    year, major, getjob, graduateschool, etc1, etc2, collegetype = [],[],[],[],[],[],[]
    cardiacData = open("졸업생의취업현황(대학,전문대학)_수정.csv", 'r', encoding='UTF8')
    next(cardiacData)
    for l in cardiacData:
        l = l.split(',')
        year.append(int(l[0]))
        collegetype.append((l[4]))
        major.append(str(l[5]))
        getjob.append(int(l[9]))
        graduateschool.append(int(l[10]))
        etc1.append(int(l[11]))
        etc2.append(int(l[12]))
    if toScale:
        getjob = scaleAttrs(getjob)
        graduateschool = scaleAttrs(graduateschool)
        etc1 = scaleAttrs(etc1)
        etc2 = scaleAttrs(etc2)
    #Build points
    points = []
    for i in range(len(getjob)):
        features = pylab.array([getjob[i], etc2[i],\
                                graduateschool[i], etc1[i]])
        pIndex = str(i)
        points.append(Patient('P'+ pIndex, features, major[i], year[i], collegetype[i]))
    return points
    
def kmeans(examples, k, verbose = False):
    #Get k randomly chosen initial centroids, create cluster for each
    initialCentroids = random.sample(examples, k)
    clusters = []
    for e in initialCentroids:
        clusters.append(cluster.Cluster([e]))

    #Iterate until centroids do not change
    converged = False
    numIterations = 0
    while not converged:
        numIterations += 1
        #Create a list containing k distinct empty lists
        newClusters = []
        for i in range(k):
            newClusters.append([])
            
        #Associate each example with closest centroid
        for e in examples:
            #Find the centroid closest to e
            smallestDistance = e.distance(clusters[0].getCentroid())
            index = 0
            for i in range(1, k):
                distance = e.distance(clusters[i].getCentroid())
                if distance < smallestDistance:
                    smallestDistance = distance
                    index = i
            #Add e to the list of examples for appropriate cluster
            newClusters[index].append(e)
            
        for c in newClusters: #Avoid having empty clusters
            if len(c) == 0:
                raise ValueError('Empty Cluster')
        
        #Update each cluster; check if a centroid has changed
        converged = True
        for i in range(k):
            if clusters[i].update(newClusters[i]) > 0.0:
                converged = False
        if verbose:
            print('Iteration #' + str(numIterations))
            for c in clusters:
                print(c)
            print('') #add blank line
    return clusters

def trykmeans(examples, numClusters, numTrials, verbose = False):
    """Calls kmeans numTrials times and returns the result with the
          lowest dissimilarity"""
    best = kmeans(examples, numClusters, verbose)
    minDissimilarity = cluster.dissimilarity(best)
    trial = 1
    while trial < numTrials:
        try:
            clusters = kmeans(examples, numClusters, verbose)
        except ValueError:
            continue #If failed, try again
        currDissimilarity = cluster.dissimilarity(clusters)
        if currDissimilarity < minDissimilarity:
            best = clusters
            minDissimilarity = currDissimilarity
        trial += 1
    return best

def printClustering(clustering):
    """Assumes: clustering is a sequence of clusters
       Prints information about each cluster
       Returns list of fraction of pos cases in each cluster"""
    posFracs = []
    similiarity = []
    for c in clustering:
        numPts = 0
        numPos = 0
        for p in c.members():
            numPts += 1
            similiarity.append(p.getLabel())
        for i in range(len(similiarity)):
            for j in range(len(similiarity)-1, i, -1):
                if similiarity[i] == similiarity[j]:
                    numPos += 1                 
        fracPos = numPos/((len(similiarity)*(len(similiarity)-1))/2)
        posFracs.append(fracPos)
        # print('Cluster of size', numPts, 'with similarity of College type =',
        #        fracPos)
    print(numpy.mean(posFracs))
    return pylab.array(posFracs)

def testClusteringAll(patients, numClusters, seed = 0, numTrials = 1):
    random.seed(seed)
    bestClustering = trykmeans(patients, numClusters, numTrials)
    for i in range(numClusters):
        x= []
        y= []
        for j in bestClustering[i].examples:
            x.append(j.getMajor())
            y.append(j.getFeatures_add())
        # plt.scatter(x,y,label = 'cluster' + str(i))
    # plt.title(str(2014)+'~'+str(2018))
    # plt.xlabel('학과명(전공)')
    # plt.ylabel('features의 합')
    # plt.legend()
    # plt.xticks(rotation=90)
    # plt.show()        
    posFracs = printClustering(bestClustering)
    return posFracs

def testClusteringEach(year, patients, numClusters, seed = 0, numTrials = 1):
    random.seed(seed)
    bestClustering = trykmeans(patients, numClusters, numTrials)

    for i in range(numClusters):
        x= []
        y= []
        for j in bestClustering[i].examples:
            if j.getYear() == year:
                x.append(j.getMajor())
                y.append(j.getFeatures_add())
        # plt.scatter(x,y,label = 'cluster' + str(i))
    # plt.title(year)
    # plt.xlabel('학과명(전공)')
    # plt.ylabel('features의 합')
    # plt.legend()
    # plt.xticks(rotation=90)
    # plt.show()
    posFracs = printClustering(bestClustering)
    return posFracs
 
patients = getData()
for k in (2,5,10,20):
    print('\n     Test k-means (k = ' + str(k) + ')')
    posFracsALl = testClusteringAll(patients, k, 2)
# for k in (2,5,9,10,20):
#     print('k= ' + str(k) + ' each cluster of similarity')
#     posFracs2014 = testClusteringEach(2014,patients, k, 2)

#numPos = 0
#for p in patients:
#    if p.getLabel() == 1:
#        numPos += 1
#print('Total number of positive patients =', numPos)