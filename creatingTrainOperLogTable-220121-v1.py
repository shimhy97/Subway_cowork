import pandas as pd
from pandas import DataFrame as df

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

import urllib.request
import urllib.parse

import json
import os
from datetime import datetime
import datetime as dt
import time
import logging
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import statistics as st
import math as math

from scipy import stats

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import variance_inflation_factor

#from pylab import figure, axes, pie, title, show
from sys import argv

#script, first = argv

from collections import Counter


def findModeFromList(candidates): # 최빈값 찾기
    assert isinstance(candidates, list), 'Must be a list type'
    if len(candidates) == 0: return None
    return Counter(candidates).most_common(n=1)[0][0]

def findPcNm(mystr):
    mystr # = "2019-10-13_Line1_rtTrainPos_merged_4356-files_utlsyslab_jh1_admin.csv"
    end = mystr.find("_admin.csv")
    start = mystr.find("_",end-4,end)+1 # mystr의 end-4번째 글짜에서 end 번째 글짜 사이에서 "_"의 위치를 찾음
    if start==0 :
        myPcNm = "0"
    elif start > 0 :
        myPcNm = mystr[start:end]
    else :
        print('pcNm has out of range:' , mystr)
    print("pc_"+myPcNm)

    return myPcNm

def findNumberOfFiles(mystr):
    mystr# = "2019-10-13_Line1_rtTrainPos_merged_4356-files_utlsyslab_jh1_admin.csv"
    end = mystr.find("-files")
    start = mystr.find("_",end-5,end)+1
    myNfiles = mystr[start:end]
    print(myNfiles+"_files")

    return myNfiles

def creatingOperLogTable(myTotalData, trainSttus):
    mydata = myTotalData[myTotalData.trainSttus==trainSttus] # 입력받은 열차상태에 대한 데이터 추출

    if len(mydata.index) < 1: # 발췌한 열차상태 데이터 갯수 체크
        # 발췌한 열차상태 데이터가 1개보다 작다면
        mydata_ttable = len(mydata.index)          # 데이터 행 갯수를 return
        return mydata_ttable

    else: # 발췌한 열차상태 데이터가 1개 이상이라면
        mydata = rtPosDuplicatedDataCheck(mydata) #데이터 중복 행 검사 및 제거
        
        mydata_ttable = mydata.pivot(index='trainNo2', columns='statnId', values='recptnDt') # 열차ID와 역ID 기준으로 pivottable 만들기
        mydata_ttable = mydata_ttable.sort_values(by='trainNo2', ascending=True)

        return mydata_ttable

def rtPosDuplicatedDataCheck(needToCheck):
    print("Duplication rtPos data checking process had been started")
    mytrain_dup_checked = df()
    run_data = needToCheck

    for i in run_data.trainNo2.unique():
        mytrain = run_data[run_data.trainNo2==i] #열차번호 i에 해당하는 도착데이터를 mytrain에 저장
        n_statnId = len(mytrain.statnId.unique()) #i열차 데이터의 역 갯수를 확인
        n_row = len(mytrain.index) #i열차 데이터의 행 갯수 저장

        if n_row > n_statnId: #i열차의 데이터 갯수와 역 갯수 비교 : mytrain은 열차 고유번호 부여 후의 데이터에서 도착데이터만 추출한 것이므로 행갯수와 역 갯수가 같아야 함
            #print("trainNo2:", i, "has duplication : nrow-", n_row, ", n_statnId-", n_statnId)
            #print(mytrain[['recptnDt', 'statnNm', 'trainNo2']])
            mytrain2 = mytrain.drop_duplicates(["statnId"]) #중복행 제거
            #print("--- Duplicated rows had been eliminated.---")
            #print(mytrain2[['recptnDt', 'statnNm', 'trainNo2']])
            mytrain_dup_checked = pd.concat([mytrain_dup_checked, mytrain2], axis=0)

        else:
            mytrain_dup_checked = pd.concat([mytrain_dup_checked, mytrain], axis=0)
    
    mytrain_dup_checked.reset_index(drop=True, inplace=True)
    needToCheck = mytrain_dup_checked
    return needToCheck

def fitTrainsAndStations(ttable, merged_trains, merged_statnId): #, trainsInfo):
    ttable = pd.concat([merged_trains, ttable], axis=1)
    #print("\n==ttable 1 \n")
    #print(ttable.iloc[0:5,])

    fittingStatnId = df()
    for station in merged_statnId:
        statnExist = np.where(ttable.columns==station)
        if len(statnExist[0])>0:
            fittingStatnId = pd.concat([fittingStatnId, ttable[station]], axis=1)
        else:
            #print(statnExist[0])
            nanCol = np.repeat(pd.NaT, [len(ttable.index)], axis=0)
            emptyStatnData = df(nanCol, index=ttable.index, columns=[station])
            fittingStatnId = pd.concat([fittingStatnId, emptyStatnData], axis=1)
        del(statnExist)
    
    ttable = fittingStatnId

    #ttable = pd.merge(trainsInfo, ttable, left_on='trainNo2', right_on=ttable.index)
    #print("\n==ttable 2 \n")
    #print(ttable.iloc[0:5,])

    # removing index Name
    ttable.index.name = None

    return ttable

def time25convert(x, tTableDate):
    hour = int(x[0:2])
    if hour >= 24:
        x = tTableDate+" "+str(int(hour)-24)+x[2:len(x)]
        convertedTime = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        convertedTime = convertedTime + dt.timedelta(days=1)

    else :
        if x == "00:00:00":
            convertedTime = "NaN"
        else : 
            x = tTableDate+" "+x
            convertedTime = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    
    return convertedTime

def checkRawdataFileList(lineNm, dataType, dataPath, mydir, dataByDayByPc):

    rawFiles = len(mydir) # 파일 갯수 추출
    print("Total number of data in the folder: "+str(rawFiles))

    # 2대 이상의 pc에서 수집된 일일 데이터를 병합하기 위해 날짜 별 데이터 정보를 정리하는 코드 생성
    
    for date in range(0,(rawFiles)): 

        # 주어진 column 이름에 대해 빈 데이터프레임 만들기
        # 참고 : https://specialscene.tistory.com/43
        tempData = df(index=range(0,1), columns=['oDate', 'dataType', 'lineNm', 'pcNm', 'nFiles'])
        
        # 폴더 내 파일이름으로 부터 정보를 추출
        oDate = mydir[date][0:10] # YYYY-MM-DD
        pcNm = findPcNm(mydir[date])
        nFiles = findNumberOfFiles(mydir[date])
        #fwDate = datetime.now()
        #fwDate = fwDate.strftime('%y%m%d') #yymmdd format

        # 추출한 정보를 임시 데이터프레임에 저장
        tempData['oDate'] = oDate
        tempData['dataType'] = dataType
        tempData['lineNm'] = lineNm
        tempData['pcNm'] = pcNm
        tempData['nFiles'] = nFiles

        #print(tempData)
        
        # 추출한 정보를 dataByDayByPc 데이터프레임에 합쳐나감
        dataByDayByPc = pd.concat([dataByDayByPc, tempData], axis=0) 
        
    dataByDayByPc.reset_index(drop=True, inplace=True)
    #print(dataByDayByPc)
    return dataByDayByPc

def creatingNewStatnId(myLine_rtData, lineNm):

    statnIdNm = myLine_rtData[['statnId', 'statnNm']] 

    # 2. 중복 제거
    statnIdNm = statnIdNm.drop_duplicates(['statnId', 'statnNm']) 

    # 3. 신분당선일 경우, 별도 처리
    if lineNm == "LineS": # 신분당선일 경우, 미금역 데이터 역 코드 오류를 보정하기 위한 별도의 행을 삽입 (일부 시기에 해당)
        migum = df({"statnId":[1077006813, 1077006813, 1077006813], "statnNm":[12, "12", "미금"]}) #12가 미금역으로 되어있음 - 2020-01-17 이전 데이터에서 발생되는 현상
        #print(migum)
        statnIdNm = statnIdNm.append(migum)

    # 4. 다시한번 중복 제거
    statnIdNm = statnIdNm.drop_duplicates(['statnNm']) 
    
    # 5. new_statnId 열 생성
    statnIdNm['new_statnId'] = statnIdNm['statnId']  

    # 6. 신분당선일 경우 역 ID 변경 수행
    if lineNm == "LineS": # 신분당선 역코드의 경우 3자리와 4자리를 왔다갔다 하는 경우가 있어서 이를 수동으로 보정
        statnIdNm = statnIdNm.replace({'new_statnId':[1077000687, 1077000688, 1077000689]}, {'new_statnId':[1077006807, 1077006808, 1077006809]})

    # 8. 노선 번호와 병합된 역 코드 데이터를 분해 : 끝 네자리를 잘라서 new_statnId에 저장
    statnIdNm['new_statnId'] = statnIdNm['new_statnId'].apply(lambda x: str(x)).apply(lambda x: x[(len(x)-4):(len(x))]).astype('int')
    return statnIdNm # 

def rtDataSortingAndDropDuplicates(givenData, trainNoColName):
    givenData = givenData.sort_values(by='trainSttus', ascending=True) # 열차상태정보에 대해 Sorting
    givenData = givenData.sort_values(by='recptnDt', ascending=True) # 열차번호에 대해 Sorting
    givenData.drop_duplicates(["directAt", "recptnDt", "statnNm", trainNoColName, "trainSttus", "updnLine"])
    givenData.reset_index(drop=True, inplace=True) # index 초기화

    return givenData

def changingShortStatnId(myLine_rtData, statnIdNm):
    # 역이름-ID 매칭 : 역이름-ID가 매칭되는 statnIdNm이란 dataframe을 생성하기 위해 전체 데이터에서 statnId, statnNm 기준으로 고유 행 추출 (중복 행 제거)
    # 1. statnId와 statnNm 열만 발췌


    # 9. 기존 역 번호와 종착역 번호를 백업
    myLine_rtData[['old_statnId', 'old_statnTid']] = myLine_rtData[['statnId', 'statnTid']]

    # 10. 역이름-ID 매칭 데이터프레임을 참조하여 역 ID를 부여
    ## 위에서 만든 statnIdNm 데이터프레임의 new_statnId열을 활용해여 새로운 역 코드를 부여
    for i in range(0,len(statnIdNm.index)):
    #print(statnIdNm.statnNm.iloc[i], statnIdNm.statnId.iloc[i])
        myLine_rtData = myLine_rtData.replace({'statnId':statnIdNm.statnId.iloc[i]}, {'statnId':statnIdNm.new_statnId.iloc[i]}) # 현재 역 코드 관련
        myLine_rtData = myLine_rtData.replace({'statnTid':statnIdNm.statnId.iloc[i]}, {'statnTid':statnIdNm.new_statnId.iloc[i]}) # 종착역 코드 관련


    # 11. 결과 반환
    return myLine_rtData


def addingArtificialTrainNo(myd1):
    myResDf = df()

    trains = myd1.trainNo.unique()
    t = trains[2] # 3 - 7/15 9편성
    for t in trains:
        mytrain = df()

        # 1. 편성 단위의 데이터를 추출
        mytrain = myd1[myd1.trainNo==t]
        mytrain = rtDataSortingAndDropDuplicates(mytrain, "trainNo")
        #mytrain.to_csv("mytrain_trinNo9-200703.csv")
        # 2. 빈 데이터프레임 생성
        updnLine2 = df(index=range(0,len(mytrain.index)), columns=['trainNo2'], dtype="str")
        #print(len(mytrain.index))
        #print(updnLine2)
        
        # 3. 논리 점검을 위한 진행방향정보 변수 설정
        prev_bnd = mytrain['updnLine'].iloc[0] # 이전 행의 진행방향정보 초기화
        
        # 4. 편성 별 운행순서 계수를 위한 변수 설정
        if prev_bnd == 0: # 제일 첫 운행이 강남방향으로 시작 될 경우
            cnt = 0 
        elif prev_bnd == 1: # 제일 첫 운행이 광교방향으로 시작 될 경우
            cnt = 1

        # 편성 별 데이터의 전체 행에 대해 반복문 시행
        for i in range(0,len(mytrain.index)): 
            
            # 5. 편성 별  가상 열차운행순번 생성
            # 5.1. 현재 행의 방향정보 저장
            current_bnd = mytrain['updnLine'].iloc[i] 

            # 5.2. 이전행과 현재 행의 방향정보가 다른 경우 편성 별 운행순번 생성
            if current_bnd != prev_bnd: 
                cnt = cnt + 1 # 달라졌으면, 운행순서를 1 증가시킴
                prev_bnd = current_bnd # 현재 방향정보를 저장
                
            # 5.3. 이전행과 현재 행의 방향정보가 같은 경우
            ## 2번째 행 부터만 들어오는 조건식 부분 (이전 행과의 비교를 해야 하므로)
            ## 이전 행과 현재 행의 방향정보가 같고 i>1보다 큰 경우,
            
            elif i>1: 
                # 5.3.1. 진행방향이 바뀌는지 안바뀌는지 체크하기 위해 역 코드간의 빼기연산을 시행
                direction_checker = mytrain['statnId'].iloc[i] - mytrain['statnId'].iloc[i-1] 
                
                # 5.3.2. 현재 행과 이전 행의 데이터 수신시각 차이 계산
                current_recptnDt = mytrain['recptnDt'].iloc[i] # 현재 행의 데이터 수신시각 저장
                previous_recptnDt = mytrain['recptnDt'].iloc[i-1] # 이전 행의 데이터 수신시각 저장
                time_diff = current_recptnDt - previous_recptnDt # 시간차이 계산

                #if t==13:
                #  print("cnt:", cnt, "stn:", mytrain['statnNm'].iloc[i], "crnt_bnd:", current_bnd, "prev_bnd:", prev_bnd, "dchk:", direction_checker, current_bnd == 0 & direction_checker > 0)

                # 5.3.3. 편성 별 운행순서 ID 부여
                if current_bnd == 0 and direction_checker > 0: # 0번방향(강남방향)인데 현재 역이 이전 행의 역보다 거슬러 올라간 경우
                    #print('\n---------')
                    #print("curdate:", current_recptnDt)
                    #print("prevdate:", previous_recptnDt)
                    #print("i11:", i, "- timediff: ", time_diff, " trainId:", t)
                    cnt = cnt + 2 # 반대방향 운행 정보가 없는것으로 처리하여 운행순서를 2만큼 증가시킴
                    prev_bnd = current_bnd

                elif current_bnd == 1 and direction_checker < 0: # 1번방향(광교방향)인데 현재 역이 이전 행의 역보다 거슬러 올라간 경우  
                    #print('\n---------')
                    #print("curdate:", current_recptnDt)
                    #print("prevdate:", previous_recptnDt)
                    #print("i21:", i, "- timediff: ", time_diff, " trainId:", t)      
                    cnt = cnt + 2 # 반대방향 운행 정보가 없는것으로 처리하여 운행순서를 2만큼 증가시킴
                    prev_bnd = current_bnd

                elif time_diff.seconds > 1800: # 같은 방향으로 운행했고 역을 거슬러 올라가지도 않았지만, 두 역간의 시간차가 클 경우
                    #print("curdate:", current_recptnDt)
                    #print("prevdate:", previous_recptnDt)
                    #print("i3:", i, "- timediff: ", time_diff, " trainId:", t)
                    cnt = cnt + 2 # 반대방향 운행 정보가 없는것으로 처리하여 운행순서를 2만큼 증가시킴
                    prev_bnd = current_bnd
                
            # 5.4. 열차 편성번호와 조합하여 가상 열차번호 부여
            mystr = format(mytrain['trainNo'].iloc[i], '02') + format(cnt, '02')
            #print("i:", i, "-", mytrain.loc[i,['statnNm', 'recptnDt', 'trainNo', 'statnTnm', 'trainSttus', 'updnLine']], "mystr:", mystr, " -", type(mystr))
            
            # 5.5. 가상 열차변호를 미리 생성한 dataFrame에 저장
            updnLine2.iloc[i] = mystr

        # 5.6. 가상 열차번호를 기존의 편성 별로 분할했던 데이터프레임 제일 우측열에 덧붙임 - 이때 index 맞아야 병합되므로 주의 
        mytrain2 = pd.concat([mytrain, updnLine2], axis=1)
        
        # 5.7. 완성된 편성 별 데이터를 아래쪽으로 합쳐나감
        myResDf = pd.concat([myResDf, mytrain2], axis=0) 


    # 5.8 가상 열차번호 부여가 완료된 데이터의 index를 초기화
    myResDf.reset_index(drop=True, inplace=True)

    #myResDf\.to_csv(path_or_buf=basePath+"/2020R01-02_data/"+"2019-08-16_LineS_rtTrainPos-trainId2_sample-200130.csv")

    # 5.9. 완성된 데이터프레임 반환 
    return myResDf

# ==========================================================================================
def adjustDist(targetDf, trainSttus, adjDist):
    rowsFitWithStattus = np.where(targetDf.trainSttus == trainSttus)[0]
    if len(rowsFitWithStattus)>0:
        targetDf.Dist.iloc[rowsFitWithStattus] = targetDf.Dist.iloc[rowsFitWithStattus].apply(lambda x: x+adjDist)
    else :
        print("length of rowsFitWithStattus is less than 1: "+len(rowsFitWithStattus))

    return targetDf

# ==========================================================================================
def chkLastStnId(lineNm, bnd, arr_ttable):
    if (lineNm == "Line9") | (lineNm == "LineS") :
        if bnd == 1 :
            lastStnRes = arr_ttable.columns[(len(arr_ttable.columns)-1)]
        elif bnd == 0 : 
            lastStnRes = arr_ttable.columns[0]
    elif lineNm == "LineA"  :
        if bnd == 1 :
            lastStnRes = arr_ttable.columns[0]
        elif bnd == 0 : 
            lastStnRes = arr_ttable.columns[(len(arr_ttable.columns)-1)]

    return lastStnRes

# ==========================================================================================
def chkNextStnId(lineNm, bnd, arr_ttable, station, lastStnId, howManyStops):
    # 1. 다음 역 정보 찾기
    if (lineNm == "Line9") | (lineNm == "LineS") :
        if bnd == 1 : # 9호선과 신분당선의 bnd1은 종점으로 갈 수록 statnId2가 늘어남
            nextStnId = arr_ttable.columns[(np.where(arr_ttable.columns==station)[0][0]+howManyStops)] # 현재 대피역의 다음 역 (대피역일 필요 없음) 정보 저장
        
        elif bnd == 0 : # 9호선과 신분당선의 bnd0은 종점으로 갈 수록 statnId2가 줄어듦
            nextStnId = arr_ttable.columns[(np.where(arr_ttable.columns==station)[0][0]-howManyStops)] # 현재 대피역의 다음 역 (대피역일 필요 없음) 정보 저장

    elif lineNm == "LineA"  :
        if bnd == 1 : # 공항철도의의 bnd1은 종점으로 갈 수록 statnId2가 줄어듦
            nextStnId = arr_ttable.columns[(np.where(arr_ttable.columns==station)[0][0]-howManyStops)] # 현재 대피역의 다음 역 (대피역일 필요 없음) 정보 저장
        
        elif bnd == 0 : # 공항철도의 bnd0은 종점으로 갈 수록 statnId2가 늘어남
            nextStnId = arr_ttable.columns[(np.where(arr_ttable.columns==station)[0][0]+howManyStops)] # 현재 대피역의 다음 역 (대피역일 필요 없음) 정보 저장
    
    # 2. 다음 역 정보가 해당 노선의 마지막 역(또는 첫 역)을 초과하지 않는지 검지 
    if (((lineNm == "Line9") | (lineNm == "LineS")) & bnd == 1) | ((lineNm == "LineA") & (bnd == 0)) : # 현재는 9호선에 대해서만 피추월횟수를 계산하므로
        if nextStnId > lastStnId :  
            nextStnId = "Err" 

    elif (((lineNm == "Line9") | (lineNm == "LineS")) & bnd == 0) | ((lineNm == "LineA") & (bnd == 1)) :
        if nextStnId < lastStnId :  
            nextStnId = "Err" 

    return nextStnId

# ==========================================================================================
def analyzingOvertake(trainArrDepOrderDiff, arrTrainNo_atStnS_ip0StnS, arrTime_atStnS_ip0StnS, depTime_atStnS_ip0StnS, arrTrainNo_atStnS_ip1StnS, arrTime_atStnS_ip1StnS, depTime_atStnS_ip1StnS) :
    if trainArrDepOrderDiff > 0 : 
        #print("\n Following train had been overtaked. \n  currentTrain: "+str(arrTrainNo_atStnS_ip0StnS)+" arr at "+str(arrTime_atStnS_ip0StnS)+",  dep at "+str(depTime_atStnS_ip0StnS)+",  \n followingTrain : "+str(arrTrainNo_atStnS_ip1StnS)+" arr at "+str(arrTime_atStnS_ip1StnS)+",  dep at "+str(depTime_atStnS_ip1StnS)+"\n\n")
        #currentOvertakeTable[str(stn)][arrTrainNo_atStnS_ip0StnS] = currentOvertakeTable[str(stn)][arrTrainNo_atStnS_ip0StnS] + 1
        currentNumOfOvertake = 1
    
    else :
        #print("\n Following train had not been overtaked. \n\n")
        #currentOvertakeTable[str(stn)][arrTrainNo_atStnS_ip0StnS] = currentOvertakeTable[str(stn)][arrTrainNo_atStnS_ip0StnS] + 0
        currentNumOfOvertake = 0

    #return currentOvertakeTable
    return currentNumOfOvertake

# ==========================================================================================
def updatingNumOfOvertake(statnId, nOvertakeTable, targetTrainNo, numberOfOvertakeFreq, overtakeTrnNoTable, overtakingTrainNo) :
    # 0. 계속 조회해야하는 열차 ID를 로컬변수로 받기
    trainIdIndex = nOvertakeTable.index
    
    # 1. 피추월횟수가 양수인 경우
    if numberOfOvertakeFreq > 0 :
        # 1.1. 해당 역 해당 열차의 기존 피추월횟수가 0이었으면 기존 값에 새로 계산된 피추월횟수만큼 더해주기
        if nOvertakeTable[str(statnId)].iloc[ np.where( trainIdIndex == targetTrainNo )[0][0] ] >= 0 :
            nOvertakeTable[str(statnId)].iloc[ np.where( trainIdIndex == targetTrainNo )[0][0] ] = nOvertakeTable[str(statnId)].iloc[ np.where( trainIdIndex == targetTrainNo )[0][0] ] + numberOfOvertakeFreq
        
        # 1.2. 피추월횟수데이터(nOvertakeTable)의 해당 역 해당 열차 값이 초기화 전(-99)이었으면 기존 값에 새로 계산된 피추월횟수를 바로 초기화시키기
        elif nOvertakeTable[str(statnId)].iloc[ np.where( trainIdIndex == targetTrainNo )[0][0] ] == -99 :
            nOvertakeTable[str(statnId)].iloc[ np.where( trainIdIndex == targetTrainNo )[0][0] ] = numberOfOvertakeFreq
        
        # 1.3. 추월 한 열차ID를 추월열차정보table (overtakeTrainNoTable) 에 업데이트 : 마지막으로 추월한 열차 정보가 저장되는 셈
        overtakeTrnNoTable[str(statnId)].iloc[np.where(overtakeTrnNoTable.index==targetTrainNo)] = str(overtakingTrainNo)

        # 1.4. 분석종료 flag 업데이트 - False : 계속 검색할 수 있도록. 추월한 열차가 없을 때 까지 검색되게 하도록 설계하였음.
        analysisEndFlag = False

    # 2. 피추월횟수가 0인 경우
    elif numberOfOvertakeFreq == 0 :
        # 2.1. 해당 역 해당 열차의 기존 피추월횟수가 0이었으면 : Err! 이미 0이었으면 피추월횟수 조회 절차가 종료되므로 다시 이 절차에 올 수 없음
        if nOvertakeTable[str(statnId)].iloc[ np.where( trainIdIndex == targetTrainNo )[0][0] ] == 0 :
            print("The algorithm tried to update the number of overtaking train! statnId:"+str(statnId)+",  targetTrain: "+str(targetTrainNo)+",  overtakingTrainNo: "+str(overtakingTrainNo)+"\n========\n\n")

        # 2.2. 피추월횟수데이터(nOvertakeTable)의 해당 역 해당 열차 값이 초기화 전(-99)이었으면 기존 값에 새로 계산된 피추월횟수(0)를 바로 초기화시키기
        elif nOvertakeTable[str(statnId)].iloc[ np.where( trainIdIndex == targetTrainNo )[0][0] ] == -99 :
            nOvertakeTable[str(statnId)].iloc[ np.where( trainIdIndex == targetTrainNo )[0][0] ] = numberOfOvertakeFreq
            overtakeTrnNoTable[str(statnId)].iloc[np.where(overtakeTrnNoTable.index==targetTrainNo)] = "Not be overtaken" # 추월열차정보를 초기화 
        
        # 2.3. 해당 역 해당 열차의 기존 피추월횟수가 양수였으면 : Do nothing
        elif nOvertakeTable[str(statnId)].iloc[ np.where( trainIdIndex == targetTrainNo )[0][0] ] > 0 :
            overtakeTrnNoTable[str(statnId)].iloc[np.where(overtakeTrnNoTable.index==targetTrainNo)] # do Nothing
        
        # 4.5.4.6.5. 피추월횟수 계산 완료 변수 초기화
        analysisEndFlag = True
        #print("Updating the number of overpassing train had been finished! statnId:"+str(statnId)+",  targetTrain: "+str(targetTrainNo)+"\n\n")

    return [analysisEndFlag, nOvertakeTable, overtakeTrnNoTable]


# ========================================================================

def findFeasibleTrainId (trainNo_atStnS_ip1StnS, initStnId, nextStnTF, nextOvtStnId, trainNoByPreviousStnList, stnIdByPreviousStnList, arrTtable, depTtable) :
    
# 추가검색대상 열차정보에 대한 고유값 도출
    trainNoForTest = list(set(trainNoByPreviousStnList)) # 추가검색대상에 대한 고유값 추출
    stnIdForTest = list(set(stnIdByPreviousStnList)) # 추가검색대상에 대한 고유값 추출

    # 추가검색 대상 열차가 1개인 경우    
    if len(trainNoForTest) == 1 :
        #print("                                    i+1열차 ID가 수정되었습니다: "+str(trainNo_atStnS_ip1StnS)+" --> "+str(candidateNaTrainNo[0])+" : 오류점검리스트의 최빈값 열차 in function")
        trainNo = trainNoForTest[0] # 현재 확인한 i+1열차의 정보를 사용하는것으로 갱신
        whichStn = stnIdForTest[0]
        arrDepDataAvailableTF = True
        
        # 열차ID가 갱신됐는지 여부 확인
        if trainNo_atStnS_ip1StnS == trainNo :
            trainIdHadBeenUpdated = False 
        else :
            trainIdHadBeenUpdated = True
        
        return trainNo, whichStn, arrDepDataAvailableTF, trainIdHadBeenUpdated

    elif len(trainNoForTest) > 1 :
        #print("S역 누락 출발열차 후보가 1개 이상입니다 in function")

        # 열차 별 출도착시각 저장할 리스트 초기화
        arrTimeByStnByTrainToBeChk = df()
        depTimeByStnByTrainToBeChk = df()

        for n in range(0,len(stnIdForTest)) :
            # 분석대상 역 정보 불러오기 : 도착, 출발 모두 불러와서 sorting 하기
            stnIdToBeChk = stnIdForTest[n]                                                  # 분석대상 역 ID 불러오기
            arrTableAtStnToBeChk = arrTtable[stnIdToBeChk].sort_values(ascending=True)      # 분석대상 역 도착데이터 불러와서 sorting
            depTableAtStnToBeChk = depTtable[stnIdToBeChk].sort_values(ascending=True)      # 분석대상 역 출발데이터 불러와서 sorting
            
            # 열차 별 출도착시각 저장할 리스트 초기화
            arrTimeByStnToBeChk = []
            depTimeByStnToBeChk = []
            trainIdList = []

            # 열차별 출도착시각 리스트 저장
            for m in range(0,len(trainNoForTest)) :
                # 분석 대상 열차 ID 추출
                trainNoToBeChk = trainNoForTest[m]
                arrTimeOfTrainToBeChk = arrTableAtStnToBeChk[arrTableAtStnToBeChk.index==trainNoToBeChk].iloc[0]
                depTimeOfTrainToBeChk = depTableAtStnToBeChk[depTableAtStnToBeChk.index==trainNoToBeChk].iloc[0]
                arrTimeByStnToBeChk.append(arrTimeOfTrainToBeChk)
                depTimeByStnToBeChk.append(depTimeOfTrainToBeChk)
                trainIdList.append(trainNoToBeChk)
            
            # 역 별 리스트에 합치기
            arrTimeByStnByTrainToBeChk = pd.concat([arrTimeByStnByTrainToBeChk, df({str(stnIdToBeChk):arrTimeByStnToBeChk}, index=trainIdList)], axis=1)
            depTimeByStnByTrainToBeChk = pd.concat([depTimeByStnByTrainToBeChk, df({str(stnIdToBeChk):depTimeByStnToBeChk}, index=trainIdList)], axis=1) 
         
        for n in range(0,len(stnIdForTest)) :
            
            minTrnID = []           # 최소값 도출 열차정보 초기화
            ListForArrMinChk = []   # 도착정보 최소값 리스트 초기화
            ListForDepMinChk = []   # 출발정보 최소값 리스트 초기화
            findResTF = False       # 결과 찾기 성공여부 변수 초기화

            if ( (nextStnTF==False) & ( (str(stnIdForTest[n]) == str(nextOvtStnId)) & (str(stnIdForTest[n]) != "902") ) ) == False :
                ListForArrMinChk = arrTimeByStnByTrainToBeChk[str(stnIdForTest[n])]         # 분석대상 역 별 분석대상 열차별 도착시각 데이터 불러오기
            
                if (len(np.where(pd.isna(ListForArrMinChk)==True)[0]) == 0) :
                    minTrnID = [ ListForArrMinChk.index[ np.where(df(ListForArrMinChk) == df(ListForArrMinChk).min()[0])[0][0]] ]
                    findResTF = True                                                        # 결과를 찾았으면 True로 초기화
                #else :
                    #print("해당 역의 도착데이터 둘 중 하나가 NA입니다")
            else :
                    print("이전 역 방향 조회 중, 이전 피추월역의 도착 정보는 사용할 수 없습니다")  
            
            if findResTF == False :
                if ( (nextStnTF==True) & ( (str(stnIdForTest[n]) == str(nextOvtStnId)) & (str(stnIdForTest[n]) != "902") ) ) == False : 
                    ListForDepMinChk = depTimeByStnByTrainToBeChk[str(stnIdForTest[n])]

                    if len(np.where(pd.isna(ListForDepMinChk)==True)[0]) == 0 :
                        minTrnID = [ ListForDepMinChk.index[np.where(df(ListForDepMinChk) == df(ListForDepMinChk).min()[0])[0][0]] ]
                        findResTF = True
                
                else :
                    print("다음 역 방향 조회 중, 다음 피추월역의 출발 정보는 사용할 수 없습니다")          
            
            if len(minTrnID) > 0 :
                trainNo = minTrnID[0]
                whichStn = stnIdForTest[n]
                arrDepDataAvailableTF = True
                
                if trainNo_atStnS_ip1StnS == trainNo :  # 열차ID가 갱신됐는지 여부 확인
                    trainIdHadBeenUpdated = False 
                else :
                    trainIdHadBeenUpdated = True

                return trainNo, whichStn, arrDepDataAvailableTF, trainIdHadBeenUpdated

        if len(minTrnID) == 0  :
            print("최소값을 갖는 열차가 도출되지 않았습니다.")
            trainNo = trainNo_atStnS_ip1StnS
            arrDepDataAvailableTF = False
            trainIdHadBeenUpdated = False 
            return trainNo_atStnS_ip1StnS, initStnId, arrDepDataAvailableTF, trainIdHadBeenUpdated

    else : # elif len(trainNoForTest) > 1 :
        print("S역 누락 출발열차 후보가 0개입니다 in function")
        trainNo = trainNo_atStnS_ip1StnS
        arrDepDataAvailableTF = False
        trainIdHadBeenUpdated = False 
        return trainNo_atStnS_ip1StnS, initStnId, arrDepDataAvailableTF, trainIdHadBeenUpdated


# ========================================================================


def findFeasibleTrainInfo (targetTrainNo, nextStnTF, trainNoByPreviousStnList, previousStnList, arrTtable, depTtable) :
    
    # 0. 열차 누락정보 조회 리스트에서 정보복원 대상 열차의 정보가 존재하는 역 정보 조회
    tempOrderInList = np.where(df(trainNoByPreviousStnList)==targetTrainNo)[0]

    if len(tempOrderInList) > 0 :
        orderInList = np.where(df(trainNoByPreviousStnList)==targetTrainNo)[0][0]

        # 1. 정보와 연계되는 역 ID 저장
        stnIdMatchedWithTargetTrain = previousStnList[orderInList]

        # 2. 정보와 연계되는 역의 출도착 정보 저장 및 Sorting
        funcArrTtable = arrTtable[stnIdMatchedWithTargetTrain].sort_values(ascending=True)
        funcDepTtable = depTtable[stnIdMatchedWithTargetTrain].sort_values(ascending=True)
        
        # 3. 정보복원 대상 열차의 출도착 순서 저장
        targetTrainArrOrder = np.where(funcArrTtable.index==targetTrainNo)[0][0]
        targetTrainDepOrder = np.where(funcDepTtable.index==targetTrainNo)[0][0]
        
        # 4. 정보복원 대상 열차의 출도착 정보 저장
        targetTrainArrTime = funcArrTtable[funcArrTtable.index==targetTrainNo].iloc[0]
        targetTrainDepTime = funcDepTtable[funcDepTtable.index==targetTrainNo].iloc[0]


        # 5. 반환 대상 정보 정리
        # 5.1. 다음역 방향 조회라면
        DataOK_TF = True

        if nextStnTF == True :    
            # 5.1.1. 가능하다면 출발 관련 정보를 반환
            if targetTrainDepTime is not pd.NaT :
                targetTrainArrDepTime = targetTrainDepTime
                targetTrainArrDepOrder = targetTrainDepOrder
                targetTrainTimetable = funcDepTtable

            # 5.1.2.
            elif targetTrainArrTime is not pd.NaT :
                targetTrainArrDepTime = targetTrainArrTime
                targetTrainArrDepOrder = targetTrainArrOrder
                targetTrainTimetable = funcArrTtable
            
            # 5.1.3.
            else :
                #print("출도착 시각이 모두 NaT입니다.")
                DataOK_TF = False

        # 5.2. 이전역 방향 조회라면
        else : # if nextStnTF == True :
            # 5.2.1. 가능하다면 도착 관련 정보를 반환
            if targetTrainArrTime is not pd.NaT :
                targetTrainArrDepTime = targetTrainArrTime
                targetTrainArrDepOrder = targetTrainArrOrder
                targetTrainTimetable = funcArrTtable
            
            # 5.2.2.
            elif targetTrainDepTime is not pd.NaT :
                targetTrainArrDepTime = targetTrainDepTime
                targetTrainArrDepOrder = targetTrainDepOrder
                targetTrainTimetable = funcDepTtable
            
            # 5.2.3.
            else :
                #("출도착 시각이 모두 NaT입니다.")
                DataOK_TF = False
    
    else : # if len(tempOrderInList) > 0 : 목표 열차에 대한 정보가 들어있지 않는 경우
        DataOK_TF = False
    
    # 6 정보 반환
    if DataOK_TF == True :
        return [DataOK_TF, targetTrainArrDepOrder, targetTrainArrDepTime, targetTrainTimetable, stnIdMatchedWithTargetTrain]
    else :
        return [DataOK_TF, np.NaN, pd.NaT, np.NaN, np.NaN]

# ==========================================================================================

def updatingFuncStationCnt (cnt, depTableTF, searchToNext) :

    # 4.5.7.2. 지금 조회한게 출발시각표 정보면
    if depTableTF == True :
        
        # 4.5.7.2.1. 출발시각표 조회 flag를 false로 변환해서 다음 while 에서 도착정보를 조회할 수 있게 처리
        depTableTF = False
        
        # 4.5.7.2.2. 현재 알고리즘이 전 역 방향으로 조회하는지 다음역 방향으로 조회하는지 확인
        if (searchToNext == True) :
            # 4.5.7.2.3. 다음 역 방향 조회라면 다음 역 조회할 수 있게 cnt 를 1 증가
            cnt = cnt + 1 
        
        else :
            # 4.5.7.2.4. 전 역 방향 조회라면 전 역 조회할 수 있게 cnt 를 1 감소
            cnt = cnt - 1
    
    # 4.5.7.3. 지금 조회한게 도착시각표 정보면
    else : # if depTableTF == True :
        
        # 4.5.7.4. 출발시각표를 조회할 수 있도록 depTableTF 를 True로 초기화.
        depTableTF = True
        # cnt는 도착데이터를 조회한 후에만 1씩 증가시키도록 하므로, 여기서는 증가시키지 않음
    
    return depTableTF, cnt

# ==========================================================================================

def findCandidNaTrainNo (trainNo_atStnS_ip1StnS, previousStnList) :
    candidateNaTrainNo = [findModeFromList(previousStnList)]
    if len(candidateNaTrainNo) == 1 :
        #print("                                    i+1열차 ID가 수정되었습니다: "+str(trainNo_atStnS_ip1StnS)+" --> "+str(candidateNaTrainNo[0])+" : 오류점검리스트의 최빈값 열차 in function")
        trainNo = candidateNaTrainNo[0] # 현재 확인한 i+1열차의 정보를 사용하는것으로 갱신
        arrDepDataAvailableTF = True
        useCandidateTrain = True
        return trainNo, arrDepDataAvailableTF, useCandidateTrain

    elif len(candidateNaTrainNo) == 0:
        print("S역 누락 출발열차 후보가 0개입니다 in function")
        trainNo =trainNo_atStnS_ip1StnS
        arrDepDataAvailableTF = False
        useCandidateTrain = False
        return trainNo_atStnS_ip1StnS, arrDepDataAvailableTF, useCandidateTrain

    else :
        print("S역 누락 출발열차 후보가 1개 이상입니다 in function")
        trainNo =trainNo_atStnS_ip1StnS
        arrDepDataAvailableTF = False
        useCandidateTrain = False
        return trainNo_atStnS_ip1StnS, arrDepDataAvailableTF, useCandidateTrain

# ==========================================================================================

#findNextOvtStnOrder(lineNm=lineNm, bnd=bnd, nextStnTF=searchingToNextStn, currentOvtStnOrder=funcOvertakeStation, ovtStnIdList=funcOvertakeStaList, wholeStnIdList=funcStations)
def findNextOvtStnOrder (lineNm, bnd, nextStnTF, currentOvtStnOrder, ovtStnIdList, wholeStnIdList) :
    # 1. 다음 피추월역 위치를 확인 : 피추월역 리스트는 진행방향에 맞게 정렬되어있음. 진행방향 상 기점역이 0번 위치, 종점역이 마지막 위치로 설정되어있음
    # 1.1. 9호선 신분당선의 bnd0이나 공항철도의 bnd1 : 역 ID가 진행방향으로 이동하며 감소함
    if (((lineNm == "Line9") | (lineNm == "LineS")) & (bnd == 0)) | (( (lineNm == "LineA") ) & (bnd == 1)) : 
        
        # 1.1.1. 진행방향 상 다음(하류) 역 방향 조회라면
        if nextStnTF == True : 
            # 1.1.1.1. 마지막 피추월역을 조회하고 있는지 확인
            if (currentOvtStnOrder + 1) <= (len(ovtStnIdList)-1) : # 마지막 피추월역을 조회하고있는게 아니라면
                nextOvertakeStnOrder = np.where(wholeStnIdList == ovtStnIdList[currentOvtStnOrder + 1])[0][0]

                # 너무 인접한 추월가능역 보정을 위해 max(가장 인접한 3번째 상하류역, 인접한 피추월역)을 조회가능범위로 설정
                #overtakeAnalysisCandidateStnOrder = max(0, np.where(wholeStnIdList == ovtStnIdList[currentOvtStnOrder])[0][0] - stationSearchBuffer) # bnd0의 하류방향 검색이라 3 을 빼줌
                #nextOvertakeStnOrder = min(nextOvertakeStnOrder, overtakeAnalysisCandidateStnOrder)

            # 1.1.1.2. 마지막 피추월역에 대해 조회하고 있다면 
            else : # 해당 노선 마지막역 정보를 입력. wholeStnIdList는 진행방향별로 정렬이 되어있으므로 (2021-08-05 updated) 마지막 역 정보로 초기화
                if lineNm == "Line9" :
                    if str(wholeStnIdList[(len(wholeStnIdList)-1)]) == '901' :
                        nextOvertakeStnOrder = (len(wholeStnIdList)-2) # 9호선은 bnd1의 급행이 정차할수 있는 마지막역이 김포공항역이므로 nextOvertakeStnOrder을 wholeStnIdList의 마지막 역의 index보다 1개 작은 값으로 초기화
                    else :
                        nextOvertakeStnOrder = len(wholeStnIdList)-1
                else :
                    nextOvertakeStnOrder = len(wholeStnIdList)-1 # 9호선은 bnd0이 공항방향, 마지막 역이 가장 작은 역 코드를 갖고있으므로 nextOvertakeStnOrder을 wholeStnIdList의 마지막 역의 index로 초기화

        # 1.1.2. 진행상황 상 전(상류) 역 방향 조회라면 
        else : # if nextStnTF == True : 
            # 1.1.2.1. 첫 피추월역을 조회하고 있는지 확인
            if (currentOvtStnOrder - 1) >= 0 : # 첫 피추월역을 조회하고있는게 아니라면 : 0보다 큰 이유는 9호선의 첫 역보다 커야 하기 때문
                nextOvertakeStnOrder = np.where(wholeStnIdList == ovtStnIdList[currentOvtStnOrder - 1])[0][0]
                
                # 너무 인접한 추월가능역 보정을 위해 max(가장 인접한 3번째 상하류역, 인접한 피추월역)을 조회가능범위로 설정
                #overtakeAnalysisCandidateStnOrder = min(len(wholeStnIdList)-1, np.where(wholeStnIdList == ovtStnIdList[currentOvtStnOrder])[0][0] + stationSearchBuffer) # bnd0의 상류방향 검색이라 3 을 더해줌
                #nextOvertakeStnOrder = max(nextOvertakeStnOrder, overtakeAnalysisCandidateStnOrder)

            # 1.1.2.2. 첫 피추월역을 조회하고있는것이라면, 
            else : # 기점역 정보를 사용. wholeStnIdList는 진행방향별로 정렬이 되어있으므로 (2021-08-05 updated) 마지막 역 정보로 초기화
                nextOvertakeStnOrder = 0 # 9호선은 bnd0의 시작역이 중앙보훈병원방향, 가장 큰 역 코드를 갖고있으므로 
        
    # 1.2. 9호선 신분당선의 bnd1이나 공항철도의 bnd0 : 역 ID가 진행방향으로 이동하며 증가함
    elif (((lineNm == "Line9") | (lineNm == "LineS")) & (bnd == 1)) | (( (lineNm == "LineA") ) & (bnd == 0)) : 

        # 1.2.1. 진행방향 상 다음(하류) 역 방향 조회라면
        if nextStnTF == True :  
            # 1.2.1.1. 마지막 피추월역을 조회하고 있는지 확인
            if (currentOvtStnOrder + 1) <= (len(ovtStnIdList)-1) :
                nextOvertakeStnOrder = np.where(wholeStnIdList == ovtStnIdList[currentOvtStnOrder + 1])[0][0]

                # 너무 인접한 추월가능역 보정을 위해 max(가장 인접한 3번째 상하류역, 인접한 피추월역)을 조회가능범위로 설정
                #overtakeAnalysisCandidateStnOrder = min(len(wholeStnIdList)-1, np.where(wholeStnIdList == ovtStnIdList[currentOvtStnOrder])[0][0] + stationSearchBuffer) # bnd1의 하류방향 검색이라 3 을 더해줌
                #nextOvertakeStnOrder = max(nextOvertakeStnOrder, overtakeAnalysisCandidateStnOrder)

            
            # 1.2.1.2. 마지막 피추월역에 대해 조회하고 있다면
            else : # 해당 노선 마지막역 정보를 입력
                nextOvertakeStnOrder = len(wholeStnIdList)-1 # 9호선은 bnd1이 중앙보훈병원방향, 가장 큰 역 코드를 갖고있으므로 nextOvertakeStnOrder을 0으로 초기화
        
        # 1.2.2. 진행상황 상 전(상류) 역 방향 조회라면
        else : # if nextStnTF == True :  
            # 1.2.2.1. 첫 피추월역을 조회하고 있는지 확인
            if (currentOvtStnOrder - 1) >= 0 :
                nextOvertakeStnOrder = np.where(wholeStnIdList == ovtStnIdList[currentOvtStnOrder - 1])[0][0]

                # 너무 인접한 추월가능역 보정을 위해 max(가장 인접한 3번째 상하류역, 인접한 피추월역)을 조회가능범위로 설정
                #overtakeAnalysisCandidateStnOrder = max(0, np.where(wholeStnIdList == ovtStnIdList[currentOvtStnOrder])[0][0] - stationSearchBuffer) # bnd1의 상류방향 검색이라 3 을 빼줌
                #nextOvertakeStnOrder = min(nextOvertakeStnOrder, overtakeAnalysisCandidateStnOrder)

            # 1.2.2.2. 첫 피추월역에 대해 조회하고 있다면 
            else : # 해당 노선 첫 역 정보를 입력
                if lineNm == "Line9" :
                    if str(wholeStnIdList[0]) == '901' :
                        nextOvertakeStnOrder = 1 # 9호선은 bnd1의 급행이 정차할수 있는 첫 시작역이 김포공항역이므로 nextOvertakeStnOrder을 1로 초기화
                    else :
                        nextOvertakeStnOrder = 0
                else :
                    nextOvertakeStnOrder = 0 # 신분당선, 공항철도의 경우 bnd1의 기점역이 가장 작은 역 코드를 갖고있으므로 nextOvertakeStnOrder을 0으로 초기화
    
    return nextOvertakeStnOrder

# ==========================================================================================

def findingdepArrTimeAtStaSpN(stationCnt, searchingToNextStn, departureTableFlag, funcStations, funcOvertakeStaList, funcOvertakeStation, lineNm, bnd, myd3_arr_ttable, myd3_dep_ttable, funcArrTrainNo_atStnS_ip0StnS, funcArrTrainNo_atStnS_ip1StnS, arrTimeTable_stnS, funcTrain, givenTrainCnt) :
    
    funcArrTrain_ip1_byPreviousStn = [] #[funcArrTrainNo_atStnS_ip1StnS]
    funcPreviousStnList = [] #[funcOvertakeStaList[funcOvertakeStation]]
    funcDepTrain_ip1_byForwardStn = [] 
    funcForwardStnList = [] 
    stnS_dataChk = False # 최초 False로 초기화. while문을 돌다가 전 역 데이터 관련 문제가 해결되면 True로 갱신되며 while에서 빠져나오게 됨
    outOfRangeTF = False

    funcArrDepTimeDataAvailableFlag = False # 출도착 변수 모두 있는지 확인하는 변수 초기화
    funcErrCntNextSta = 0 # 다음 역 조회 관련 무한루프 방지 변수
    currentDepTableFlag = departureTableFlag # 출발테이블 먼저 조회하는지 확인하는 변수 복제

    funcStationCnt = stationCnt

    # 1. 다음 피추월역 위치를 확인 : 피추월역 리스트는 진행방향에 맞게 정렬되어있음. 진행방향 상 기점역이 0번 위치, 종점역이 마지막 위치로 설정되어있음
    nextOvertakeStnOrder = findNextOvtStnOrder(lineNm=lineNm, bnd=bnd, nextStnTF=searchingToNextStn, currentOvtStnOrder=funcOvertakeStation, ovtStnIdList=funcOvertakeStaList, wholeStnIdList=funcStations)

    # 1.3. 조회역 유효성 검토를 위한 오류cnt 값 초기화
    currentOvertakeStnOrder = np.where(funcStations == funcOvertakeStaList[funcOvertakeStation])[0][0] # 현재 분석대상 피추월역 저장 
    minChkCnt = abs(currentOvertakeStnOrder-nextOvertakeStnOrder)*2
    nextOvertakeStnId = funcStations[nextOvertakeStnOrder]

    # 2. bnd에 따른 다음 조회역의 유효성 검토
    #    while 문 조건 : 역 정보가 마지막 역을 넘지 않고, 분석 종료 Flag가 True가 아니고, 무한루프 방지를 위한 errCnt가 10 미만인 상황에서 while을 돌림
    while ((stnS_dataChk == False) & (funcArrDepTimeDataAvailableFlag == False)) & (funcErrCntNextSta < (minChkCnt+5) ) :
        
        # 2.0. 역 ID 유효성 검토변수 및 출도착시각표 유효성 확인변수 False로 초기화
        funcStaIdFeasibility = False
        funcStationIdAvailabilty = False
        timetableFeasibility = False 

        # 2.1. 조회대상 역 및 분석 역 순서 저장하기
        # 2.1.1. 조회 대상 역(현재 피추월역에 funcStationCnt를 더한 값)이 전체 역 목록에서 몇번째인지 확인 
        overtakeStationOrderInStations = np.where(funcStations == funcOvertakeStaList[funcOvertakeStation])[0][0]
        
        # 2.1.2. 분석 역 순서 저장 : bnd0은 다음역으로 갈 수록 값이 작아지지만, 역 순서가 열차 진행방향 순서로 정렬되어잇으므로 덧셈 적용 (역 순서가 역 코드의 오름차순 정렬일 경우 뺄셈 적용 - 이 경우, 2.2.0, 2.3.0을 다시 살려야 함)
        analysisStnOrder = overtakeStationOrderInStations + funcStationCnt # funcSationCnt 값이 양수, 음수 다 되고 역 순서가 진행방향으로 정렬되어있으므로 덧셈으로 처리

        # 2.2. 유효성 검토 : 2021-08-05-v1까지 있던 노선별 방향별 구분을 수정 : 실시간 출도착데이터의 역 코드 순서가 무조건 열차 진행방향에 맞춰 기록되도록 수정되었기 때문.
        # 2.2.1.
        if searchingToNextStn == True : # 진행방향 상 다음(하류) 역 방향 조회라면
            
            if analysisStnOrder <= nextOvertakeStnOrder : # bnd0은 하류역 ID가 더 작음. equal을 넣을거면 상류방향 조회는 다음 피추월역의 도착데이터까지 사용 가능
                funcDepTimeDataStation = funcStations[analysisStnOrder] 
                funcStaIdFeasibility = True
            else :
                if (analysisStnOrder >= 0) & (analysisStnOrder < len(funcStations)) :
                    funcDepTimeDataStation = funcStations[analysisStnOrder] 
                else :
                    funcDepTimeDataStation = "OutOfRange"
                    
                #print("\n조회 가능한 역 범위를 넘어섰습니다. | 현재역:"+str(funcDepTimeDataStation)+" | 마지막 피추월역:"+str(funcStations[nextOvertakeStnOrder]) + " | TRNID : "+str(arrTimeTable_stnS.index[funcTrain]) +" | bnd:"+str(bnd)+" | serchingToNextStn:"+str(searchingToNextStn)+" | funcErrCntNextSta: " + str(funcErrCntNextSta) + "\n")
                funcArrDepTimeDataAvailableFlag = False
                #stnS_dataChk = True
                outOfRangeTF = True
                
        # 2.2.2. 
        else : # 진행상황 상 전(상류) 역 방향 조회라면
            if analysisStnOrder >= nextOvertakeStnOrder : # bnd0은 상류역 ID가 더 큼. equal을 넣을거면 상류방향 조회는 다음 피추월역의 출발데이터까지 사용 가능
                funcDepTimeDataStation = funcStations[analysisStnOrder] 
                funcStaIdFeasibility = True
            else :
                if (analysisStnOrder >= 0) & (analysisStnOrder < len(funcStations)) :
                    funcDepTimeDataStation = funcStations[analysisStnOrder] 
                else :
                    funcDepTimeDataStation = "OutOfRange"

                #print("\n조회 가능한 역 범위를 넘어섰습니다. | 현재역:"+str(funcDepTimeDataStation)+" | 마지막 피추월역:"+str(funcStations[nextOvertakeStnOrder]) + " | TRNID : "+str(arrTimeTable_stnS.index[funcTrain]) +" | bnd:"+str(bnd)+" | serchingToNextStn:"+str(searchingToNextStn)+" | funcErrCntNextSta: " + str(funcErrCntNextSta) + "\n")
                funcArrDepTimeDataAvailableFlag = False
                #stnS_dataChk = True
                outOfRangeTF = True


        # 조회역이 유효하다면
        if funcStaIdFeasibility == True : 
            
            # 3. 조회대상역의 출|도착 시각표 불러오기    
            # 3.1. currentDepTableFlag 가 False면 
            if currentDepTableFlag == False : #
                if (str(funcDepTimeDataStation) != "902") & ((analysisStnOrder == nextOvertakeStnOrder) & (searchingToNextStn == False))  : # 현재 분석역이 다음 추월역인지 확인. 단 김포공항역의 경우 도착순서대로 출발하므로 관계없음. 각 종착역의 경우 도착순서와 출발순서가 달라질 수 있으므로, 다음 피추월역이 종점역인 경우에도 출발 또는 도착데이터 중 한개만 써야 함
                    
                    if searchingToNextStn == False :        # 이전 역 방향 조회일 경우
                        timetableFeasibility = False            # timetable 유효성 변수 False로 초기화
                        #print("이전역 검색 시 이전 피추월역의 도착정보는 사용할 수 없습니다")

                    else :                                  # 다음역 방향 조회일 경우
                        timetableFeasibility = True             # 다음역 방향 조회일 때 해당 피추월역의 도착데이터까지는 볼 수 있으므로 currentDepTableFlag == False 일 때 까지는 유효함 

                        # 역 조회변수 및 출도착시각표 조회변수 초기화
                        currentDepTableFlag, funcStationCnt = updatingFuncStationCnt(cnt=funcStationCnt, depTableTF=currentDepTableFlag, searchToNext=searchingToNextStn)
                    
                else :
                    timetableFeasibility = True
                    
                    # 3.2.1. S+N 역 도착정보 발췌
                    funcTimeTable_stnSpN = myd3_arr_ttable[funcDepTimeDataStation]

                    # 3.2.2. S+N 역 도착정보 정렬
                    funcTimeTable_stnSpN = funcTimeTable_stnSpN.sort_values(ascending=True)

            # 3.2. currentDepTableFlag 가 False면 
            elif currentDepTableFlag == True :
                if (str(funcDepTimeDataStation) != "902") & ((analysisStnOrder == nextOvertakeStnOrder) & (searchingToNextStn == True)) :  # 현재 분석역이 다음 추월역인지 확인.  단 김포공항역의 경우 도착순서대로 출발하므로 관계없음. 각 종착역의 경우 도착순서와 출발순서가 달라질 수 있으므로, 다음 피추월역이 종점역인 경우에도 출발 또는 도착데이터 중 한개만 써야 함
                    
                    if searchingToNextStn == True :         # 다음 역 방향 조회중이라면
                        timetableFeasibility = False            # timetable 유효성 변수 False로 초기화
                        #print("다음역 검색 시 다음 피추월역의 출발정보는 사용할 수 없습니다")

                    else :                                  # 이전 역 방향 조회중이라면
                        timetableFeasibility = True             # 이전 역 방향 조회일 때 해당 피추월역의 출발데이터까지는 볼 수 있으므로 currentDepTableFlag == True 일 때 까지는 유효함 

                        # 역 조회변수 및 출도착시각표 조회변수 초기화
                        currentDepTableFlag, funcStationCnt = updatingFuncStationCnt(cnt=funcStationCnt, depTableTF=currentDepTableFlag, searchToNext=searchingToNextStn)

                else :
                    timetableFeasibility = True

                    # 3.2.1. S+N 역 출발정보 발췌 (맨 처음엔 S역 부터 시작함)
                    funcTimeTable_stnSpN = myd3_dep_ttable[funcDepTimeDataStation]

                    # 3.2.2. S+N 역 출발정보 발췌 (맨 처음엔 S역 부터 시작함)
                    funcTimeTable_stnSpN = funcTimeTable_stnSpN.sort_values(ascending=True)
            
        if (timetableFeasibility == True) & (funcStaIdFeasibility == True) :
            # 4. S+N역 시각표에서 S역의 선후행열차의 시간정보 저장 : S+N역 출발시각표에서 S역 i, i+1 열차 ID 가 있는 위치 찾기
            # 4.1. 선행 도착열차의 경우 기존과 같이 찾아서 저장
            funcDepTrainOrderOf_ip0StnS_atStnSpN = np.where(funcTimeTable_stnSpN.index==funcArrTrainNo_atStnS_ip0StnS)[0][0]

            # 4.2. 후행 도착열차의 경우 인근역 조회 방향에 따라 찾는 열차를 다르게 설정하기
            # 다음 역 방향 조회를 하는 경우 (피추월역의 다음역) 
            if (searchingToNextStn == True) : # 도착정보 유효성은 확인 된 상태에서 출발정보의 유효성을 검증하는 부분
                
                # 4.2.1. 피추월역(S역)의 후착 열차의 ID가 S+1역에서 몇번째 도착한것인지 확인
                funcDepTrainOrderOf_ip1StnS_atStnSpN = np.where(funcTimeTable_stnSpN.index==funcArrTrainNo_atStnS_ip1StnS)[0][0]
                
                # 4.2.2. 마지막 열차에 대한 정보일경우 
                if funcDepTrainOrderOf_ip1StnS_atStnSpN >= len(funcTimeTable_stnSpN.index) :
                    # i+1열차에 대한 index를 수정하여 같은 열차에 대해 비교하도록 처리. 이렇게 되었을 때 그 결과가 0 또는 오류가 나올것이므로 OK
                    funcDepTrainOrderOf_ip1StnS_atStnSpN = len(funcTimeTable_stnSpN.index) - 1 # funcDepTrainOrderOf_ip1StnS_atStnSpN - 1 #givenTrainCnt
           
            # 4.3.이전 역 방향 조회를 하는 경우 (피추월역의 전역) 
            else : # 도착정보 유효성을 확인하는 중 (출발정보의 유효성을 검증하기 전 단계)
                
                # 4.3.1. 피추월역(S역)의 후착 열차의 ID가 S-1역에서도 선행열차 바로 다음에 도착한것인지 확인하기 위해 선행열차의 도착순서에 함수에서 입력받은 숫자를 더함
                funcDepTrainOrderOf_ip1StnS_atStnSpN = funcDepTrainOrderOf_ip0StnS_atStnSpN + givenTrainCnt
                
                # 4.3.2. 마지막 열차에 대한 정보일경우
                if funcDepTrainOrderOf_ip1StnS_atStnSpN >= len(funcTimeTable_stnSpN.index) :
                    
                    #  i+1열차에 대한 index를 수정하여 같은 열차에 대해 비교하도록 처리. 이렇게 되었을 때 그 결과가 0 또는 오류가 나올것이므로 OK
                    funcDepTrainOrderOf_ip1StnS_atStnSpN = funcDepTrainOrderOf_ip1StnS_atStnSpN - givenTrainCnt 
                
                # 4.3.3. S-1역에서 i 열차 다음으로 도착한 열차의 열차 번호 저장
                funcArrTrainNo_atStnSmN_ip1StnSmN = funcTimeTable_stnSpN.index[funcDepTrainOrderOf_ip1StnS_atStnSpN]
        
            # 5. S+N역 출발시각표에서 S역 i, i+1 열차의 시간정보 저장
            funcDepTime_atStnSpN_ip0StnS = funcTimeTable_stnSpN.iloc[funcDepTrainOrderOf_ip0StnS_atStnSpN] # i번째 열차의 출발시각
            funcDepTime_atStnSpN_ip1StnS = funcTimeTable_stnSpN.iloc[funcDepTrainOrderOf_ip1StnS_atStnSpN] # i+trainCnt번째 열차의 출발시각

            # 6. S+N역 출발시각표에서 S역 i, i+1 열차의 시간정보가 있는지 확인
            # 6.1. 전 역 방향 조회(출발데이터 관련)일 경우 둘 다 있고, 다음 역 방향 조회일경우 i+1 열차의 시간정보가 있다면 : 다음역 방향 조회일 경우 어차피 찾아야 할 i+1열차의 열차ID를 정해두고 찾기 때문에 i열차의 정보가 pd.NaT인지 확인 할 필요가ㅋ
            if (((funcDepTime_atStnSpN_ip0StnS is not pd.NaT) & (funcDepTime_atStnSpN_ip1StnS is not pd.NaT))) : #& (searchingToNextStn == False)) : #| ((funcDepTime_atStnSpN_ip1StnS is not pd.NaT) & (searchingToNextStn == True))  :
                # print("arrTime and depTime of funcTrain i and i+1 had been found." )

                # 6.1.1. 다음 역 (S+1역) 조회 중이라면
                if (searchingToNextStn == True) :
                    #funcDepTrain_ip1_byForwardStn.append( funcDepTrainOrderOf_ip1StnS_atStnSpN )
                    #funcForwardStnList.append( funcDepTimeDataStation )
                    
                    # 6.1.2. 다음 역 (S+1역) 조회 중이면, 다음 역 검색이 완료되었으므로 True로 초기화
                    funcArrDepTimeDataAvailableFlag = True
                    stnS_dataChk = True
                
                # 6.1.2. 전 역 (S-1역) 조회 중이면,
                else : # 기존의 i+1 열차 번호와 S-1역 열차 번호가 동일한지 확인

                    # 6.1.3. S역의 i+1 열차 번호와 S-1역 i+1 열차의 열차 번호가 동일한지 확인
                    if funcArrTrainNo_atStnSmN_ip1StnSmN == funcArrTrainNo_atStnS_ip1StnS :
                        
                        # 6.1.3.1. 동일하면, 이전 역 별 i+1열차 id 저장
                        funcArrTrain_ip1_byPreviousStn.append( funcArrTrainNo_atStnSmN_ip1StnSmN )
                        funcPreviousStnList.append( funcDepTimeDataStation )

                        # 6.1.3.2. 현재역-다음 추월역 간 최소 조회빈도를 다 맞췄으면
                        if funcErrCntNextSta > minChkCnt :
                            # 그 중에 다음열차 정보 찾기
                            funcArrTrainNo_atStnS_ip1StnS, funcFeasibleStnId, funcArrDepTimeDataAvailableFlag, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=funcArrTrainNo_atStnS_ip1StnS, initStnId=funcOvertakeStaList[funcOvertakeStation], nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=funcArrTrain_ip1_byPreviousStn, stnIdByPreviousStnList=funcPreviousStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)

                            funcArrDepTimeDataAvailableFlag, funcDepTrainOrderOf_ip1StnS_atStnSpN, funcDepTime_atStnSpN_ip1StnS, funcTimeTable_stnSpN, funcDepTimeDataStation = findFeasibleTrainInfo(targetTrainNo=funcArrTrainNo_atStnS_ip1StnS, nextStnTF=searchingToNextStn, trainNoByPreviousStnList=funcArrTrain_ip1_byPreviousStn, previousStnList=funcPreviousStnList,  arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                            stnS_dataChk = True
                            
                        else : 
                            # 6.1.3.3.1. 출발데이터를 다 검색했다면
                            if currentDepTableFlag == True : 
                                # 역 조회변수 및 출도착시각표 조회변수 초기화
                                currentDepTableFlag, funcStationCnt = updatingFuncStationCnt(cnt=funcStationCnt, depTableTF=currentDepTableFlag, searchToNext=searchingToNextStn)
                            
                            # 6.1.3.3.2. 출발시각표를 조회할 수 있도록 currentDepTableFlag 를 True로 초기화.
                            else :
                                currentDepTableFlag = True
                                # funcStationCnt는 도착데이터를 조회한 후에만 1씩 증가시키도록 하므로, 여기서는 증가시키지 않음

                    # 6.1.4. 전 역 (S-1역) 조회 중이면, 기존의 i+1 열차 번호와 S-1역 열차 번호가 다르면
                    else : # if funcArrTrainNo_atStnSmN_ip1StnSmN == funcArrTrainNo_atStnS_ip1StnS :
                        #print("ERR. S역과 S-1역의 i+1열차 ID가 다릅니다.")

                        # 6.1.4.1. S-N역에서의 i+1열차의 출도착 순서 저장
                        funcArrDepTrainOrderOf_atStnSmN_ip1StnS = np.where(funcTimeTable_stnSpN.index==funcArrTrainNo_atStnS_ip1StnS)[0][0]
                        timeChk_ip1StnS_atStnSmN = funcTimeTable_stnSpN.iloc[funcArrDepTrainOrderOf_atStnSmN_ip1StnS]
                        funcArrDepTrainOrderOf_atStnSmN_ip1StnSmN = np.where(funcTimeTable_stnSpN.index==funcArrTrainNo_atStnSmN_ip1StnSmN)[0][0]

                        # 6.1.4.2. S-N역에서의 두 열차의 출도착순서 대소비교
                        # 6.1.4.2.1. # S-N역 i+1열차의 S-N역 출도착순서가 S역 i+1열차의 S-N역 출도착순서보다 작다면
                        if (timeChk_ip1StnS_atStnSmN is not pd.NaT) :
                            if funcArrDepTrainOrderOf_atStnSmN_ip1StnSmN < funcArrDepTrainOrderOf_atStnSmN_ip1StnS : 
                                # 이전 역 별 i+1열차 id 저장
                                funcArrTrain_ip1_byPreviousStn.append( funcArrTrainNo_atStnSmN_ip1StnSmN )
                                funcPreviousStnList.append( funcDepTimeDataStation )

                            # 6.1.4.2.2. # S-N역 i+1열차의 S-N역 출도착순서가 S역 i+1열차의 S-N역 출도착순서보다 크다면
                            elif funcArrDepTrainOrderOf_atStnSmN_ip1StnSmN > funcArrDepTrainOrderOf_atStnSmN_ip1StnS : 
                                funcArrTrain_ip1_byPreviousStn.append( funcArrTrainNo_atStnS_ip1StnS )
                                funcPreviousStnList.append( funcDepTimeDataStation )
                            
                        #else :
                            #print("atStnSmN_ip1StnSmN의 출도착시각이 NA 입니다")
                        
                        # 6.1.4.3. 현재 분석역의 최소 조회빈도를 넘어섰으면
                        if funcErrCntNextSta > minChkCnt :
                            funcArrTrainNo_atStnS_ip1StnS, funcFeasibleStnId, funcArrDepTimeDataAvailableFlag, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=funcArrTrainNo_atStnS_ip1StnS, initStnId=funcOvertakeStaList[funcOvertakeStation], nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=funcArrTrain_ip1_byPreviousStn, stnIdByPreviousStnList=funcPreviousStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)

                            funcArrDepTimeDataAvailableFlag, funcDepTrainOrderOf_ip1StnS_atStnSpN, funcDepTime_atStnSpN_ip1StnS, funcTimeTable_stnSpN, funcDepTimeDataStation = findFeasibleTrainInfo(targetTrainNo=funcArrTrainNo_atStnS_ip1StnS, nextStnTF=searchingToNextStn, trainNoByPreviousStnList=funcArrTrain_ip1_byPreviousStn, previousStnList=funcPreviousStnList,  arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                            stnS_dataChk = True
                            
                        # 6.1.4.4. 만약 마지막 두 역의 i+1열차 정보가 다르다면
                        else : 
                            # 6.1.4.4.1. 전 역 방향 조회이고 출도착데이터를 다 검색했다면
                            if currentDepTableFlag == True :
                                # 역 조회변수 및 출도착시각표 조회변수 초기화
                                currentDepTableFlag, funcStationCnt = updatingFuncStationCnt(cnt=funcStationCnt, depTableTF=currentDepTableFlag, searchToNext=searchingToNextStn)

                            # 6.1.4.4.2.
                            else :
                                # 4.5.6.10. 출발시각표를 조회할 수 있도록 currentDepTableFlag 를 True로 초기화.
                                currentDepTableFlag = True
                                # funcStationCnt는 도착데이터를 조회한 후에만 1씩 증가시키도록 하므로, 여기서는 증가시키지 않음
            
            # 6.2. 둘 중 하나의 시간이라도 없으면
            else : # (funcDepTime_atStnSpN_ip0StnS is not pd.NaT) & (funcDepTime_atStnSpN_ip1StnS is not pd.NaT) :

                if funcErrCntNextSta > minChkCnt :
                    if len(funcArrTrain_ip1_byPreviousStn) > 0 :
                        funcArrTrainNo_atStnS_ip1StnS, funcFeasibleStnId, funcArrDepTimeDataAvailableFlag, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=funcArrTrainNo_atStnS_ip1StnS, initStnId=funcOvertakeStaList[funcOvertakeStation], nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=funcArrTrain_ip1_byPreviousStn, stnIdByPreviousStnList=funcPreviousStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)

                        funcArrDepTimeDataAvailableFlag, funcDepTrainOrderOf_ip1StnS_atStnSpN, funcDepTime_atStnSpN_ip1StnS, funcTimeTable_stnSpN, funcDepTimeDataStation = findFeasibleTrainInfo(targetTrainNo=funcArrTrainNo_atStnS_ip1StnS, nextStnTF=searchingToNextStn, trainNoByPreviousStnList=funcArrTrain_ip1_byPreviousStn, previousStnList=funcPreviousStnList,  arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                        stnS_dataChk = True

                    else :
                        if currentDepTableFlag == True :
                            # 역 조회변수 및 출도착시각표 조회변수 초기화
                            currentDepTableFlag, funcStationCnt = updatingFuncStationCnt(cnt=funcStationCnt, depTableTF=currentDepTableFlag, searchToNext=searchingToNextStn)
                        
                        # 4.5.7.3. 지금 조회한게 도착시각표 정보면
                        else : # if currentDepTableFlag == True :
                            
                            # 4.5.7.4. 출발시각표를 조회할 수 있도록 currentDepTableFlag 를 True로 초기화.
                            currentDepTableFlag = True
                            # funcStationCnt는 도착데이터를 조회한 후에만 1씩 증가시키도록 하므로, 여기서는 증가시키지 않음
                
                # 4.5.7. 하나라도 없으면 while문 오류 상황에 따라 반복시행 시 조건 검토 시행
                else :
                    
                    # 4.5.7.2. 지금 조회한게 출발시각표 정보면
                    if currentDepTableFlag == True :
                        # 역 조회변수 및 출도착시각표 조회변수 초기화
                        currentDepTableFlag, funcStationCnt = updatingFuncStationCnt(cnt=funcStationCnt, depTableTF=currentDepTableFlag, searchToNext=searchingToNextStn)
                    
                    # 4.5.7.3. 지금 조회한게 도착시각표 정보면
                    else : # if currentDepTableFlag == True :
                        
                        # 4.5.7.4. 출발시각표를 조회할 수 있도록 currentDepTableFlag 를 True로 초기화.
                        currentDepTableFlag = True
                        # funcStationCnt는 도착데이터를 조회한 후에만 1씩 증가시키도록 하므로, 여기서는 증가시키지 않음

        else : #if funcStationIdAvailabilty == True :
            if (funcErrCntNextSta > minChkCnt) | (outOfRangeTF == True): 
                if len(funcArrTrain_ip1_byPreviousStn) > 0 :
                    funcArrTrainNo_atStnS_ip1StnS, funcFeasibleStnId, funcArrDepTimeDataAvailableFlag, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=funcArrTrainNo_atStnS_ip1StnS, initStnId=funcOvertakeStaList[funcOvertakeStation],  nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=funcArrTrain_ip1_byPreviousStn, stnIdByPreviousStnList=funcPreviousStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)

                    funcArrDepTimeDataAvailableFlag, funcDepTrainOrderOf_ip1StnS_atStnSpN, funcDepTime_atStnSpN_ip1StnS, funcTimeTable_stnSpN, funcDepTimeDataStation = findFeasibleTrainInfo(targetTrainNo=funcArrTrainNo_atStnS_ip1StnS, nextStnTF=searchingToNextStn, trainNoByPreviousStnList=funcArrTrain_ip1_byPreviousStn, previousStnList=funcPreviousStnList,  arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                    stnS_dataChk = True
                
                else : # 유효범위의 역에 대한 조회가 끝났는데 조회된 열차 정보가 없는 경우, 분석 실패로 처리
                    stnS_dataChk = True
                    funcArrDepTimeDataAvailableFlag = False

            else :              
                # 역 조회변수 및 출도착시각표 조회변수 초기화
                currentDepTableFlag, funcStationCnt = updatingFuncStationCnt(cnt=funcStationCnt, depTableTF=currentDepTableFlag, searchToNext=searchingToNextStn)
        
        # 4.5.7.5. 무한루프에 빠지지 않도록 errCntNextSta를 1 증가
        funcErrCntNextSta = funcErrCntNextSta + 1

    if funcArrDepTimeDataAvailableFlag == True :
        return [funcArrDepTimeDataAvailableFlag, funcDepTrainOrderOf_ip0StnS_atStnSpN, funcDepTime_atStnSpN_ip0StnS, funcDepTrainOrderOf_ip1StnS_atStnSpN, funcDepTime_atStnSpN_ip1StnS, funcTimeTable_stnSpN, funcDepTimeDataStation]
    else :
        return [funcArrDepTimeDataAvailableFlag, funcErrCntNextSta]

# ==========================================================================================

def initOrAppendArrTrinNoSeq(funcArrSeqList, trainNo_ip0, trainNo_ip1) :
    # 1. 열차 도착순서 리스트가 있다면
    if len(funcArrSeqList) > 0 : #in locals():
        # 2. 도착 순서 저장 : 도착 순서는 순차적으로 저장하면 됨
        funcArrSeqList.append(trainNo_ip1)
    
    elif len(funcArrSeqList) == 0 :
        # 3. 기존에 저장된 도착정보가 없다면, 도착정보를 초기화
        funcArrSeqList = [trainNo_ip0, trainNo_ip1]
    
    # 4. 도착순서 리스트 반환
    return funcArrSeqList

# ===========================================================================
def initOrAppendDepTrinNoSeq(funcDepSeqList, trainNo_ip0, trainNo_ip1, funcNumOfOvertake) :
    # 1. 열차 출발순서 리스트가 있다면
    if len(funcDepSeqList) > 0 : # in locals():

        # 2. 출발순서 저장 : 도착 순서는 순차적으로 저장하면 됨
        if funcNumOfOvertake == 0 :

            # 2.1. 검토 대상 S역 i+n번째 열차에 대한 피추월횟수가 0이면, 기존 출발순서 정보 제일 뒤에 덧붙여서 저장
            funcDepSeqList.append(trainNo_ip1)
        
        elif funcNumOfOvertake > 0 :
            # 2.2. 검토 대상 S역 i+n번째 열차에 대한 피추월횟수가 1이면, 
            # 2.2.1. i열차가 출발순서 중 몇번째에 있는지 찾기
            tempDepSeq_trainNo_ip0 = [i for i, value in enumerate(funcDepSeqList) if value == trainNo_ip0]

            # 2.2.2. i열차의 앞 순서까지의 도착 순서를 funcDepSeqListBeforeTrain_ip0 변수에 저장
            if len(tempDepSeq_trainNo_ip0) == 1 :
                funcDepSeqListBeforeTrain_ip0 = funcDepSeqList[0:tempDepSeq_trainNo_ip0[0]]

                # 2.2.3. 추월한 열차의 정보를 덧붙이기
                funcDepSeqListBeforeTrain_ip0.append(trainNo_ip1)

                # 2.2.4. 분석대상 열차의 정보를 제일 마지막에 덧붙이기
                funcDepSeqListBeforeTrain_ip0.append(trainNo_ip0)

                # 2.2.5. 출발순서 변수 갱신
                funcDepSeqList = funcDepSeqListBeforeTrain_ip0
            elif len(tempDepSeq_trainNo_ip0)>1 :
                print("열차 ID 중복 in funcDepSeqList!")
                return False
            else :
                print("열차 ID가 없습니다 in funcDepSeqList")
                return False

    elif len(funcDepSeqList) == 0 :
        # 3. 기존에 저장된 출발순서 정보가 없다면, 도착정보를 초기화
        if funcNumOfOvertake == 0 :

            # 2.1. 검토 대상 S역 i+n번째 열차에 대한 피추월횟수가 0이면, 기존 출발순서 정보 제일 뒤에 덧붙여서 저장
            funcDepSeqList = [trainNo_ip0, trainNo_ip1]
        
        elif funcNumOfOvertake > 0 :
            funcDepSeqList = [trainNo_ip1, trainNo_ip0]
    
    # 4. 출발순서 리스트 반환
    return funcDepSeqList

# ===========================================================================

def findSeqOrderAndAppned(funcTargetSeqList, seqOfCurrentTrain, appendPoint) :
    targetSeqListWithoutFirstArrDepTrain = funcTargetSeqList[0:appendPoint]

    # 해당 List에다가 분석대상열차의 출도착리스트를 붙여넣기
    for trnId in seqOfCurrentTrain :
        targetSeqListWithoutFirstArrDepTrain.append(trnId)
    
    # 전체 출도착열차 순서 List 초기화
    return targetSeqListWithoutFirstArrDepTrain

# ===========================================================================

def appendArrDepTrainSeq(tempSeq, targetSeqList, analysisTrainArrTime) :
    # 0. 전체 출도착순서 List의 길이가 0보다 크면
    if len(targetSeqList) > 0 :

        # 1. 분석대상열차에 대한 출도착 순서 리스트의 길이 확인 : 0보다 크면
        if len(tempSeq) > 0:

            # 2. 분석대상열차에 대한 출도착순서 리스트 중 몇개의 열차가 전체 출도착순서 리스트에 있는지 확인하기
            numOfTrainInTargetSeqList = 0
            trainArrDepOrder_inSeqList = [] # 분석대상 열차에 대한 출도착순서 리스트 중 검색중인 열차가 전체 출도착순서 리스트의 몇번째에 위치하는지 저장할 list
            trainArrDepOrder_inTempSeqList = [] # 분석대상 열차에 대한 출도착순서 리스트의 index 저장용 list  

            for i in range(0,len(tempSeq)) : 
                # 2.1. tempSeqTrainId가 전체 출도착순서 리스트에 있는지 확인
                tempTrainArrDepOrder_inSeqList = [j for j, value in enumerate(targetSeqList) if value == tempSeq[i]]
                
                # 2.2. 만약 전체 출도착순서 리스트에 1개 열차만 있다면 (정상적이라면 1개만 있어야 함)
                if len(tempTrainArrDepOrder_inSeqList) == 1 :
                    # 2.2.1. 중복 열차갯수 갱신
                    numOfTrainInTargetSeqList = numOfTrainInTargetSeqList + 1
                    
                    # 2.2.2. 해당 열차가 전체 리스트에서 몇 번째 순서인지 저장
                    if len(trainArrDepOrder_inSeqList) == 0 :
                        trainArrDepOrder_inSeqList = tempTrainArrDepOrder_inSeqList # 전체 출도착순서 리스트에 대한 검색대상열차의 순서 저장
                        trainArrDepOrder_inTempSeqList = [i] # 검색대상열차가 분석대상열차의 출도착순서리스트 중 몇번째였는지 저장
                    else :
                        trainArrDepOrder_inSeqList.append(tempTrainArrDepOrder_inSeqList[0])
                        trainArrDepOrder_inTempSeqList.append(i)
                
                # 2.3. 만약 전체 출도착순서 리스트에 여러개의 열차 ID가 검색된다면
                elif len(tempTrainArrDepOrder_inSeqList) > 1 :
                    print("열차 ID 중복! in targetSeqList")
                    
                    # 2.3.1. 중복 열차갯수 갱신
                    numOfTrainInTargetSeqList = numOfTrainInTargetSeqList + len(tempTrainArrDepOrder_inSeqList)
                    
                    # 2.3.2. 해당 열차가 전체 리스트에서 몇 번째 순서인지 저장 : 전체 출도착순서 리스트에서 첫번째로 중복되는 순서만 저장
                    if len(trainArrDepOrder_inSeqList) == 0 :
                        trainArrDepOrder_inSeqList.append(tempTrainArrDepOrder_inSeqList[0])
                        trainArrDepOrder_inTempSeqList = [i]
                    else :
                        trainArrDepOrder_inSeqList.append(tempTrainArrDepOrder_inSeqList[0])
                        trainArrDepOrder_inTempSeqList.append(i)

            # 분석대상열차에 대한 출도착순서 리스트 중 1개 열차만 전체 출도착순서 리스트에 포함되어있는 경우
            if numOfTrainInTargetSeqList == 1 :
                # 출도착순서 리스트 갱신
                targetSeqList = findSeqOrderAndAppned(funcTargetSeqList=targetSeqList, seqOfCurrentTrain=tempSeq, appendPoint=trainArrDepOrder_inSeqList[0])
                
            else :
                if (len(trainArrDepOrder_inSeqList) == 2) & (len(tempSeq) == 2) :
                    if trainArrDepOrder_inSeqList[1] - trainArrDepOrder_inSeqList[0] != trainArrDepOrder_inTempSeqList[1] - trainArrDepOrder_inTempSeqList[0] :
                        if trainArrDepOrder_inSeqList[1] - trainArrDepOrder_inSeqList[0] > trainArrDepOrder_inTempSeqList[1] - trainArrDepOrder_inTempSeqList[0] :
                            trainArrDepOrder_inTempSeqList[0] #print("전체 출도착순서 리스트 검색 결과 분석대상열차 출도착순서 리스트의 순서 사이에 다른 열차가 있습니다")
                        else :
                            # 분석대상열차의 출도착리스트에 전체 출도착순서 리스트에는 없는 열차가 포함되어있어, 이를 포함하여 열차 삽입
                            if (trainArrDepOrder_inSeqList[1] > trainArrDepOrder_inSeqList[0]) & (trainArrDepOrder_inTempSeqList[1] > trainArrDepOrder_inTempSeqList[0]) :
                                targetSeqList = findSeqOrderAndAppned(funcTargetSeqList=targetSeqList, seqOfCurrentTrain=tempSeq, appendPoint=trainArrDepOrder_inSeqList[0])

                            else :
                                if analysisTrainArrTime is not pd.NaT : # 분석대상열차의 도착시간이 NaN이면 해당역에서 조회할 수 있는 열차는 다 조회했다는 말. 
                                    print("오류점검 필요! 전체 출도착순서 리스트의 순서와 분석대상 열차의 출도착순서 리스트의 순서가 다름" )

                elif (len(trainArrDepOrder_inSeqList) == 2) & (len(tempSeq) > 2) :
                    if trainArrDepOrder_inSeqList[1] - trainArrDepOrder_inSeqList[0] != trainArrDepOrder_inTempSeqList[1] - trainArrDepOrder_inTempSeqList[0] :
                        if trainArrDepOrder_inSeqList[1] - trainArrDepOrder_inSeqList[0] > trainArrDepOrder_inTempSeqList[1] - trainArrDepOrder_inTempSeqList[0] :
                            trainArrDepOrder_inTempSeqList[0] ##print("전체 출도착순서 리스트 검색 결과 분석대상열차 출도착순서 리스트의 순서 사이에 다른 열차가 있습니다")
                        else :
                            # 분석대상열차의 출도착리스트에 전체 출도착순서 리스트에는 없는 열차가 포함되어있어, 이를 포함하여 열차 삽입
                            if (trainArrDepOrder_inSeqList[1] > trainArrDepOrder_inSeqList[0]) & (trainArrDepOrder_inTempSeqList[1] > trainArrDepOrder_inTempSeqList[0]) :
                                targetSeqList = findSeqOrderAndAppned(funcTargetSeqList=targetSeqList, seqOfCurrentTrain=tempSeq, appendPoint=trainArrDepOrder_inSeqList[0])

                            else :
                                if analysisTrainArrTime is not pd.NaT : # 분석대상열차의 도착시간이 NaN이면 해당역에서 조회할 수 있는 열차는 다 조회했다는 말. 
                                    print("오류점검 필요! 전체 출도착순서 리스트의 순서와 분석대상 열차의 출도착순서 리스트의 순서가 다름" )
                    else : 
                        targetSeqList = findSeqOrderAndAppned(funcTargetSeqList=targetSeqList, seqOfCurrentTrain=tempSeq, appendPoint=trainArrDepOrder_inSeqList[0])
                
                elif len(trainArrDepOrder_inSeqList) >= 3 :
                    if trainArrDepOrder_inSeqList[2] - trainArrDepOrder_inSeqList[0] != trainArrDepOrder_inTempSeqList[2] - trainArrDepOrder_inTempSeqList[0] :
  
                        # 분석대상 열차의 출도착순서 리스트에 속한 열차 중 전체 출도착열차 순서 List에서 가장 앞에 있는 열차는 몇번째에 있는지 저장
                        minOrderInSeqList = [j for j, value in enumerate(trainArrDepOrder_inSeqList) if value == min(trainArrDepOrder_inSeqList)][0] 
                        
                        # 만약 분석대상 열차의 출도착순서 리스트 중 첫번째 열차가 전체 출도착열차 순서에서 가장 앞에 있는게 아니라면 = 중간에 NA인 열차가 있어서 누락된 부분때문에 전체 출도착열차 순서 List가 잘못된 경우
                        if minOrderInSeqList > 0 :
                            # 순서 조정해서 출도착순서 리스트 갱신
                            targetSeqList = findSeqOrderAndAppned(funcTargetSeqList=targetSeqList, seqOfCurrentTrain=tempSeq, appendPoint=trainArrDepOrder_inSeqList[minOrderInSeqList])
                        else :
                            #print("전체 출도착순서 리스트와 분석대상열차 출도착순서 리스트의 순서가 다릅니다")
                            print("3개 이상의 열차가 전체 리스트에 포함! in targetSeqList")
                            #return False

                elif len(trainArrDepOrder_inSeqList) == 0 : 
                    # 분석대상 열차의 출도착순서 리스트에 속한 열차들이 전체 출도착순서 리스트에는 없는 경우
                    # 합쳐나감
                    for trnId in tempSeq :
                        targetSeqList.append(trnId)
                else :
                    print("오류 점검 필요! in targetSeqList")
                    print("tempSeq: "+str(tempSeq))
                    print("targetSeqList: "+str(targetSeqList))
                    return False
        
        #else : # if len(tempSeq) > 0:
            # 만약 중간에 데이터 결측으로 인해 출도착순서가 빵구가 날 경우 NaN을 삽ㅂ입
            #targetSeqList.append(np.NaN)


    # 6. 전체 출도착순서 List의 길이가 0이면 
    else :
        # 7. 전체 출도착순서 List에다가 분석대상열차의 출도착리스트를 붙여넣기
        for trnId in tempSeq :
                targetSeqList.append(trnId)

    # 8. 전체 출도착순서 List 반환
    return targetSeqList

# ===========================================================================

def findColIdOfrtTableByStn (rtTable, stationIdInteger) :
    return np.where(rtTable.columns == stationIdInteger)[0][0]

# ===========================================================================

def returnSortedrtTableByStn (rtTable, rtTableByStn, stationIdInteger) :
    colId = np.where(rtTable.columns == stationIdInteger)[0][0]
    return rtTableByStn[colId]

# ===========================================================================

def appendTwice (targetList, appendingValue) :
    targetList.append( appendingValue )
    targetList.append( appendingValue )
    return targetList

# ===========================================================================

def updatingTempArrDepSeq (funcTempArrSeq, funcTempDepSeq, funcTrainNo_ip0, funcTrainNo_ip1, numOfOvertake, funcOvertakedTrnIdList):
    
    # temArrSeq의 마지막에 분석대상 열차 정보가 저장되어있지 않다면 
    if len(funcTempArrSeq) > 0 : 
        if funcTempArrSeq[(len(funcTempArrSeq)-1)] != funcTrainNo_ip1 :
            # 도착 순서 저장
            funcTempArrSeq = initOrAppendArrTrinNoSeq(funcArrSeqList=funcTempArrSeq, trainNo_ip0=funcTrainNo_ip0, trainNo_ip1=funcTrainNo_ip1)

            # 출발 순서 저장
            funcTempDepSeq = initOrAppendDepTrinNoSeq(funcDepSeqList=funcTempDepSeq, trainNo_ip0=funcTrainNo_ip0, trainNo_ip1=funcTrainNo_ip1, funcNumOfOvertake=numOfOvertake)

            # 추월열차 ID 누적해서 저장하기
            if numOfOvertake >= 1 :
                funcOvertakedTrnIdList.append(funcTrainNo_ip1)
    else : 
        # 도착 순서 저장
        funcTempArrSeq = initOrAppendArrTrinNoSeq(funcArrSeqList=funcTempArrSeq, trainNo_ip0=funcTrainNo_ip0, trainNo_ip1=funcTrainNo_ip1)

        # 출발 순서 저장
        funcTempDepSeq = initOrAppendDepTrinNoSeq(funcDepSeqList=funcTempDepSeq, trainNo_ip0=funcTrainNo_ip0, trainNo_ip1=funcTrainNo_ip1, funcNumOfOvertake=numOfOvertake)

        # 추월열차 ID 누적해서 저장하기
        if numOfOvertake >= 1 :
            funcOvertakedTrnIdList.append(funcTrainNo_ip1)
    
    # 리스트 반환
    return funcTempArrSeq, funcTempDepSeq, funcOvertakedTrnIdList

# ===========================================================================

def creatingRtTimetable(myLine_rtData, lineNm, dataType, dataTypeFull, keyStations, operDist, stationOrderChk, basePath, dataPath, outputPath, oDate, pcNm, nFiles, fwDate, tTableDate):   
    # 2. 데이터 수신시각 Datetime 변환 ( 및 신분당선 열차번호 부여 )
    # 2.1. 신분당선이 아니면, 데이터수신시각을 datetime으로 변경 (열차운영번호 부여 알고리즘 생략)
    if lineNm!="LineS":
        myd1 = myLine_rtData
        #myd1['recptnDt'] = pd.to_datetime(myd1['recptnDt'], format="%Y-%m-%d %H:%M:%S") # 데이터 생성시각을 datetime 변수로 변환

        # datetime으로 변경
        if type(myd1.recptnDt[0])==str:
            myd1['recptnDt'] = pd.to_datetime(myd1['recptnDt'], format="%Y-%m-%d %H:%M:%S") 
        
        myd2 = myd1
        myd2['trainNo2'] = myd2['trainNo']  # trainNo2 열 생성. 신분당선 데이터는 trainNo에 편성번호가 들어가 있어, 열차번호 부여 알고리즘을 통해 trainNo2에 열차번호를 부여하는 구조임. 데이터 구조 통일을 위해, 신분당선이 아닌 데이터에는 trainNo를 복제해서 trainNo2를 생성함.
        #print("\n myd2 \n")
        #print(myd2)
        

    # 2.2. 신분당선이면, 데이터수신시각0 datetime 변경 + 열차번호 할당 알고리즘 시작
    elif lineNm=="LineS":
        #myvalues = myLine_rtData.trainNo.unique()   # 해당 일 운행 편성번호 확인하기
        myd1 = myLine_rtData
        #myd1['recptnDt'] = pd.to_datetime(myd1['recptnDt'], format="%Y-%m-%d %H:%M:%S") # 데이터 생성시각을 datetime 변수로 변환

        # 2.2.1. datetime으로 변경
        if type(myd1.recptnDt.iloc[0])==str:
            myd1['recptnDt'] = pd.to_datetime(myd1['recptnDt'], format="%Y-%m-%d %H:%M:%S") 

        # 2.2.2. 해당일 운행 편성번호와 운행순서를 기준으로 4자리 가상 열차번호 생성
        myd2 = df()
        myd2 = addingArtificialTrainNo(myd1)
        

    # 2.3. 분기가 있는 노선에 대한 undnLine 세분화 : 지선운행여부 구분 변수(isBranch) 추가
    # 2.3.1. 분기정보 저장하 isBranch 열 추가
    ## 현재는 1호선의 경우만 binary로 값을 제공하며, 그 외의 노선의 경우 목적지역 ID를 그대로 할당함
    # 2.3.1.1. 분기정보 저장할 isBranch column 저장
    myd2['isBranch'] = myd2['old_statnTid'] # myTrains_rtPos['bnd'] = myTrains_rtPos['updnLine'] 
    
    # 2.3.1.2. 데이터수신시각 기준으로 sorting후 열차ID와 행선지, 지선운행여부 구분 정보 작성을 위한 별도의 변수로 저장
    mytrains_rtPos = rtDataSortingAndDropDuplicates(myd2, "trainNo2")# myd2.sort_values(by='recptnDt', ascending=True) 

    # 2.3.1.3. 막차 정보 저장 : 중복 제거 과정에서 막차 정보는 고려하지 않고, 차후에 다시 넣을것이기 때문
    lstcarTrainNo2List = mytrains_rtPos.trainNo2.iloc[np.where(mytrains_rtPos.lstcarAt==1)].unique()
    
    # 2.3.1.4. 열차번호, 상하행정보, 급행정보를 기준으로 중복 행 제거
    mytrains_rtPos = mytrains_rtPos[["trainNo", "trainNo2", "updnLine", "statnTnm", "old_statnTid", "statnTid", "recptnDt", "old_statnId", "statnId", "directAt", "lstcarAt", "isBranch"]].drop_duplicates(["trainNo2", "updnLine", "directAt"]) #, "lstcarAt"])  
    
    # 2.3.1.5. 막차정보 복원
    if len(lstcarTrainNo2List) > 0 :
        lstcar = lstcarTrainNo2List[0]
        for lstcar in lstcarTrainNo2List :
            currentLstcarId = np.where(mytrains_rtPos.trainNo2 == lstcar)
            if len(currentLstcarId) == 1 :
                mytrains_rtPos.lstcarAt.iloc[currentLstcarId] = 1
            else : 
                print("There are duplicated trainNo2!")
                break

    # 2.3.2. 지선정보 Binary 값 추가 (1호선에만 해당)
    if lineNm == "Line1" :
        ## 2.3.2. 행선지 또는 출발지 정보를 기준으로 지선운행여부 확인하기
        for i in range(0, len(mytrains_rtPos.index)):
            currStatnTid = str(mytrains_rtPos['old_statnTid'].iloc[i])    # 해당 열차의 목적지 행 세분화
            currUpdnLine = mytrains_rtPos['updnLine'].iloc[i]         #
            currStatnId = str(mytrains_rtPos['old_statnId'].iloc[i])    # 해당 열차의 목적지 행 세분화
            
            if i%50 == 0:
                print(i)

            if (int(currStatnId[len(currStatnTid)-5])==0) & (int(currStatnTid[len(currStatnTid)-5])==0): # 종착역이나 현재역(해당열차 별 가장 빠른 데이터) 정보가 지선이 아닐경우
                if (int(currStatnId[len(currStatnTid)-6])==0) & (int(currStatnTid[len(currStatnTid)-6])==0): # 서동탄역은 6번째 자리에 있었음..
                    mytrains_rtPos['isBranch'].iloc[i] = 0
                elif (int(currStatnId[len(currStatnTid)-6])>0) | (int(currStatnTid[len(currStatnTid)-6])>0): # 종착역이나 현재역(해당열차 별 가장 빠른 데이터) 정보가 지선일 경우
                    mytrains_rtPos['isBranch'].iloc[i] = 1
            elif (int(currStatnId[len(currStatnTid)-5])>0) | (int(currStatnTid[len(currStatnTid)-5])>0): # 종착역이나 현재역(해당열차 별 가장 빠른 데이터) 정보가 지선일 경우
                mytrains_rtPos['isBranch'].iloc[i] = 1
            else:
                print("currUpdnLine and currStatnTid condition has not been met. currStatnTid: "+str(currStatnTid)+", currUpdnLine : "+str(currUpdnLine))

        print(mytrains_rtPos)
    
    mytrains_rtPos.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bndA-"+dataType+"_trainsInfo-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)


    # -----
    # 3. 실시간 열차 출발&도착&접근 시각표 및 점유시간표 작성

    #type(myd2['recptnDt'])
    myd2 = rtDataSortingAndDropDuplicates(myd2, "trainNo2")# myd2.sort_values(by='recptnDt', ascending=True)

    for bnd in range(0,2): # 상하행 별 접근&도착&출발 시각표 만들기
        
        myd3 = myd2[myd2.updnLine==bnd] # 방향 별 데이터 추출해서 myd3에 저장
        
        if(len(myd3.index)>1): ## Pivottable을 만들 정도의 데이터가 있는지 체크
            myd3.head(n=5)

            # 3.1.0. myd3에서 역별 열차별 접근 / 도착 / 출발의 시각 기록이 논리적으로 이상 있는지 여부를 확인 --> 이상이 있는 경우, 해당 행 삭제
            delRowInfoList = [] # 정보를 삭제해야 하는 행을 append
            tmpTrnIdList = myd3['trainNo2'].unique()
            for tmpTrnId in tmpTrnIdList: # tmpTrnId = tmpTrnIdList[51]
                tmpCheckDf = myd3.loc[myd3['trainNo2']==tmpTrnId]
                tmpStnIdList = tmpCheckDf['statnId'].unique()
                for tmpStnId in tmpStnIdList: # tmpStnId = tmpStnIdList[4]
                    tmpCheckDf2 = tmpCheckDf.loc[tmpCheckDf['statnId']==tmpStnId]
                    for tmpRow in range(tmpCheckDf2.shape[0]-1): # tmpRow = 1
                        if tmpCheckDf2.iloc[tmpRow,3] > tmpCheckDf2.iloc[tmpRow+1,3]:
                            delRowInfoList.append(tmpCheckDf2.index[tmpRow])

            # myd3Backup = myd3
            # myd3 = myd3Backup
            myd3 = myd3.loc[np.array(list(map(lambda x: x not in delRowInfoList, myd3.index)))]

            # myd3.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_myd3-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)

            # 3.1.1. 접근데이터 추출 및 실시간 열차 접근 기초시각표 만들기
            myd3_app_ttable = creatingOperLogTable(myd3, 0)

            # 3.1.2. 도착데이터 추출 및 실시간 열차 도착 기초시각표 만들기
            myd3_arr_ttable = creatingOperLogTable(myd3, 1)

            # 3.1.3. 출발데이터 추출 및 실시간 열차 출발 기초시각표 만들기
            myd3_dep_ttable = creatingOperLogTable(myd3, 2)
            
            # 3.2. 접근&출발&도착 시각표가 같은 행 갯수를 갖도록 보정
            ## 데이터 결측치가 각 열차 별, 역 별로 균등하게 발생하지 않으므로 행 갯수(열차정보)와
            ## 열 갯수(역 정보) 등을 맞출 필요가 있음

            # 3.2.1 app, arr, dep 모두 table이 없는 지 확인하는 변수 초기화
            ttableErr = 0 

            # 3.2.2 app, arr, dep에 대한 기초 table 존재여부를 확인하여 행&열 정보 병합
            if( str(type(myd3_app_ttable))=="<class 'int'>" ):
                # app - X
                if( str(type(myd3_arr_ttable))=="<class 'int'>" ):
                    # app - X, arr - X
                    if( str(type(myd3_dep_ttable))=="<class 'int'>" ): 
                        # app - X, arr - X, dep - X
                        ttableErr = -1

                    else : # if( str(type(myd3_dep_ttable))=="<class 'int'>" ): 
                        # app - X, arr - X, dep - O
                        merged_trains = myd3_dep_ttable.index
                        merged_statnId = myd3_dep_ttable.columns
                
                else : #if( str(type(myd3_arr_ttable))=="<class 'int'>" ): 
                    # app - X, arr - O
                    if( str(type(myd3_dep_ttable))=="<class 'int'>" ): 
                        # app - X, arr - O, dep - X
                        merged_trains = myd3_arr_ttable.index
                        merged_statnId = myd3_arr_ttable.columns

                    else : # if( str(type(myd3_dep_ttable))=="<class 'int'>" ): 
                        # app - X, arr - O, dep - O
                        merged_trains = myd3_arr_ttable.index.append(myd3_dep_ttable.index) # 접근&도착 시각표의 열차 정보(index) 병합
                        merged_statnId = myd3_arr_ttable.columns.append(myd3_dep_ttable.columns) # 접근&도착 시각표의 역 정보(columns) 병합

            else : # if( str(type(myd3_app_ttable))=="<class 'int'>" ):
                # app - O
                if( str(type(myd3_arr_ttable))=="<class 'int'>" ):
                    # app - O, arr - X
                    if( str(type(myd3_dep_ttable))=="<class 'int'>" ): 
                        # app - O, arr - X, dep - X
                        merged_trains = myd3_app_ttable.index
                        merged_statnId = myd3_app_ttable.columns

                    else : # if( str(type(myd3_dep_ttable))=="<class 'int'>" ): 
                        # app - O, arr - X, dep - O
                        merged_trains = myd3_app_ttable.index.append(myd3_dep_ttable.index) # 접근&도착 시각표의 열차 정보(index) 병합
                        merged_statnId = myd3_app_ttable.columns.append(myd3_dep_ttable.columns) # 접근&도착 시각표의 역 정보(columns) 병합

                else : #if( str(type(myd3_arr_ttable))=="<class 'int'>" ): 
                    # app - O, arr - O
                    if( str(type(myd3_dep_ttable))=="<class 'int'>" ): 
                        # app - O, arr - O, dep - X
                        merged_trains = myd3_app_ttable.index.append(myd3_arr_ttable.index) # 접근&도착 시각표의 열차 정보(index) 병합
                        merged_statnId = myd3_app_ttable.columns.append(myd3_arr_ttable.columns) # 접근&도착 시각표의 역 정보(columns) 병합

                    else : # if( str(type(myd3_dep_ttable))=="<class 'int'>" ): 
                        # app - O, arr - O, dep - O
                        merged_trains = myd3_app_ttable.index.append(myd3_arr_ttable.index) # 접근&도착 시각표의 열차 정보(index) 병합
                        merged_trains = merged_trains.append(myd3_dep_ttable.index) # 병합된 접근&도착 시각표의 열차 정보(index)에 출발 시각표의 열차 정보(index) 병합

                        merged_statnId = myd3_app_ttable.columns.append(myd3_arr_ttable.columns) # 접근&도착 시각표의 역 정보(columns) 병합
                        merged_statnId = merged_statnId.append(myd3_dep_ttable.columns) # 병합된 접근&도착 시각표의 역 정보(columns)에 출발 시각표의 역 정보(columns) 병합


            # 3.2.2 app, arr, dep 모두 table이 없는 지 확인 후 고유값으로 추출
            if ttableErr == -1 : # 모두 다 없으면
                print("ttable Err == -1") # 오류메세지 출력

            else : ## 1개 이상 정보가 있으면
                merged_trains = merged_trains.unique() # 열차 정보는 고유값만 남기기
                merged_trains = df(index=merged_trains) # 열차 정보 - dataframe으로 변환
                merged_statnId = merged_statnId.unique().sort_values() # 고유값만 남기기

                # 각 방향별 역 운영순서에 맞춰 역 코드를 정렬 : 일부 노선(공항철도)의 경우 역 코드의 integer 순서가 역 운행순서와 다르기때문에 이 과정이 필요함
                if bnd == 0:
                    temp_merged_statnId = operDist[['statnId','bnd0_statnId']] # bnd0 방향 역 순서 정보와 역 코드 발췌
                    temp_merged_statnId = temp_merged_statnId.sort_values('bnd0_statnId', ascending=True) # bnd0 방향 역 순서 정보(bnd0_statnId열)을 활용해 정렬
                    merged_statnId = temp_merged_statnId['statnId'].unique() # 정렬한 데이터를 merged_statnId에 병합
                    
                elif bnd == 1:
                    temp_merged_statnId = operDist[['statnId','bnd1_statnId']] # bnd1 방향 역 순서 정보와 역 코드 발췌
                    temp_merged_statnId = temp_merged_statnId.sort_values('bnd1_statnId', ascending=True) # bnd1방향 역 순서 정보(bnd1_statnId열)을 활용해 정렬
                    merged_statnId = temp_merged_statnId['statnId'].unique() # 정렬한 데이터를 merged_statnId에 병합


                # 2020-10-15~2021-03-17 사이에 수집된 데이터는 도착 정보의 역 코드가 한개씩 밀려있어서 이를 보정해야 함
                if lineNm=="LineA":                        
                    if (myd3.iloc[0,2] > pd.to_datetime('2020-10-15 03:00:00', format="%Y-%m-%d %H:%M:%S")) and (myd3.iloc[0,2] < pd.to_datetime('2021-03-18 03:00:00', format="%Y-%m-%d %H:%M:%S")):
                    #if (myd3_arr_ttable.iloc[1,1] > pd.to_datetime('2020-10-15 03:00:00', format="%Y-%m-%d %H:%M:%S")) and (myd3_arr_ttable.iloc[1,1] < pd.to_datetime('2021-03-18 03:00:00', format="%Y-%m-%d %H:%M:%S")):
                        
                        temp_arr_ttable_colNames = []                                       # 도착테이블 열 이름 보정용 빈 리스트 생성
                        
                        # 도착정보에 대해 역 코드가 들어간 열 이름을 변경하기 (반복문 활용)
                        for tempStnId in myd3_arr_ttable.columns :
                            currentStnOrderList = np.where(merged_statnId==tempStnId)[0]    # 현재 역 코드가 merged_statnId 상 몇번째인지 찾기

                            if len(currentStnOrderList) == 1 :                              # merged_statnId 안에 있으면 (위 문제에 해당하는 기간에는 종착역 코드 정보가 없는 상태)
                                currentStnOrder = currentStnOrderList[0]

                                if currentStnOrder < (len(merged_statnId)-1) :
                                    temp_arr_ttable_colNames.append(merged_statnId[currentStnOrder+1])
                                else :
                                    print("마지막 역이므로 다음 역 코드를 부여할 수 없습니다. [공항철도 도착정보 역 코드 밀림 수정]")
                        
                        if len(temp_arr_ttable_colNames) == len(myd3_arr_ttable.columns) :
                            myd3_arr_ttable.columns = temp_arr_ttable_colNames
                            print("oDate:"+ oDate + " || 공항철도 도착정보 역 코드 밀림 수정 절차가 완료되었습니다 ")
                            
                        else :
                            print("대체할 역 코드 갯수가 일치하지 않습니다. [공항철도 도착정보 역 코드 밀림 수정] || " +  datetime.now())
                        
                        del(temp_arr_ttable_colNames)
                

                # 3.2.3 merged_trains에 포함된 열차정보 추출
                ## 해당일의 모든 운행열차 정보를 저장한 mytrains_rtPos와 merged_trains를 병합하여 열차 관련된 정보만 추출
                tempTrainsInfo = mytrains_rtPos[['trainNo2', 'updnLine', 'statnTid', 'statnTnm', 'isBranch', 'directAt', 'lstcarAt']]
                
                # 3.2.3.1. 방향 별 종점역 정보 저장
                if lineNm == 'LineWS':
                    if bnd == 0 :
                        targetStatnTid = 4713
                    elif bnd == 1 :
                        targetStatnTid = 4701
                        
                    # 3.2.3.2. 종점역과 방향정보 논리구조 적합한 행 정보 추출
                    trueTrnIdSet = set(np.where(tempTrainsInfo.updnLine == bnd)[0]).intersection(np.where(tempTrainsInfo.statnTid == targetStatnTid)[0])  # 터미널역 ID와 updnLine 논리구조 체크 : 우이신설 신설동4713행-undn0, 북한산우이4701-updn1
                    
                    # 3.2.3.3. 적합한 행 정보로 방향 별 열차운행정보 저장
                    tempTrainsInfo = tempTrainsInfo.iloc[list(trueTrnIdSet)]
                    tempTrainsInfo = tempTrainsInfo.sort_values("trainNo2", ascending=True)

                    
                trainsInfo = pd.merge(df(merged_trains), tempTrainsInfo, on='trainNo2') 

                ## 파일로 출력
                trainsInfo.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_trainsInfo-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)

            
                # 3.2.3 app, arr, dep 각각 table이 없는 지 확인하는 변수 초기화
                appTimetableErr = 0
                arrTimetableErr = 0
                depTimetableErr = 0
                
                # 3.2.3.1. app_ttable 있는지 확인 후 저장
                if( str(type(myd3_app_ttable))=="<class 'int'>" ):
                    appTimetableErr = -1
                    print(oDate+"-"+lineNm+"_"+dataType+"-bnd"+str(bnd)+"-appTimetableErr == -1. rtPos_app_ttable.csv_had_not_been_created."+"\n") 
                    f = open(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_app_ttable_csv_had_not_been_created-pc_"+pcNm+"-"+fwDate+".txt", mode='wt', encoding='utf-8')
                    f.write(oDate+"-"+lineNm+"_"+dataType+"-bnd"+str(bnd)+"-appTimetableErr == -1. rtPos_app_ttable.csv_had_not_been_created."+"\n") 
                    f.close()
                else :
                    ### approach : fitTrainsAndStations 함수를 통해 열차데이터-역 데이터를 일 단위로 통일시킴
                    myd3_app_ttable = fitTrainsAndStations(myd3_app_ttable, merged_trains, merged_statnId)#, trainsInfo)
                    #myd3_app_ttable #= statnIdNm['statnId']  # new_statnId 열 생성
                    myd3_app_ttable = pd.merge(trainsInfo, myd3_app_ttable, left_on='trainNo2', right_on=myd3_app_ttable.index) # 열차번호 기준의 열차정보 병합
                    
                    # 파일 출력
                    myd3_app_ttable.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_app_ttable-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)

                if( str(type(myd3_arr_ttable))=="<class 'int'>" ):
                    arrTimetableErr = -1
                    print(oDate+"-"+lineNm+"_"+dataType+"-bnd"+str(bnd)+"-arrTimetableErr == -1. rtPos_arr_ttable.csv_had_not_been_created."+"\n") 
                    f = open(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_arr_ttable_csv_had_not_been_created-pc_"+pcNm+"-"+fwDate+".txt", mode='wt', encoding='utf-8')
                    f.write(oDate+"-"+lineNm+"_"+dataType+"-bnd"+str(bnd)+"-arrTimetableErr == -1. rtPos_arr_ttable.csv_had_not_been_created."+"\n") 
                    f.close()
                else :
                    ### arrival
                    myd3_arr_ttable = fitTrainsAndStations(myd3_arr_ttable, merged_trains, merged_statnId)#, trainsInfo)


                if( str(type(myd3_dep_ttable))=="<class 'int'>" ): 
                    bufferForMakingDepTimeByArrTime = datetime.strptime("00:01:30", "%H:%M:%S")-datetime.strptime("00:00:00", "%H:%M:%S")
                    myd3_dep_ttable = myd3_arr_ttable + bufferForMakingDepTimeByArrTime
                    myd3_dep_ttable = fitTrainsAndStations(myd3_dep_ttable, merged_trains, merged_statnId)#, trainsInfo)

                    #depTimetableErr = -1
                    print(oDate+"-"+lineNm+"_"+dataType+"-bnd"+str(bnd)+"-depTimetableErr == -1. rtPos_dep_ttable.csv_had_been_created_by_adding_buffer_to_rtPos_arr_ttable."+"\n") 
                    f = open(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_dep_ttable_csv_had_been_created_by_estimation-pc_"+pcNm+"-"+fwDate+".txt", mode='wt', encoding='utf-8')
                    f.write(oDate+"-"+lineNm+"_"+dataType+"-bnd"+str(bnd)+"-depTimetableErr == -1. rtPos_dep_ttable.csv_had_been_created_by_adding_buffer_to_rtPos_arr_ttable."+"\n") 
                    f.close()
                else :
                    ### departure
                    myd3_dep_ttable = fitTrainsAndStations(myd3_dep_ttable, merged_trains, merged_statnId)#, trainsInfo)


                # 4. 역별 열차별 점유시각 테이블 만들기 및 피추월횟수 계산하기
                #print(myd3_dep_ttable.iloc[0,0])
                #type(myd3_dep_ttable.iloc[0,0])

                if ((depTimetableErr != -1) & (arrTimetableErr != -1))==True :
                    # 4.1. 장내점유시간 계신하기
                    myd3_occ = myd3_dep_ttable - myd3_arr_ttable
                    #print("myd3_staOccTime \n")
                    #print(myd3_occ)

                    # 4.2. staOccTime이 계산된 요소에 대해 '초'단위로 값 변환하기
                    for i in range(0,len(myd3_occ.index)):
                        nonNanOcc = np.where(pd.isnull(myd3_occ.iloc[i:i+1,:])==False)[1] # NaN or NaT가 아닌 요소를 행 단위로 검색
                        if len(nonNanOcc)>0: 
                            myd3_occ.iloc[i:i+1,nonNanOcc] = myd3_occ.iloc[i:i+1,nonNanOcc].apply(lambda x: x/dt.timedelta(seconds=1)) # 1초에 해당하는 datetime 변수로 나눠서 초 단위 int로 환산
                        del(nonNanOcc)

                    # 4.3. 점유시간 데이터 분포 만들기
                    stations = myd3_occ.columns[0:len(myd3_occ.columns)] 

                    plt.rcParams["figure.figsize"] = (20,7)

                    for station in stations:
                        ## Draw the density plot
                        naRmVec = myd3_occ[station].iloc[ np.where(pd.isnull(myd3_occ[station])==False) ] 
                        sns.distplot(naRmVec, hist = False, kde = True,
                                    kde_kws = {'linewidth': 2},
                                    label = station)
                        
                    # 4.4. 장내 점유시간 그리기
                    # 4.4.1. Plot formatting
                    plt.legend(prop={'size': 16}, title = 'Occupied time at each station')
                    plt.title('Density Plot with Each Stations')
                    plt.xlabel('Occupied time (sec)')
                    plt.ylabel('Density')

                    # 4.4.2. Saving the plot
                    plt.savefig(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_occDensity-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".png", dpi=300)
                    plt.close()

                    
                    # 4.5. 피추월횟수 계산
                    print(myd3_arr_ttable.columns)
                    #if lineNm == "Line9" #| lineNm == "LineA" 

                    # 4.5.1. 피추월횟수 연산을 위한 기본정보 저장
                    # 4.5.1.1. 추월가능역 정보 저장
                    # 4.5.1.1.1. 피추월횟수 진행여부 TF변수 초기화 및 역 별로 정렬된 출도착시각표를 만들기 위한 빈 리스트 생성
                    overtakeAnalysisTF = False
                    arrTableByStn = []      # = []와 = list() 는 같은 역할을 함
                    depTableByStn = list()

                    # 4.5.1.1.2. 노선 별 피추월회숫 관련 정보 초기화
                    if lineNm == "Line9" :
                        # 4.5.1.1.2.1. 방향 별 피추월역 정보 초기화 : 기점역에서 종점역 순으로 기재할 것!
                        if bnd == 0 : 
                            overtakeStaList = [934, 931, 928, 925, 924, 920, 916, 912, 907, 905]    #list(reversed(overtakeStaList)) #  bnd1의 순서를 뒤집었음
                        else :
                            overtakeStaList = [905, 907, 912, 916, 920, 924, 928, 931, 934]         # bnd1 (중앙보훈병원) 방향 정보

                        # 4.5.1.1.2.3. 피추월횟수 진행 TF변수 초기화
                        overtakeAnalysisTF = True

                        # 4.5.1.1.2.4. stnS_dataChkErrCnt 변수 기준값 설정
                        max_stnS_dataChkErrCnt = 5
                    
                    #elif lineNm == "LineA"
                        #overtakeStaList = [902, 905, 907, 912, 916, 920, 924, 925, 928, 931, 934]
                    
                    else :
                        print("lineNm : "+lineNm+" , bnd : bnd"+str(bnd)+" ==> Counting the number of Overtake have not been proceeded. ")


                    # 4.5.1.1.3. 역 별로 정렬된 도착시각표를 만들기
                    for station in myd3_arr_ttable.columns :
                        arrTimeTable_stnS = myd3_arr_ttable[ station ]                      # 한개 역의 시각표 발췌
                        arrTimeTable_stnS = arrTimeTable_stnS.sort_values(ascending=True)   # 정렬
                        arrTableByStn.append(df({str(station):arrTimeTable_stnS}))          # 합치기
                    
                    # 4.5.1.1.4. 역 별로 정렬된 출발시각표를 만들기
                    for station in myd3_dep_ttable.columns :
                        depTimeTable_stnS = myd3_dep_ttable[ station ]                      # 한개 역의 시각표 발췌
                        depTimeTable_stnS = depTimeTable_stnS.sort_values(ascending=True)   # 정렬
                        depTableByStn.append(df({str(station):depTimeTable_stnS}))          # 합치기


                    # 피추월횟수 분석을 할거면 (LineNm에 따라 overtakeAnalysisTF가 자동으로 초기화 됨)    
                    if overtakeAnalysisTF == True : 
                        # 4.5.1.2. 추월가능 역 별 출도착 순서 데이터테이블 저장용 데이터프레임 만들기
                        arrSeqByStn = df() 
                        depSeqByStn = df()
                        
                        # 4.5.1.3. 마지막 역 정보 저장 : '다음역'이 역 번호 체계 범위를 벗어나지 않도록 하기 위해
                        lastStn = chkLastStnId(lineNm, bnd, myd3_arr_ttable)

                        # 4.5.1.4. 피추월횟수 결과 저장할 데이터프레임 생성
                        numOvertakeTable = df(index=myd3_arr_ttable.index)# columns=['TRNID'])
                        overtakeTrainNoTable = df(index=myd3_arr_ttable.index)# columns=['TRNID'])
                        #numOvertakeTable['TRNID'] = myd3_arr_ttable.index

                        # ================================================================
                        # 4.5.2. 피추월여부 판정을 위해 S역에서 다음 피추월역 범위에 대해 S역의 i, i+1열차에 대한 출도착 정보 취합. S역의 정보가 없을 경우, S+1역의 출발정보와 S-1역의 도착정보를 조회함
                        overtakeStation = 0 #overtakeStaList[1]
                        for overtakeStation in range(0,len(overtakeStaList)) : #range(0,(len(myd3_arr_ttable.columns)-1)):
                            
                            overtakeStnId = overtakeStaList[overtakeStation]
                            
                            print("\n=======\n overtaking StatnId : "+str(overtakeStnId))
                            
                            # 4.5.2.1. 피추월횟수 저장할 열 추가
                            numOvertakeTable[ str(overtakeStnId) ] = np.repeat([-99], len(numOvertakeTable.index),axis=0)
                            overtakeTrainNoTable[ str(overtakeStnId) ] = np.repeat([str(-99)], len(overtakeTrainNoTable.index),axis=0)
                            
                            # 4.5.2.2. 피추월 역 (S역) 출도착정보 발췌 및 정렬
                            arrTimeTable_stnS = returnSortedrtTableByStn(stationIdInteger=overtakeStnId, rtTableByStn=arrTableByStn, rtTable=myd3_arr_ttable)   # myd3_arr_ttable[ overtakeStaList[overtakeStation] ]
                            depTimeTable_stnS = returnSortedrtTableByStn(stationIdInteger=overtakeStnId, rtTableByStn=depTableByStn, rtTable=myd3_dep_ttable) #myd3_dep_ttable[ overtakeStaList[overtakeStation] ]

                            # ================================================================
                            # 4.5.3. 피추월횟수 계산
                            train = 0 # 열차 조회변수 초기화
                            arrSeq = [] # 출발순서 리스트 초기화
                            depSeq = [] # 도착순서 리스트 초기화
                            tempArrSeq = [] # while문 용 출발순서 리스트 초기화
                            tempDepSeq = [] # while문 용 도착순서 리스트 초기화
                            arrTrain_ip1_byPreviousStn = [] # 출발역 도착데이터 누락 열차번호 저장용 리스트 초기화
                            previousStnList = [] # 출발역 도착데이터 누락 열차 발생 역 ID  저장용 리스트 초기화
                            depTrain_ip1_byForwardStn = [] # 출발역 도착데이터 누락 열차번호 저장용 리스트 초기화
                            forwardStnList = [] # 출발역 도착데이터 누락 열차 발생 역 ID  저장용 리스트 초기화
                            overtakedTrnIdList = [] # 추월한 열차가 특정 열차에 대한 분석 끝난 뒤 그 다음으로 시작할 수 있도록 저장해두기
                            if 'nextTrainByForce' in locals() : # 출발역 도착데이터 누락 열차ID 저장용 리스트 초기화 : 있으면 삭제
                                del(nextTrainByForce)

                            # 4.5.3.0. while문을 통해 현재 피추월역의 도착테이블을 기준으로 도착테이블의 모든 열차를 검색.
                            #          도착테이블의 데이터 누락 시 상하류 추월가능 역 전까지 데이터를 검색하며 추월여부를 검지
                            while train < len(arrTimeTable_stnS.index):
                                
                                # 4.5.3.0.1. 변수 초기화
                                trainCnt = 1                    # 열차 추가조회 관련 변수 초기화
                                endOvertakingAnalysis = False   # 피추월횟수 분석을 종료할것인가? True면 분석이 종료, False면 분석 계속
                                errCntNextTrain = 0             # 오류 저장변수 초기화
                                numOfOvertake = 0               # 피추월횟수 저장 변수 초기화
                                natTF_ip0StnS = False           # 현재 분석열차의 정보가 누락상태인지를 나타내는 변수 초기화 

                                # 4.5.3.0.2. 앞선 조회에서 S역 기준 누락이 있었을 경우(), 누락이 있던 열차ID를 nextTrainByForce 변수에 저장하고 다른 변수는 초기화
                                if len(arrTrain_ip1_byPreviousStn) >= 2 : 
                                    # 다음 피추월 역 정보 생성 : 이전역 조회 기준으로
                                    tempNxtOvtStnId = findNextOvtStnOrder(lineNm=lineNm, bnd=bnd, nextStnTF=False, currentOvtStnOrder=overtakeStation, ovtStnIdList=overtakeStaList, wholeStnIdList=stations)   

                                    # 현재 있는 누락열차 보완 점검 리스트에서 가장 빨리 도착한 열차정보 찾기
                                    tempNxtStartingStnId = list(findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnSmN_ip1StnSmN, initStnId=arrTimeDataStnSmN, nextStnTF=False, nextOvtStnId=stations[tempNxtOvtStnId], trainNoByPreviousStnList=arrTrain_ip1_byPreviousStn, stnIdByPreviousStnList=previousStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable))[0]

                                    if (len(overtakedTrnIdList) == 0) : 
                                        natTF_ip0StnS = True                        # 현재 분석대상열차의 도착정보가 NaT임을 표시
                                        nextTrainByForce = tempNxtStartingStnId     # 분석대상열차 강제 지정
                                        train = train - 1                           # train 조회변수 한 개 역전시키기 : 누락열차를 조회하므로 while문에 의해 1 증가한 train 변수를 원복시켜둠.
                                        arrTrain_ip1_byPreviousStn = []             # 출발역 도착데이터 누락 열차번호 저장용 리스트 초기화
                                        previousStnList = []                        # 출발역 도착데이터 누락 열차 발생 역 ID  저장용 리스트 초기화
                                        #else :
                                            #print("누락열차 리스트 확인요망")
                                            #arrTrain_ip1_byPreviousStn = [] # 출발역 도착데이터 누락 열차번호 저장용 리스트 초기화
                                            #previousStnList = [] # 출발역 도착데이터 누락 열차 발생 역 ID  저장용 리스트 초기화
                                    
                                    elif (overtakedTrnIdList[0] == tempNxtStartingStnId ) :
                                        natTF_ip0StnS = True                        # 현재 분석대상열차의 도착정보가 NaT임을 표시
                                        nextTrainByForce = overtakedTrnIdList[0]    # 분석대상열차 강제 지정
                                        overtakedTrnIdList.pop(0)                   # 미 조회 급행열차목록에서 분석을 시작 할 분석대상열차를 삭제
                                        train = train - 1                           # train 조회변수 한 개 역전시키기 : 누락열차를 조회하므로 while문에 의해 1 증가한 train 변수를 원복시켜둠.
                                        arrTrain_ip1_byPreviousStn = []             # 출발역 도착데이터 누락 열차번호 저장용 리스트 초기화
                                        previousStnList = []                        # 출발역 도착데이터 누락 열차 발생 역 ID  저장용 리스트 초기화

                                    else : 
                                        print("\n아직 분석하지 않은 추월 열차가 있습니다. \n누락 열차 리스트는 다시 조회될것이므로 초기화합니다. | "+str(overtakedTrnIdList)+"\n")
                                        arrTrain_ip1_byPreviousStn = []             # 출발역 도착데이터 누락 열차번호 저장용 리스트 초기화
                                        previousStnList = []                        # 출발역 도착데이터 누락 열차 발생 역 ID  저장용 리스트 초기화
                                    
                                # 4.5.3.0.3. S역 도착시각 누락 보정 여부 True/False 변수 초기화
                                useTrain_ip1StnSmN_atStnSmN = False
                                
                                # while문을 통해 추월을 안 한 열차가 검색될 때 까지 반복해서 시행 : 이렇게 해야 동시에 2대 대피하는 상황을 검지할 수 있음
                                while ( ((train + trainCnt) < len(arrTimeTable_stnS.index)) & (trainCnt < 5) & (endOvertakingAnalysis == False) ) & (errCntNextTrain < 10) : # 현재 조회하는 열차 순번(n번째 열차)가 도착테이블 상에 존재하는 총 열차 댓수보다 작으며, 열차 추가조회변수가 5보다 작고, 피추월횟수 분석종료 Flag 변수가 False고 errCnt가 10보다 작은 경우=
                                    
                                    arrDataGoodStnS = False # 최초 False로 초기화. while문을 돌다가 전 역 데이터 관련 문제가 해결되면 True로 갱신되며 while에서 빠져나오게 됨
                                    depDataGoodStnS = False
                                    arrDepBothDataGood = False 
                                    
                                    # 4.5.3.1. S역의 i(=train)번째(분석대상 열차) 열차 도착정보 저장 
                                    # 4.5.3.1.1. 만약 앞선 열차 분석 중 출발역 데이터 관련 오류를 발견했다면, 아얘 S-1역 정보에 대해 조회하기
 
                                    # 4.5.3.1.2. 만약 앞선 열차 분석 중 출발역 출발시각 데이터 관련 오류가 있었으면, 누락이 있던 열차부터 조회 시작
                                    if 'nextTrainByForce' in locals() :
                                        arrTime_atStnS_ip0StnS = pd.NaT # 도착시각
                                        arrTrainNo_atStnS_ip0StnS = nextTrainByForce # 열차id
                                        del(nextTrainByForce)

                                    else :
                                        # 4.5.3.1.3. S역 i열차 도착시각과 열차ID 저장 : 만약 앞선 열차 분석 중 출발역 출발시각 데이터 관련 오류가 없었으면, 원래대로 S역 정보에 대해 조회하기
                                        arrTime_atStnS_ip0StnS = arrTimeTable_stnS.iloc[(train+0)][0] # 도착시각
                                        arrTrainNo_atStnS_ip0StnS = arrTimeTable_stnS.index[(train+0)] # 열차id

                                        # 4.5.3.1.4. 추월열차 목록 조회를 통해 추월열차 검색이 누락되지 않도록 처리하기
                                        if (numOfOvertake == 0)  & (len(overtakedTrnIdList) > 0) :
                                            if overtakedTrnIdList[0] == arrTrainNo_atStnS_ip0StnS :
                                                overtakedTrnIdList.pop(0) # 추월열차 목록에서 분석대상열차 삭제하기

                                                # 4.5.3.1.5. 만약 이 때 누락열차 리스트가 남아있으면?
                                                if len(arrTrain_ip1_byPreviousStn) > 0 :
                                                    print("누락열차 목록이 남아있음")
                                                    #arrTrain_ip1_byPreviousStn = [] # 출발역 도착데이터 누락 열차번호 저장용 리스트 초기화
                                                    #previousStnList = [] # 출발역 도착데이터 누락 열차 발생 역 ID  저장용 리스트 초기화
                                            else :
                                                print("=== 추월열차가 분석되지 않았음. 미조회 추월열차부터 분석을 다시 시작합니다. 열차번호:"+str(overtakedTrnIdList[0]))
                                                arrTime_atStnS_ip0StnS = pd.NaT # 도착시각
                                                arrTrainNo_atStnS_ip0StnS = overtakedTrnIdList[0] # 열차id
                                                overtakedTrnIdList.pop(0)
                                                train = train - 1

                                    # 4.5.3.2. S역의 i+1번째 열차 (후착) 도착시각과 열차ID 저장 
                                    if (train + trainCnt) < len(arrTimeTable_stnS.index) :
                                        arrTime_atStnS_ip1StnS = arrTimeTable_stnS.iloc[(train+trainCnt)][0] # 도착시각
                                        arrTrainNo_atStnS_ip1StnS = arrTimeTable_stnS.index[(train+trainCnt)] # 열차id
                                        if (train + trainCnt + 1) < len(arrTimeTable_stnS.index) :
                                            arrTime_atStnS_ip2StnS = arrTimeTable_stnS.iloc[(train+trainCnt+1)][0] # 도착시각
                                            arrTrainNo_atStnS_ip2StnS = arrTimeTable_stnS.index[(train+trainCnt+1)] # 열차id
                                        else :
                                            print("i+1 열차가 마지막 데이터라 i+2열차 정보는 i+1열차 정보로 대체하였음.")
                                            arrTime_atStnS_ip2StnS = arrTimeTable_stnS.iloc[(train+trainCnt)][0] # 도착시각
                                            arrTrainNo_atStnS_ip2StnS = arrTimeTable_stnS.index[(train+trainCnt)] # 열차id
                                    else :
                                        print("This is the last row of train data.")

                                    # 4.5.3.3. S역의 분석대상 열차정보 표출
                                    print(str(oDate)+" | "+str(lineNm)+" | "+str(overtakeStnId)+" | bnd:"+str(bnd)+" | S역 i열차: "+str(arrTrainNo_atStnS_ip0StnS)+" | i+1열차: "+str(arrTrainNo_atStnS_ip1StnS))

                                    # 4.5.3.4. S역의 i, i+1 열차의 도착정보가 있으면 S-1역의 i열차 ID 찾기 : 실시간 데이터가 누락되었는지 여부를 판단하기 위해, 전 역에 대해서도 i열차와 i+1열차를 확인했을 때 그 사이에 열차가 없는지 점검하는 절차. 만악 직전역의 도착정보도 누락일경우 오류를 검지하지 못하고 뒷 절차에서 검지하도록 코딩되어있음.

                                    if (arrTime_atStnS_ip0StnS is not pd.NaT) & (arrTime_atStnS_ip1StnS is not pd.NaT) :
                                        
                                        # 4.5.3.5. 역 별 i+1열차 ID를 저장 : S역부터 하나씩 거슬러올라가면서 저장
                                        stationCnt = 0 # 기준역에서 전/후역으로 이동하면서 검색하게 해주는 cnt 변수 초기화. while문 안에서 1씩 증가하며 검색할것임.

                                        # 4.5.3.6. 출발역의 전역에 대해 분석대상열차와 누락된 열차정보를 조회
                                        # while문을 활용해 역을 거슬러가면서 S역의 도착열차의 일관성 검토
                                        stnS_dataChkErrCnt = 0
                                        while (arrDataGoodStnS == False) & (stnS_dataChkErrCnt < max_stnS_dataChkErrCnt) :
                                            stationCnt = stationCnt - 1     # 역 추가검토 관련 변수 초기화 : 전 역을 조회하기 위해 -1로 지정
                                            searchingToNextStn = False      # 다음역 방향으로 조회하는지 결정하는 변수 초기화 : False를 줘서 전 역 방향으로 조회되도록 설정
                                            nextOvertakeStnId = stations[ findNextOvtStnOrder(lineNm=lineNm, bnd=bnd, nextStnTF=searchingToNextStn, currentOvtStnOrder= overtakeStation, ovtStnIdList=overtakeStaList, wholeStnIdList=stations) ]
                                            
                                            departureTableFlag = False      # 도착 데이터부터 조회할 수 있도록 parameter를 변경
                                            
                                            # 만약 S역 도착정보의 오류를 보정한 상태라면 후속열차 조회하는 변수를 현재값에서 1 증가시킴
                                            if useTrain_ip1StnSmN_atStnSmN == True :
                                                trainCnt = trainCnt + 1                 # 이 위치에서 보정상태에서만 증가시키는 이유는 trainCnt는 현재역과 전 역 모두에 영향을 주기 때문.
                                                                                        #  이 if문에 True인 상태라는 것은 이미 누락열차에 대한 1차 분석을 마치고 while문에 의해 S역의 i열차와 S역의 i+1열차 (누락열차를 포함하면 실제로는 i+2열차)를 검지하는 부분이기 때문에, S-1역에서는 i+2열차가 조회되어야 하기 때문임.
                                                useTrain_ip1StnSmN_atStnSmN = False     # S역 도착정보 오류 보정 Flag 변수를 False로 초기화

                                            # 4.5.3.7. 전 역의 i열차와 i+1열차(정보 누락되서 빠졌던 열차)의 출도착시각 얻기
                                            arrDepDataAvailableRes = findingdepArrTimeAtStaSpN(stationCnt, searchingToNextStn, departureTableFlag, stations, overtakeStaList, overtakeStation, lineNm, bnd, myd3_arr_ttable, myd3_dep_ttable, arrTrainNo_atStnS_ip0StnS, arrTrainNo_atStnS_ip1StnS, arrTimeTable_stnS, train, trainCnt)
                                            
                                            # 4.5.3.8. findingdepArrTimeAtStaSpN 함수 결과 저장
                                            if arrDepDataAvailableRes[0] != False :
                                                arrDepDataAvailableFlag = arrDepDataAvailableRes[0]                 # 출도착정보 가용여부 Flag
                                                arrTrainOrderOf_ip0StnS_atStnSmN = arrDepDataAvailableRes[1]        # S역 i열차의 S-1역 도착(출발) 순서
                                                arrTime_atStnSmN_ip0StnS = arrDepDataAvailableRes[2]                # S역 i열차의 S-1역 도착(출발) 시각
                                                arrTrainOrderOf_ip1StnSmN_atStnSmN = arrDepDataAvailableRes[3]      # S-1역 i+1열차(누락됐던 열차)의 S-1역 도착(출발) 순서 (S-1역 기준으로 i열차의 후속열차를 재검색)
                                                arrTime_atStnSmN_ip1StnSmN = arrDepDataAvailableRes[4]              # S-1역 i+1열차(누락됐던 열차)의 S-1역 도착(출발) 시간 (S-1역 기준으로 i열차의 후속열차를 재검색)
                                                arrTimeTable_stnSmN = arrDepDataAvailableRes[5]                     # S-1역의 열차 관련 시각표
                                                arrTimeDataStnSmN = arrDepDataAvailableRes[6]                       # S-1역 역 ID
                                            
                                                # 4.5.3.9. 현재 역과 전 역의 후속열차 ID 비교
                                                # 4.5.3.9.1. # S-1역 i+1열차(누락됐던 열차)의 열차ID 저장
                                                arrTrainNo_atStnSmN_ip1StnSmN = arrTimeTable_stnSmN.index[arrTrainOrderOf_ip1StnSmN_atStnSmN]

                                                # 4.5.3.9.2. # S역과 S-1역의 i+1열차 ID 비교
                                                if (arrTrainNo_atStnS_ip1StnS == arrTrainNo_atStnSmN_ip1StnSmN) & ( (len(arrTrain_ip1_byPreviousStn) == 0) | (len(arrTrain_ip1_byPreviousStn) >= 2) )  :
                                                    if (arrTime_atStnS_ip1StnS is not pd.NaT) & (arrTime_atStnSmN_ip1StnSmN is not pd.NaT) :
                                                        # 4.5.3.9.4. 출발역 출발데이터 점검 Flag 초기화
                                                        arrDataGoodStnS = True      

                                                    else :
                                                        if len(arrTrain_ip1_byPreviousStn) > 0 :
                                                            # 누락열차 조회중이므로 출발역 출발데이터 점검 Flag 초기화
                                                            arrDataGoodStnS = True    

                                                        else :
                                                            print("S역과 S-1역의 열차정보는 같으나 S-1역의 i, i+1열차 중 하나의 시각정보가 없음")
                                                            arrDataGoodStnS = False
                                                
                                                else : # if (arrTrainNo_atStnS_ip1StnS == arrTrainNo_atStnSmN_ip1StnSmN) & (arrTime_atStnS_ip1StnS is not pd.NaT) & (arrTime_atStnSmN_ip1StnSmN is not pd.NaT) :
                                                    # 4.5.3.9.5. S역 i+1열차와 다르다고 나온 역의 i+1열차의 도착순서 비교
                                                    
                                                    # S-1역의 도착시각표 불러오기
                                                    tempArrtimeTable_stnSpN = returnSortedrtTableByStn(stationIdInteger=arrTimeDataStnSmN, rtTableByStn=arrTableByStn, rtTable=myd3_arr_ttable) 
                                                    arrOrderOfTrain0 = np.where(tempArrtimeTable_stnSpN.index==arrTrainNo_atStnS_ip1StnS)[0][0]         # S역 i+1열차의 S-1역 도착순서 저장
                                                    arrOrderOfTrain1 = np.where(tempArrtimeTable_stnSpN.index==arrTrainNo_atStnSmN_ip1StnSmN)[0][0]     # S-1역 i+1열차의 S-1역 도착순서 저장

                                                    # S-1역의 출발시각표 불러오기
                                                    tempDeptimeTable_stnSpN = returnSortedrtTableByStn(stationIdInteger=arrTimeDataStnSmN, rtTableByStn=depTableByStn, rtTable=myd3_dep_ttable) 
                                                    depOrderOfTrain0 = np.where(tempDeptimeTable_stnSpN.index==arrTrainNo_atStnS_ip1StnS)[0][0]         # S역 i+1열차의 S-1역 출발순서 저장
                                                    depOrderOfTrain1 = np.where(tempDeptimeTable_stnSpN.index==arrTrainNo_atStnSmN_ip1StnSmN)[0][0]     # S-1역 i+1열차의 S-1역 출발순서 저장
                                                    
                                                    # 어떤열차가 정말 앞선 열차인지 확인
                                                    if (tempArrtimeTable_stnSpN.iloc[arrOrderOfTrain0][0] is not pd.NaT) & (tempArrtimeTable_stnSpN.iloc[arrOrderOfTrain1][0] is not pd.NaT) :    # 두 데이터의 시각정보가 모두 유효하다면 (만약 한쪽 시각이 NaT인 경우 비교 불가)
                                                        if arrOrderOfTrain0 < arrOrderOfTrain1 :    # S역 i+1열차의 정보가 참
                                                            print("원래의 i+1열차가 맞는 것으로 확인되었습니다: "+str(arrTrainNo_atStnS_ip1StnS)) 
                                                            arrTrainNo_atStnS_ip1StnS = arrTrainNo_atStnS_ip1StnS
                                                            arrDataGoodStnS = True
                                                            #useTrain_ip1StnSmN_atStnSmN = True
                                                            #arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                            #previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN)
                                                        
                                                        else :                                      # S-1역 i+1열차의 정보가 참 - 같을 순 없음. 같았으면 위에서 같다고 하고 끝났을것임
                                                            print("                                    i+1열차 ID가 수정되었습니다: "+str(arrTrainNo_atStnS_ip1StnS)+" --> "+str(arrTrainNo_atStnSmN_ip1StnSmN))
                                                            arrTrainNo_atStnS_ip1StnS = arrTrainNo_atStnSmN_ip1StnSmN
                                                            arrDataGoodStnS = True
                                                            useTrain_ip1StnSmN_atStnSmN = True
                                                            arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                            previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN) 
                                                    
                                                    elif (tempDeptimeTable_stnSpN.iloc[depOrderOfTrain0][0] is not pd.NaT) & (tempDeptimeTable_stnSpN.iloc[depOrderOfTrain1][0] is not pd.NaT) :
                                                        if depOrderOfTrain0 < depOrderOfTrain1 :    # S역 i+1열차의 정보가 참
                                                            print("원래의 i+1열차가 맞는 것으로 확인되었습니다: "+str(arrTrainNo_atStnS_ip1StnS)) 
                                                            arrTrainNo_atStnS_ip1StnS = arrTrainNo_atStnS_ip1StnS
                                                            arrDataGoodStnS = True
                                                            #useTrain_ip1StnSmN_atStnSmN = True
                                                            #arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                            #previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN)
                                                        
                                                        else :                                      # S-1역 i+1열차의 정보가 참 - 같을 순 없음. 같았으면 위에서 같다고 하고 끝났을것임
                                                            print("                                    i+1열차 ID가 수정되었습니다: "+str(arrTrainNo_atStnS_ip1StnS)+" --> "+str(arrTrainNo_atStnSmN_ip1StnSmN)) 
                                                            arrTrainNo_atStnS_ip1StnS = arrTrainNo_atStnSmN_ip1StnSmN
                                                            arrDataGoodStnS = True
                                                            useTrain_ip1StnSmN_atStnSmN = True
                                                            arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                            previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN)
                                                    
                                                    else :
                                                        print("S-1역:"+str(arrTimeDataStnSmN)+"의 출도착 시각표 모두에서 S역 i+1열차와 S-1역 i+1열차의 출도착시각이 없습니다.")
                                                    
                                                        # 4.5.3.9.6. 이전 역 별 i+1열차 id 저장
                                                        arrTrain_ip1_byPreviousStn.append( arrTrainNo_atStnSmN_ip1StnSmN )
                                                        previousStnList.append( arrTimeDataStnSmN )

                                                        # 4.5.3.9.7. 만약 마지막 두 역의 i+1열차 정보가 맞다면, S역의 정보가 틀렸다고 보는게 합리적일것임
                                                        if ((stnS_dataChkErrCnt >= 2) & (len(arrTrain_ip1_byPreviousStn)%2 == 1)) : #& (arrTrain_ip1_byPreviousStn[(len(arrTrain_ip1_byPreviousStn)-1)] == arrTrain_ip1_byPreviousStn[(len(arrTrain_ip1_byPreviousStn)-2)]) :
                                                            
                                                            #arrTrainNo_atStnS_ip1StnS, arrDataGoodStnS, useTrain_ip1StnSmN_atStnSmN = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=arrTrain_ip1_byPreviousStn)
                                                            arrTrainNo_atStnS_ip1StnS, feasibleStnId, arrDataGoodStnS, useTrain_ip1StnSmN_atStnSmN = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=arrTimeDataStnSmN, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=arrTrain_ip1_byPreviousStn, stnIdByPreviousStnList=previousStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)


                                                        else : 
                                                            # 4.5.3.9.6. 한개 더 전 역에 대해서 조회할 수 있도록 False 유지
                                                            #stationCnt = stationCnt - 1
                                                            arrDataGoodStnS = False

                                            else : # if arrDepDataAvailableRes[0] != False :
                                                arrDepDataAvailableFlag = arrDepDataAvailableRes[0]
                                                #stnS_dataChkErrCnt = arrDepDataAvailableRes[1]
                                                if len(arrTrain_ip1_byPreviousStn) >= 3 : 
                                                    #arrTrainNo_atStnS_ip1StnS, arrDataGoodStnS, useTrain_ip1StnSmN_atStnSmN = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=arrTrain_ip1_byPreviousStn)
                                                    arrTrainNo_atStnS_ip1StnS, feasibleStnId, arrDataGoodStnS, useTrain_ip1StnSmN_atStnSmN = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=arrTimeDataStnSmN, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=arrTrain_ip1_byPreviousStn, stnIdByPreviousStnList=previousStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                                                    
                                                elif (len(arrTrain_ip1_byPreviousStn) == 2) & (stnS_dataChkErrCnt >= 4) :
                                                    if (arrTrain_ip1_byPreviousStn[0] == arrTrain_ip1_byPreviousStn[1]) :
                                                        #arrTrainNo_atStnS_ip1StnS, arrDataGoodStnS, useTrain_ip1StnSmN_atStnSmN = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=arrTrain_ip1_byPreviousStn)
                                                        arrTrainNo_atStnS_ip1StnS, feasibleStnId, arrDataGoodStnS, useTrain_ip1StnSmN_atStnSmN = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=arrTimeDataStnSmN, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=arrTrain_ip1_byPreviousStn, stnIdByPreviousStnList=previousStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                                                    else : # HERE
                                                        if previousStnList[0] == previousStnList[1] :
                                                            
                                                            tempArrtimeTable_stnSpN = returnSortedrtTableByStn(stationIdInteger=previousStnList[0], rtTableByStn=arrTableByStn, rtTable=myd3_arr_ttable) 
                                                            arrOrderOfTrain0 = np.where(tempArrtimeTable_stnSpN.index==arrTrain_ip1_byPreviousStn[0])[0][0]         # S역 i+1열차의 S-1역 도착순서 저장
                                                            arrOrderOfTrain1 = np.where(tempArrtimeTable_stnSpN.index==arrTrain_ip1_byPreviousStn[1])[0][0]     # S-1역 i+1열차의 S-1역 도착순서 저장

                                                            # S-1역의 출발시각표 불러오기
                                                            tempDeptimeTable_stnSpN = returnSortedrtTableByStn(stationIdInteger=previousStnList[0], rtTableByStn=depTableByStn, rtTable=myd3_dep_ttable) 
                                                            depOrderOfTrain0 = np.where(tempDeptimeTable_stnSpN.index==arrTrain_ip1_byPreviousStn[0])[0][0]         # S역 i+1열차의 S-1역 출발순서 저장
                                                            depOrderOfTrain1 = np.where(tempDeptimeTable_stnSpN.index==arrTrain_ip1_byPreviousStn[1])[0][0]     # S-1역 i+1열차의 S-1역 출발순서 저장
                                                            
                                                            # 어떤열차가 정말 앞선 열차인지 확인
                                                            if (tempArrtimeTable_stnSpN.iloc[arrOrderOfTrain0][0] is not pd.NaT) & (tempArrtimeTable_stnSpN.iloc[arrOrderOfTrain1][0] is not pd.NaT) :    # 두 데이터의 시각정보가 모두 유효하다면 (만약 한쪽 시각이 NaT인 경우 비교 불가)
                                                                if arrOrderOfTrain0 < arrOrderOfTrain1 :    # S역 i+1열차의 정보가 참
                                                                    arrTrainNo_atStnS_ip1StnS = arrTrain_ip1_byPreviousStn[0]
                                                                    arrDataGoodStnS = True
                                                                    useTrain_ip1StnSmN_atStnSmN = True
                                                                    arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                    previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN)
                                                                    print("i+1열차 ID가 확인되었습니다: "+str(arrTrainNo_atStnS_ip1StnS)) 
                                                                
                                                                else :                                      # S-1역 i+1열차의 정보가 참 - 같을 순 없음. 같았으면 위에서 같다고 하고 끝났을것임
                                                                    arrTrainNo_atStnS_ip1StnS = arrTrain_ip1_byPreviousStn[1]
                                                                    arrDataGoodStnS = True
                                                                    useTrain_ip1StnSmN_atStnSmN = True
                                                                    arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                    previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN) 
                                                                    print("i+1열차 ID가 확인되었습니다: "+str(arrTrainNo_atStnS_ip1StnS)) 
                                                            
                                                            elif (tempDeptimeTable_stnSpN.iloc[depOrderOfTrain0][0] is not pd.NaT) & (tempDeptimeTable_stnSpN.iloc[depOrderOfTrain1][0] is not pd.NaT) :
                                                                if depOrderOfTrain0 < depOrderOfTrain1 :    # S역 i+1열차의 정보가 참
                                                                    arrTrainNo_atStnS_ip1StnS = arrTrain_ip1_byPreviousStn[0]
                                                                    arrDataGoodStnS = True
                                                                    useTrain_ip1StnSmN_atStnSmN = True
                                                                    arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                    previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN)
                                                                    print("i+1열차 ID가 확인되었습니다: "+str(arrTrainNo_atStnS_ip1StnS))  
                                                                
                                                                else :                                      # S-1역 i+1열차의 정보가 참 - 같을 순 없음. 같았으면 위에서 같다고 하고 끝났을것임
                                                                    arrTrainNo_atStnS_ip1StnS = arrTrain_ip1_byPreviousStn[1]
                                                                    arrDataGoodStnS = True
                                                                    useTrain_ip1StnSmN_atStnSmN = True
                                                                    arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                    previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN)
                                                                    print("i+1열차 ID가 확인되었습니다: "+str(arrTrainNo_atStnS_ip1StnS)) 

                                                        else : 
                                                            print("S역 누락열차 확인 리스트의 열차 ID 두개가 정보가 서로 다릅니다")
                                                else :
                                                    print("S역 누락열차 확인 필요")
                                                

                                            # 무한루프 방지 변수 초기화
                                            stnS_dataChkErrCnt = stnS_dataChkErrCnt + 1
                                        
                                        if arrDataGoodStnS == True : # 도착데이터 상에서 i열차와 i+1열차가 확인되었으면
                                            # 4.5.3.10. s역부터 다음 피추월역까지의 출도착 정보를 조회하여 출도착 정보가 둘 다 있는 경우를 찾아서 확인
                                            stationCnt = -1 # 역 추가검토 관련 변수 초기화. S역 출발데이터부터 검색해야 하므로 -1 로 초기화
                                            depDataGoodStnS = False 
                                            stnS_dataChkErrCnt = 0
                                            
                                            # 4.5.3.11. 출발역의 다음역에 대해 i열차(분석대상열차)와 i+1열차를 조회 : i+1열차의 ID를 가지고 출발순서를 찾는게 핵심
                                            #           while문을 활용해 하류 역으로 이동해가면서 S역의 출발열차의 일관성 검토
                                            while (depDataGoodStnS == False) & (stnS_dataChkErrCnt < max_stnS_dataChkErrCnt) :
                                                stationCnt = stationCnt + 1
                                                searchingToNextStn = True # 다음역 방향으로 조회하는지 결정하는 변수 초기화
                                                departureTableFlag = True # 출발역 데이터부터 조회할 수 있도록 True로 초기화
                                                nextOvertakeStnId = stations[ findNextOvtStnOrder(lineNm=lineNm, bnd=bnd, nextStnTF=searchingToNextStn, currentOvtStnOrder=overtakeStation, ovtStnIdList=overtakeStaList, wholeStnIdList=stations) ]

                                                # 4.5.3.11.1. 다음 역의 i열차와 i+1열차(정보 누락되서 빠졌던 열차)의 출도착시각 얻기
                                                arrDepDataAvailableRes = findingdepArrTimeAtStaSpN(stationCnt, searchingToNextStn, departureTableFlag, stations, overtakeStaList, overtakeStation, lineNm, bnd, myd3_arr_ttable, myd3_dep_ttable, arrTrainNo_atStnS_ip0StnS, arrTrainNo_atStnS_ip1StnS, arrTimeTable_stnS, train, trainCnt)

                                                # 4.5.3.11.2. findingdepArrTimeAtStaSpN 함수 결과 저장
                                                if arrDepDataAvailableRes[0] != False :
                                                    arrDepDataAvailableFlag = arrDepDataAvailableRes[0]
                                                    depTrainOrderOf_ip0StnS_atStnSpN = arrDepDataAvailableRes[1]
                                                    depTime_atStnSpN_ip0StnS = arrDepDataAvailableRes[2]
                                                    depTrainOrderOf_ip1StnS_atStnSpN = arrDepDataAvailableRes[3]
                                                    depTime_atStnSpN_ip1StnSpN = arrDepDataAvailableRes[4]
                                                    depTimeTable_stnSpN = arrDepDataAvailableRes[5]
                                                    depTimeDataStnSpn = arrDepDataAvailableRes[6]

                                                    # 4.5.3.11.3. 현재 역과 전 역의 후속열차 ID 비교
                                                    # 4.5.3.11.3.1. # S-1역 i+1열차(누락됐던 열차)의 열차ID 저장
                                                    depTrainNo_atStnSpN_ip1StnSpN = depTimeTable_stnSpN.index[depTrainOrderOf_ip1StnS_atStnSpN]

                                                    # 4.5.3.9.2. # S역의 i+1열차와 S역(또는 S+1역)의 i+1열차 ID 비교 : 애초에 출발열차 ID는 확인된 출발열차 ID를 통해 검색하므로 당연히 같아야 함
                                                    if (arrTrainNo_atStnS_ip1StnS == depTrainNo_atStnSpN_ip1StnSpN) & ( (len(depTrain_ip1_byForwardStn) == 0) | (len(depTrain_ip1_byForwardStn) >= 2) )  :
                                                        if (depTime_atStnSpN_ip0StnS is not pd.NaT) & (depTime_atStnSpN_ip1StnSpN is not pd.NaT) :
                                                            # 4.5.3.9.3. S역과 S-1역의 i+1열차 ID가 같다면
                                                            #print("OK")

                                                            # 4.5.3.9.4. 출발역 출발데이터 점검 Flag 초기화
                                                            depDataGoodStnS = True  

                                                        else :
                                                            print("S역과 S-1역의 열차정보는 같으나 S-1역의 i, i+1열차 중 하나의 시각정보가 없음")
                                                            depDataGoodStnS = False
                                                    
                                                    else : # if (arrTrainNo_atStnS_ip1StnS == depTrainNo_atStnSpN_ip1StnSpN) & (depTime_atStnSpN_ip0StnS is not pd.NaT) & (depTime_atStnSpN_ip1StnSpN is not pd.NaT) :
                                                        # 4.5.3.9.5. S역과 S-1역의 i+1열차 ID가 다르다면
                                                        #print("ERR. S역과 S-1역의 i+1열차 ID가 다릅니다.")

                                                        # 4.5.3.9.6. 이전 역 별 i+1열차 id 저장
                                                        depTrain_ip1_byForwardStn.append( depTrainNo_atStnSpN_ip1StnSpN )
                                                        forwardStnList.append( depTimeDataStnSpn )

                                                        # 4.5.3.9.7. 만약 마지막 두 역의 i+1열차 정보가 맞다면, S역의 정보가 틀렸다고 보는게 합리적일것임
                                                        if (stnS_dataChkErrCnt >= 2)  & (len(depTrain_ip1_byForwardStn)%2 == 1) : #& (depTrain_ip1_byForwardStn[(len(depTrain_ip1_byForwardStn)-1)] == depTrain_ip1_byForwardStn[(len(depTrain_ip1_byForwardStn)-2)]) :
                                                            
                                                            #arrTrainNo_atStnS_ip1StnS, depDataGoodStnS, useTrainTF_notUsed = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=depTrain_ip1_byForwardStn)
                                                            arrTrainNo_atStnS_ip1StnS, feasibleStnId, depDataGoodStnS, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=depTimeDataStnSpn, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=depTrain_ip1_byForwardStn, stnIdByPreviousStnList=forwardStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)

                                                        else : 
                                                            # 4.5.3.9.6. 한개 더 전 역에 대해서 조회할 수 있도록 False 유지
                                                            #stationCnt = stationCnt - 1
                                                            depDataGoodStnS = False

                                                
                                                else : # if arrDepDataAvailableRes[0] != False :
                                                    arrDepDataAvailableFlag = arrDepDataAvailableRes[0]
                                                    #stnS_dataChkErrCnt = arrDepDataAvailableRes[1]
                                                    
                                                    if len(depTrain_ip1_byForwardStn) >= 3 : 
                                                        #arrTrainNo_atStnS_ip1StnS, depDataGoodStnS, useTrainTF_notUsed = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=depTrain_ip1_byForwardStn)
                                                        arrTrainNo_atStnS_ip1StnS, feasibleStnId, depDataGoodStnS, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=depTimeDataStnSpn, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=depTrain_ip1_byForwardStn, stnIdByPreviousStnList=forwardStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                                                    
                                                    elif (len(depTrain_ip1_byForwardStn) == 2) & (stnS_dataChkErrCnt >= 4) :
                                                        if (depTrain_ip1_byForwardStn[0] == depTrain_ip1_byForwardStn[1]) :
                                                            #arrTrainNo_atStnS_ip1StnS, depDataGoodStnS, useTrainTF_notUsed = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=depTrain_ip1_byForwardStn)
                                                            arrTrainNo_atStnS_ip1StnS, feasibleStnId, depDataGoodStnS, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=depTimeDataStnSpn, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=depTrain_ip1_byForwardStn, stnIdByPreviousStnList=forwardStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                                                        else :
                                                            print("S역 누락 출발열차 확인 리스트의 열차 ID 두개가 정보가 서로 다릅니다 2")
                                                
                                                    else :
                                                        print("S역 누락 출발열차 확인 필요 2") 
                                                
                                                # 무한루프 방지 변수 초기화
                                                stnS_dataChkErrCnt = stnS_dataChkErrCnt + 1
                                            
                                            if arrDataGoodStnS == True & depDataGoodStnS == True :  # 출발, 도착데이터가 모두 양호하면
                                                arrDepBothDataGood = True                           # 출도착데이터 확인변수 True로 초기화
                                            else :
                                                arrDepBothDataGood = False
                                            
                                        else :
                                            print("출발역 도착열차 순서의 일관성이 확보되지 않았음")
                                            arrDepBothDataGood = False

                                    # 4.5.3.13. S역의 i, i+1 열차의 도착정보가 둘 중 하나라도 없으면
                                    else : # if (arrTime_atStnS_ip0StnS is not pd.NaT) & (arrTime_atStnS_ip1StnS is not pd.NaT) :
                                        
                                        # 4.5.3.14. S역의 i열차 도착시각은 있으나 i+1열차 도착시각이 없는 경우
                                        if  (arrTime_atStnS_ip0StnS is not pd.NaT) & (arrTime_atStnS_ip1StnS is pd.NaT):
                                            print("\nS역 도착정보 중 i+1 열차-"+str(arrTrainNo_atStnS_ip1StnS)+" 의 정보가 없음\n")
                                            # 4.5.3.14.0. 출도착정보 가용여부 Flag False로 초기화
                                            arrDepBothDataGood = False 

                                            # 4.5.3.14.1. S역 i열차가 막차인지 확인
                                            if arrTrainNo_atStnS_ip0StnS in lstcarTrainNo2List :
                                                # 4.5.3.14.2. 막차가 맞다면 피추월횟수를 0으로 강제 지정하기
                                                updatingNumOfOvertakeRes = updatingNumOfOvertake(statnId=overtakeStaList[overtakeStation], nOvertakeTable=numOvertakeTable, targetTrainNo=arrTrainNo_atStnS_ip0StnS, numberOfOvertakeFreq=0, overtakeTrnNoTable=overtakeTrainNoTable, overtakingTrainNo=arrTrainNo_atStnS_ip1StnS)
                                                
                                                endOvertakingAnalysis = updatingNumOfOvertakeRes[0] # 피추월횟수가 0이면 True, >=1 이면 False : 1회 이상 추월되는 경우를 확인하기 위해 추월이 안일어나는것을 확인할 때 까지 반복시행함
                                                
                                                numOvertakeTable = updatingNumOfOvertakeRes[1]
                                                overtakeTrainNoTable = updatingNumOfOvertakeRes[2]

                                                # 4.5.3.14.3. 도착 순서 저장
                                                tempArrSeq = [arrTrainNo_atStnS_ip0StnS]

                                                # 4.5.3.14.4. 출발 순서 저장
                                                tempDepSeq = [arrTrainNo_atStnS_ip0StnS]

                                                endOvertakingAnalysis = True

                                            # 4.5.3.14.3.
                                            elif arrTrainNo_atStnS_ip0StnS > arrTrainNo_atStnS_ip1StnS :
                                                print("here 3-2-1") 

                                            # 4.5.3.14.4.
                                            elif arrTrainNo_atStnS_ip0StnS == arrTrainNo_atStnS_ip1StnS :
                                                print("\nLast recored of train had been analyzed. ====================\n")
                                            
                                            # 4.5.3.14.5.
                                            else :
                                                print("here 3-2-2")
                                            
                                        # 4.5.3.15. S역의 i열차, i+1열차 모두 도착정보가 없는 경우  
                                        elif (arrTime_atStnS_ip0StnS is pd.NaT) & (arrTime_atStnS_ip1StnS is pd.NaT):
                                            # 4.5.3.15.0. 출도착정보 가용여부 Flag False로 초기화
                                            arrDepBothDataGood = False 

                                            # 4.5.3.15.1. 한번도 조회되지 않은 열차일 경우 NaN으로 초기화
                                            if numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS )[0][0] ] == -99 :    
                                                #print("There's no arr or dep time information of train i and i+1. train_i: "+str(arrTrainNo_atStnS_ip0StnS)+",  train_i+1: "+str(arrTrainNo_atStnS_ip1StnS) )
                                                numOvertakeTable[str(overtakeStaList[overtakeStation])][arrTrainNo_atStnS_ip0StnS] = np.nan
                                                endOvertakingAnalysis = True
                                                #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                            
                                            # 4.5.3.15.2. 이미 NaN인 열차인 경우 분석 종료
                                            elif np.isnan(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS )[0][0] ]) :    
                                                endOvertakingAnalysis = True
                                            
                                            elif numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS )[0][0] ] >= 0 :    
                                                print("이미 초기화되었던 이력이 있음 1")  
                                                endOvertakingAnalysis = True 

                                            # 4.5.3.15.3. 기타상황
                                            else :
                                                print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                                print("here 12")      
                                                endOvertakingAnalysis = True                                      
                                            

                                        # 4.5.3.16. S역의 i열차는 정보가 없는데, i+1열차의 정보가 있는 경우
                                        elif (arrTime_atStnS_ip0StnS is pd.NaT) & (arrTime_atStnS_ip1StnS is not pd.NaT) :
                                            # 이 경우는 누락된 선행열차 정보를 S 역 i열차로 삼아 분석하는 부분임.
                                            # 4.5.3.16.1. 역 별 i+1열차 ID를 저장 : S역부터 하나씩 거슬러올라가면서 저장
                                            #arrTrain_ip1_byPreviousStn = [arrTrainNo_atStnS_ip1StnS]
                                            #previousStnList = [overtakeStaList[overtakeStation]]
                                            #stnS_dataChk = False # 최초 False로 초기화. while문을 돌다가 전 역 데이터 관련 문제가 해결되면 True로 갱신되며 while에서 빠져나오게 됨
                                            stationCnt = 0

                                            # 4.5.3.16.2. while문을 활용해 역을 거슬러가면서 검지
                                            stnS_dataChkErrCnt = 0
                                            while (arrDataGoodStnS == False) & (stnS_dataChkErrCnt < max_stnS_dataChkErrCnt) :
                                                # 4.5.3.16.3. 출발역의 전역에 대해 분석대상열차와 누락된 열차정보를 조회
                                                stationCnt = stationCnt - 1 # 역 추가검토 관련 변수 초기화 : 전 역을 조회하기 위해 -1로 지정
                                                searchingToNextStn = False # 다음역 방향으로 조회하는지 결정하는 변수 초기화 : False를 줘서 전 역 방향으로 조회되도록 설정
                                                departureTableFlag = False # 도착 데이터부터 조회할 수 있도록 parameter를 변경
                                                
                                                nextOvertakeStnId = stations[ findNextOvtStnOrder(lineNm=lineNm, bnd=bnd, nextStnTF=searchingToNextStn, currentOvtStnOrder=overtakeStation, ovtStnIdList=overtakeStaList, wholeStnIdList=stations) ]

                                                # 만약 S역 도착정보의 오류를 보정한 상태라면
                                                if useTrain_ip1StnSmN_atStnSmN == True :
                                                    trainCnt = trainCnt + 1 # 후속열차 조회하는 변수를 현재값에서 1 증가시킴
                                                                            # S역 도착정보 오류 보정 Flag 변수가 True인 상태라는건 현재 누락된 열차에 대해 조회하고 있다는 의미이며, 해당 조회를 마무리하기 위해 trainCnt를 1 증가시켜 후속열차 조회 시 이미 누락여부 파악 할 때 조회했던 열차의 그 다음열차를 조회하도록 하는 역할을 함
                                                    useTrain_ip1StnSmN_atStnSmN = False # S역 도착정보 오류 보정 Flag 변수를 False로 초기화
                                                
                                                # 4.5.3.16.4. 전 역의 i열차와 i+1열차(정보 누락되서 빠졌던 열차)의 출도착시각 얻기
                                                arrDepDataAvailableRes = findingdepArrTimeAtStaSpN(stationCnt, searchingToNextStn, departureTableFlag, stations, overtakeStaList, overtakeStation, lineNm, bnd, myd3_arr_ttable, myd3_dep_ttable, arrTrainNo_atStnS_ip0StnS, arrTrainNo_atStnS_ip1StnS, arrTimeTable_stnS, train, trainCnt)
                                                
                                                # 4.5.3.16.5. findingdepArrTimeAtStaSpN 함수 결과 저장
                                                if arrDepDataAvailableRes[0] != False :
                                                    arrDepDataAvailableFlag = arrDepDataAvailableRes[0] # 출도착정보 가용여부 Flag
                                                    arrTrainOrderOf_ip0StnS_atStnSmN = arrDepDataAvailableRes[1] # S역 i열차의 S-1역 도착(출발) 순서
                                                    arrTime_atStnSmN_ip0StnS = arrDepDataAvailableRes[2] # S역 i열차의 S-1역 도착(출발) 시각
                                                    arrTrainOrderOf_ip1StnSmN_atStnSmN = arrDepDataAvailableRes[3] # S-1역 i+1열차(누락됐던 열차)의 S-1역 도착(출발) 순서 (S-1역 기준으로 i열차의 후속열차를 재검색)
                                                    arrTime_atStnSmN_ip1StnSmN = arrDepDataAvailableRes[4] # S-1역 i+1열차(누락됐던 열차)의 S-1역 도착(출발) 시간 (S-1역 기준으로 i열차의 후속열차를 재검색)
                                                    arrTimeTable_stnSmN = arrDepDataAvailableRes[5] # S-1역의 열차 관련 시각표
                                                    arrTimeDataStnSmN = arrDepDataAvailableRes[6] # S-1역 역 ID
                                                
                                                    # 4.5.3.16.6. 현재 역과 전 역의 후속열차 ID 비교
                                                    #             S-1역 i+1열차(누락됐던 열차)의 열차ID 저장
                                                    arrTrainNo_atStnSmN_ip1StnSmN = arrTimeTable_stnSmN.index[arrTrainOrderOf_ip1StnSmN_atStnSmN]

                                                    # 4.5.3.16.7. # S역과 S-1역의 i+1열차 ID 비교
                                                    if (arrTrainNo_atStnS_ip1StnS == arrTrainNo_atStnSmN_ip1StnSmN) & ( (len(arrTrain_ip1_byPreviousStn) == 0) | (len(arrTrain_ip1_byPreviousStn) >= 2) ) :
                                                        if (arrTime_atStnS_ip1StnS is not pd.NaT) & (arrTime_atStnSmN_ip1StnSmN is not pd.NaT) :

                                                            # 4.5.3.16.9. 출발역 데이터 점검 Flag 초기화
                                                            arrDataGoodStnS = True

                                                        else :
                                                            print("S역과 S-1역의 열차정보는 같으나 S-1역의 i, i+1열차 중 하나의 시각정보가 없음")
                                                    
                                                    else : # if (arrTrainNo_atStnS_ip1StnS == arrTrainNo_atStnSmN_ip1StnSmN) & (arrTime_atStnS_ip1StnS is not pd.NaT) & (arrTime_atStnSmN_ip1StnSmN is not pd.NaT) :
                                                        # 4.5.3.16.10. S역과 S-1역의 i+1열차 ID가 다르다면
                                                        # S-1역의 도착시각표 불러오기
                                                        tempArrtimeTable_stnSpN = returnSortedrtTableByStn(stationIdInteger=arrTimeDataStnSmN, rtTableByStn=arrTableByStn, rtTable=myd3_arr_ttable) 
                                                        arrOrderOfTrain0 = np.where(tempArrtimeTable_stnSpN.index==arrTrainNo_atStnS_ip1StnS)[0][0]         # S역 i+1열차의 S-1역 도착순서 저장
                                                        arrOrderOfTrain1 = np.where(tempArrtimeTable_stnSpN.index==arrTrainNo_atStnSmN_ip1StnSmN)[0][0]     # S-1역 i+1열차의 S-1역 도착순서 저장

                                                        # S-1역의 출발시각표 불러오기
                                                        tempDeptimeTable_stnSpN = returnSortedrtTableByStn(stationIdInteger=arrTimeDataStnSmN, rtTableByStn=depTableByStn, rtTable=myd3_dep_ttable) 
                                                        depOrderOfTrain0 = np.where(tempDeptimeTable_stnSpN.index==arrTrainNo_atStnS_ip1StnS)[0][0]         # S역 i+1열차의 S-1역 출발순서 저장
                                                        depOrderOfTrain1 = np.where(tempDeptimeTable_stnSpN.index==arrTrainNo_atStnSmN_ip1StnSmN)[0][0]     # S-1역 i+1열차의 S-1역 출발순서 저장
                                                        
                                                        # 어떤열차가 정말 앞선 열차인지 확인
                                                        if (tempArrtimeTable_stnSpN.iloc[arrOrderOfTrain0][0] is not pd.NaT) & (tempArrtimeTable_stnSpN.iloc[arrOrderOfTrain1][0] is not pd.NaT) :    # 두 데이터의 시각정보가 모두 유효하다면 (만약 한쪽 시각이 NaT인 경우 비교 불가)
                                                            if arrOrderOfTrain0 < arrOrderOfTrain1 :    # S역 i+1열차의 정보가 참
                                                                print("원래의 i+1열차가 맞는 것으로 확인되었습니다: "+str(arrTrainNo_atStnS_ip1StnS)) 
                                                                arrTrainNo_atStnS_ip1StnS = arrTrainNo_atStnS_ip1StnS
                                                                arrDataGoodStnS = True
                                                                #useTrain_ip1StnSmN_atStnSmN = True
                                                                #arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                #previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN)
                                                            
                                                            else :                                      # S-1역 i+1열차의 정보가 참 - 같을 순 없음. 같았으면 위에서 같다고 하고 끝났을것임
                                                                print("                                    i+1열차 ID가 수정되었습니다: "+str(arrTrainNo_atStnS_ip1StnS)+" --> "+str(arrTrainNo_atStnSmN_ip1StnSmN))
                                                                arrTrainNo_atStnS_ip1StnS = arrTrainNo_atStnSmN_ip1StnSmN
                                                                arrDataGoodStnS = True
                                                                useTrain_ip1StnSmN_atStnSmN = True
                                                                arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN) 
                                                        
                                                        elif (tempDeptimeTable_stnSpN.iloc[depOrderOfTrain0][0] is not pd.NaT) & (tempDeptimeTable_stnSpN.iloc[depOrderOfTrain1][0] is not pd.NaT) :
                                                            if depOrderOfTrain0 < depOrderOfTrain1 :    # S역 i+1열차의 정보가 참
                                                                print("원래의 i+1열차가 맞는 것으로 확인되었습니다: "+str(arrTrainNo_atStnS_ip1StnS)) 
                                                                arrTrainNo_atStnS_ip1StnS = arrTrainNo_atStnS_ip1StnS
                                                                arrDataGoodStnS = True
                                                                #useTrain_ip1StnSmN_atStnSmN = True
                                                                #arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                #previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN)
                                                            
                                                            else :                                      # S-1역 i+1열차의 정보가 참 - 같을 순 없음. 같았으면 위에서 같다고 하고 끝났을것임
                                                                print("                                    i+1열차 ID가 수정되었습니다: "+str(arrTrainNo_atStnS_ip1StnS)+" --> "+str(arrTrainNo_atStnSmN_ip1StnSmN)) 
                                                                arrTrainNo_atStnS_ip1StnS = arrTrainNo_atStnSmN_ip1StnSmN
                                                                arrDataGoodStnS = True
                                                                useTrain_ip1StnSmN_atStnSmN = True
                                                                arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN)
                                                        
                                                        else :
                                                            print("S-1역:"+str(arrTimeDataStnSmN)+"의 출도착 시각표 모두에서 S역 i+1열차와 S-1역 i+1열차의 출도착시각이 없습니다.")
                                                        
                                                            # 4.5.3.16.11. 이전 역 별 i+1열차 id 저장
                                                            arrTrain_ip1_byPreviousStn.append( arrTrainNo_atStnSmN_ip1StnSmN )
                                                            previousStnList.append( arrTimeDataStnSmN )

                                                            # 4.5.3.16.12. 만약 마지막 두 역의 i+1열차 정보가 맞다면, S역의 정보가 틀렸다고 보는게 합리적일것임
                                                            if ( (stnS_dataChkErrCnt >= 2) & (len(arrTrain_ip1_byPreviousStn)%2 == 1)) : # & (arrTrain_ip1_byPreviousStn[(len(arrTrain_ip1_byPreviousStn)-1)] == arrTrain_ip1_byPreviousStn[(len(arrTrain_ip1_byPreviousStn)-2)])  :

                                                                #arrTrainNo_atStnS_ip1StnS, arrDataGoodStnS, useTrain_ip1StnSmN_atStnSmN = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=arrTrain_ip1_byPreviousStn)
                                                                arrTrainNo_atStnS_ip1StnS, feasibleStnId, arrDataGoodStnS, useTrain_ip1StnSmN_atStnSmN = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=arrTimeDataStnSmN, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId,  trainNoByPreviousStnList=arrTrain_ip1_byPreviousStn, stnIdByPreviousStnList=previousStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                                                            
                                                            else : 
                                                                # 4.5.3.16.13. 한개 더 전 역에 대해서 조회할 수 있도록 False 유지
                                                                #stationCnt = stationCnt - 1
                                                                arrDataGoodStnS = False


                                                else : # if arrDepDataAvailableRes[0] != False :
                                                    arrDepDataAvailableFlag = arrDepDataAvailableRes[0]
                                                    #stnS_dataChkErrCnt = arrDepDataAvailableRes[1]
                                                    if len(arrTrain_ip1_byPreviousStn) >= 3 : 
                                                        #arrTrainNo_atStnS_ip1StnS, arrDataGoodStnS, useTrain_ip1StnSmN_atStnSmN = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=arrTrain_ip1_byPreviousStn)
                                                        arrTrainNo_atStnS_ip1StnS, feasibleStnId, arrDataGoodStnS, useTrain_ip1StnSmN_atStnSmN = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=arrTimeDataStnSmN, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=arrTrain_ip1_byPreviousStn, stnIdByPreviousStnList=previousStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                                                    
                                                    elif (len(arrTrain_ip1_byPreviousStn) == 2) & (stnS_dataChkErrCnt >= 4) :
                                                        if (arrTrain_ip1_byPreviousStn[0] == arrTrain_ip1_byPreviousStn[1]) :
                                                            #arrTrainNo_atStnS_ip1StnS, arrDataGoodStnS, useTrain_ip1StnSmN_atStnSmN = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=arrTrain_ip1_byPreviousStn)
                                                            arrTrainNo_atStnS_ip1StnS, feasibleStnId, arrDataGoodStnS, useTrain_ip1StnSmN_atStnSmN = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=arrTimeDataStnSmN, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=arrTrain_ip1_byPreviousStn, stnIdByPreviousStnList=previousStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                                                        else : # HERE
                                                            if previousStnList[0] == previousStnList[1] :
                                                                
                                                                tempArrtimeTable_stnSpN = returnSortedrtTableByStn(stationIdInteger=previousStnList[0], rtTableByStn=arrTableByStn, rtTable=myd3_arr_ttable) 
                                                                arrOrderOfTrain0 = np.where(tempArrtimeTable_stnSpN.index==arrTrain_ip1_byPreviousStn[0])[0][0]         # S역 i+1열차의 S-1역 도착순서 저장
                                                                arrOrderOfTrain1 = np.where(tempArrtimeTable_stnSpN.index==arrTrain_ip1_byPreviousStn[1])[0][0]     # S-1역 i+1열차의 S-1역 도착순서 저장

                                                                # S-1역의 출발시각표 불러오기
                                                                tempDeptimeTable_stnSpN = returnSortedrtTableByStn(stationIdInteger=previousStnList[0], rtTableByStn=depTableByStn, rtTable=myd3_dep_ttable) 
                                                                depOrderOfTrain0 = np.where(tempDeptimeTable_stnSpN.index==arrTrain_ip1_byPreviousStn[0])[0][0]         # S역 i+1열차의 S-1역 출발순서 저장
                                                                depOrderOfTrain1 = np.where(tempDeptimeTable_stnSpN.index==arrTrain_ip1_byPreviousStn[1])[0][0]     # S-1역 i+1열차의 S-1역 출발순서 저장
                                                                
                                                                # 어떤열차가 정말 앞선 열차인지 확인
                                                                if (tempArrtimeTable_stnSpN.iloc[arrOrderOfTrain0][0] is not pd.NaT) & (tempArrtimeTable_stnSpN.iloc[arrOrderOfTrain1][0] is not pd.NaT) :    # 두 데이터의 시각정보가 모두 유효하다면 (만약 한쪽 시각이 NaT인 경우 비교 불가)
                                                                    if arrOrderOfTrain0 < arrOrderOfTrain1 :    # S역 i+1열차의 정보가 참
                                                                        arrTrainNo_atStnS_ip1StnS = arrTrain_ip1_byPreviousStn[0]
                                                                        arrDataGoodStnS = True
                                                                        useTrain_ip1StnSmN_atStnSmN = True
                                                                        arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                        previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN)
                                                                        print("(누락열차 보정중) i+1열차 ID가 확인되었습니다 : "+str(arrTrainNo_atStnS_ip1StnS)) 
                                                                    
                                                                    else :                                      # S-1역 i+1열차의 정보가 참 - 같을 순 없음. 같았으면 위에서 같다고 하고 끝났을것임
                                                                        arrTrainNo_atStnS_ip1StnS = arrTrain_ip1_byPreviousStn[1]
                                                                        arrDataGoodStnS = True
                                                                        useTrain_ip1StnSmN_atStnSmN = True
                                                                        arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                        previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN) 
                                                                        print("(누락열차 보정중) i+1열차 ID가 확인되었습니다 : "+str(arrTrainNo_atStnS_ip1StnS)) 
                                                                
                                                                elif (tempDeptimeTable_stnSpN.iloc[depOrderOfTrain0][0] is not pd.NaT) & (tempDeptimeTable_stnSpN.iloc[depOrderOfTrain1][0] is not pd.NaT) :
                                                                    if depOrderOfTrain0 < depOrderOfTrain1 :    # S역 i+1열차의 정보가 참
                                                                        arrTrainNo_atStnS_ip1StnS = arrTrain_ip1_byPreviousStn[0]
                                                                        arrDataGoodStnS = True
                                                                        useTrain_ip1StnSmN_atStnSmN = True
                                                                        arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                        previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN)
                                                                        print("(누락열차 보정중) i+1열차 ID가 확인되었습니다 : "+str(arrTrainNo_atStnS_ip1StnS)) 
                                                                    
                                                                    else :                                      # S-1역 i+1열차의 정보가 참 - 같을 순 없음. 같았으면 위에서 같다고 하고 끝났을것임
                                                                        arrTrainNo_atStnS_ip1StnS = arrTrain_ip1_byPreviousStn[1]
                                                                        arrDataGoodStnS = True
                                                                        useTrain_ip1StnSmN_atStnSmN = True
                                                                        arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                        previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN)
                                                                        print("(누락열차 보정중) i+1열차 ID가 확인되었습니다 : "+str(arrTrainNo_atStnS_ip1StnS)) 

                                                            else : 
                                                                print("S역 누락열차 확인 리스트의 열차 ID 두개가 정보가 서로 다릅니다")
                                                
                                                    else :
                                                        print("S역 누락열차 확인 필요 [2]")

                                                # 무한루프 방지 변수 초기화
                                                stnS_dataChkErrCnt = stnS_dataChkErrCnt + 1
                                            
                                            if arrDataGoodStnS == True :
                                                # 4.5.3.10. s역부터 다음 피추월역까지의 출도착 정보를 조회하여 출도착 정보가 둘 다 있는 경우를 찾아서 확인
                                                stationCnt = -1 # 역 추가검토 관련 변수 초기화. S역 출발데이터부터 검색해야 하므로 -1 로 초기화
                                                depDataGoodStnS = False
                                                stnS_dataChkErrCnt = 0

                                                # 4.5.3.11. 출발역의 다음역에 대해 분석대상열차와 누락된 열차정보를 조회
                                                #           while문을 활용해 하류 역으로 이동해가면서 S역의 출발열차의 일관성 검토
                                                while (depDataGoodStnS == False) & (stnS_dataChkErrCnt < max_stnS_dataChkErrCnt) :
                                                    stationCnt = stationCnt + 1
                                                    searchingToNextStn = True # 다음역 방향으로 조회하는지 결정하는 변수 초기화
                                                    departureTableFlag = True # 출발역 데이터부터 조회할 수 있도록 True로 초기화

                                                    nextOvertakeStnId = stations[ findNextOvtStnOrder(lineNm=lineNm, bnd=bnd, nextStnTF=searchingToNextStn, currentOvtStnOrder=overtakeStation, ovtStnIdList=overtakeStaList, wholeStnIdList=stations) ]

                                                    # 4.5.3.11.1. 다음 역의 i열차와 i+1열차(정보 누락되서 빠졌던 열차)의 출도착시각 얻기
                                                    arrDepDataAvailableRes = findingdepArrTimeAtStaSpN(stationCnt, searchingToNextStn, departureTableFlag, stations, overtakeStaList, overtakeStation, lineNm, bnd, myd3_arr_ttable, myd3_dep_ttable, arrTrainNo_atStnS_ip0StnS, arrTrainNo_atStnS_ip1StnS, arrTimeTable_stnS, train, trainCnt)

                                                    # 4.5.3.11.2. findingdepArrTimeAtStaSpN 함수 결과 저장
                                                    if arrDepDataAvailableRes[0] != False :
                                                        arrDepDataAvailableFlag = arrDepDataAvailableRes[0]
                                                        depTrainOrderOf_ip0StnS_atStnSpN = arrDepDataAvailableRes[1]
                                                        depTime_atStnSpN_ip0StnS = arrDepDataAvailableRes[2]
                                                        depTrainOrderOf_ip1StnS_atStnSpN = arrDepDataAvailableRes[3] 
                                                        depTime_atStnSpN_ip1StnSpN = arrDepDataAvailableRes[4]
                                                        depTimeTable_stnSpN = arrDepDataAvailableRes[5]
                                                        depTimeDataStnSpn = arrDepDataAvailableRes[6]

                                                        # 4.5.3.11.3. 현재 역과 전 역의 후속열차 ID 비교
                                                        # 4.5.3.11.3.1. # S-1역 i+1열차(누락됐던 열차)의 열차ID 저장
                                                        depTrainNo_atStnSpN_ip1StnSpN = depTimeTable_stnSpN.index[depTrainOrderOf_ip1StnS_atStnSpN]

                                                        # 4.5.3.9.2. # S역과 S-1역의 i+1열차 ID 비교
                                                        if (arrTrainNo_atStnS_ip1StnS == depTrainNo_atStnSpN_ip1StnSpN) & ( (len(depTrain_ip1_byForwardStn) == 0) | (len(depTrain_ip1_byForwardStn) >= 2) )  :
                                                            if (depTime_atStnSpN_ip0StnS is not pd.NaT) & (depTime_atStnSpN_ip1StnSpN is not pd.NaT) : # 둘 다 있는게 중요함. 둘 다 있어야, 두 열차 사이의 간격 정보와 trainCnt 정보를 비교해볼 수 있음

                                                                # 4.5.3.9.4. 출발역 출발데이터 점검 Flag 초기화
                                                                depDataGoodStnS = True    

                                                            else :
                                                                print("S역과 S-1역의 열차정보는 같으나 S-1역의 i, i+1열차 중 하나의 시각정보가 없음")
                                                                depDataGoodStnS = False
                                                        
                                                        else : # if (arrTrainNo_atStnS_ip1StnS == depTrainNo_atStnSpN_ip1StnSpN) & (depTime_atStnSpN_ip0StnS is not pd.NaT) & (depTime_atStnSpN_ip1StnSpN is not pd.NaT) :
                                                            # 4.5.3.9.5. S역과 S-1역의 i+1열차 ID가 다르다면
                                                            #print("ERR. S역과 S-1역의 i+1열차 ID가 다릅니다.")

                                                            # 4.5.3.9.6. 이전 역 별 i+1열차 id 저장
                                                            depTrain_ip1_byForwardStn.append( depTrainNo_atStnSpN_ip1StnSpN )
                                                            forwardStnList.append( depTimeDataStnSpn )

                                                            # 4.5.3.9.7. 만약 마지막 두 역의 i+1열차 정보가 맞다면, S역의 정보가 틀렸다고 보는게 합리적일것임
                                                            if (stnS_dataChkErrCnt >= 2)  & (len(depTrain_ip1_byForwardStn)%2 == 1)  :
                                                                
                                                                #arrTrainNo_atStnS_ip1StnS, depDataGoodStnS, useTrainTF_notUsed = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=depTrain_ip1_byForwardStn)
                                                                arrTrainNo_atStnS_ip1StnS, feasibleStnId, depDataGoodStnS, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=depTimeDataStnSpn, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=depTrain_ip1_byForwardStn, stnIdByPreviousStnList=forwardStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)

                                                            else : 
                                                                # 4.5.3.9.6. 한개 더 전 역에 대해서 조회할 수 있도록 False 유지
                                                                #stationCnt = stationCnt - 1
                                                                depDataGoodStnS = False

                                                    
                                                    else : # if arrDepDataAvailableRes[0] != False :
                                                        arrDepDataAvailableFlag = arrDepDataAvailableRes[0]
                                                        #stnS_dataChkErrCnt = arrDepDataAvailableRes[1]
                                                        # 출발열차 후보군 점검 관련 : 
                                                        if len(depTrain_ip1_byForwardStn) >= 3 : 
                                                            #arrTrainNo_atStnS_ip1StnS, depDataGoodStnS, useTrainTF_notUsed = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=depTrain_ip1_byForwardStn)
                                                            arrTrainNo_atStnS_ip1StnS, feasibleStnId, depDataGoodStnS, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=depTimeDataStnSpn, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=depTrain_ip1_byForwardStn, stnIdByPreviousStnList=forwardStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                                                        
                                                        elif (len(depTrain_ip1_byForwardStn) == 2) & (stnS_dataChkErrCnt >= 4) :
                                                            if (depTrain_ip1_byForwardStn[0] == depTrain_ip1_byForwardStn[1]) :
                                                                #arrTrainNo_atStnS_ip1StnS, depDataGoodStnS, useTrainTF_notUsed = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=depTrain_ip1_byForwardStn)
                                                                arrTrainNo_atStnS_ip1StnS, feasibleStnId, depDataGoodStnS, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=depTimeDataStnSpn, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=depTrain_ip1_byForwardStn, stnIdByPreviousStnList=forwardStnList, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                                                                
                                                            else :
                                                                print("S역 누락 출발열차 확인 리스트의 열차 ID 두개가 정보가 서로 다릅니다 2 | stnS_dataChkErrCnt:"+str(stnS_dataChkErrCnt))

                                                        else :
                                                            print("S역 누락 출발열차 확인 필요 2 | stnS_dataChkErrCnt:"+str(stnS_dataChkErrCnt)) 
                                                    
                                                    # 무한루프 방지 변수 초기화
                                                    stnS_dataChkErrCnt = stnS_dataChkErrCnt + 1
                                                
                                                if arrDataGoodStnS == True & depDataGoodStnS == True :
                                                    arrDepBothDataGood = True
                                                else :
                                                    arrDepBothDataGood = False
                                            else :
                                                print("출발역 도착열차 순서의 일관성이 확보되지 않았음 3")
                                                arrDepBothDataGood = False
                                        
                                        else :
                                            arrDepBothDataGood = False # 출도착정보 가용여부 Flag False로 초기화
                                            print("오류 확인 필요! 3")
                                            
                                    
                                    # =====================
                                    # 이 행을 지나면 다음 역에 대한 출도착 시각에 대해 조회가 끝난 상태임. 선후행열차의 출도착 관련 네 종류 데이터가 다 있는상태임.

                                    if arrDepBothDataGood == True : # 출발, 도착열차 데이터의 인접역 간 1차 일관성 검토가 정상인것으로 확인됐으면, 이 절차를 통해 해당 열차에 대한 검색을 마칠 지 판단

                                        # 4.5.8. 출발데이터 상에서의 순서 차이와 S역 도착 데이터 상에서의 분석대상열차 순서 차이 계산
                                        trainArrDepOrderDiff = depTrainOrderOf_ip0StnS_atStnSpN - depTrainOrderOf_ip1StnS_atStnSpN

                                        # 4.5.9. 출발데이터 상에서의 순서 차이와 S역 도착 데이터 상에서의 분석대상열차 순서 차이가 trainCnt보다 작거나 같으면
                                        #         trainCnt + 0 : 선행 도착이 일반, 후착이 급행이어서 출발할 때 역전이 된 경우
                                        #         trainCnt + 2 : 선행 출발이 급행, 후착이 일반인데 선행 급행은 그 앞의 일반을 추월했고, 후착 열차가 그 뒤에 온 급행열차에게 추월 당해 총 3대 차이가 나는 상황
                                        if (abs(trainArrDepOrderDiff) <= (trainCnt)) | (abs(trainArrDepOrderDiff) == (trainCnt+2)) : #(abs(trainArrDepOrderDiff) == trainCnt) | (abs(trainArrDepOrderDiff) == (trainCnt+2)) :
                                            
                                            # 4.5.9.1. 피추월횟수 계산
                                            numOfOvertake = analyzingOvertake(trainArrDepOrderDiff, arrTrainNo_atStnS_ip0StnS, arrTime_atStnS_ip0StnS, depTime_atStnSpN_ip0StnS, arrTrainNo_atStnS_ip1StnS, arrTime_atStnS_ip1StnS, depTime_atStnSpN_ip1StnSpN) 
                                        
                                            # 4.5.9.2. 피추월횟수 계산 결과 검토 및 추월당한 경우 추월한 열차정보 기재. 
                                            #         updatingNumOfOvertake 함수의 결과로서 계속 피추월횟수를 계산해야하는지에 대한 
                                            #         True(피추월횟수 분석 종료), False(피추월횟수 추가 분석 필요)값을 제공함
                                            updatingNumOfOvertakeRes = updatingNumOfOvertake(statnId=overtakeStaList[overtakeStation], nOvertakeTable=numOvertakeTable, targetTrainNo=arrTrainNo_atStnS_ip0StnS, numberOfOvertakeFreq=numOfOvertake, overtakeTrnNoTable=overtakeTrainNoTable, overtakingTrainNo=arrTrainNo_atStnS_ip1StnS)
                                            
                                            endOvertakingAnalysis = updatingNumOfOvertakeRes[0] # 피추월횟수가 0이면 True, >=1 이면 False : 1회 이상 추월되는 경우를 확인하기 위해 추월이 안일어나는것을 확인할 때 까지 반복시행함
                                            
                                            numOvertakeTable = updatingNumOfOvertakeRes[1]
                                            overtakeTrainNoTable = updatingNumOfOvertakeRes[2]
                                            #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                            
                                            # 4.5.9.3. S역 i번째 도착열차 분석에 대한 임시 Sequence update, 추월열차정보 update
                                            tempArrSeq, tempDepSeq, overtakedTrnIdList = updatingTempArrDepSeq(funcTempArrSeq=tempArrSeq, funcTempDepSeq=tempDepSeq, funcTrainNo_ip0=arrTrainNo_atStnS_ip0StnS, funcTrainNo_ip1=arrTrainNo_atStnS_ip1StnS, numOfOvertake=numOfOvertake, funcOvertakedTrnIdList=overtakedTrnIdList)

                                            # 4.5.9.4. 만약 현재 분석대상열차가 누락열차였고, 그 열차가 추월당한것으로 나타나면 
                                            if (numOfOvertake == 1) & (natTF_ip0StnS == True) :
                                                
                                                # 다음 조회에서 다시 현재 분석대상열차로부터 시작할 수 있도록 nextTrainByForce 로 열차ID 지정
                                                nextTrainByForce = arrTrainNo_atStnS_ip0StnS
                                        
                                        # 4.5.9. 출발데이터 상에서의 순서 차이와 S역 도착 데이터 상에서의 분석대상열차 순서 차이가 trainCnt보다 작거나 같으면
                                        #         trainCnt + 1 : 
                                        #               case 1 : 선행 출발이 급행, 후착이 일반인데 선행 급행은 그 앞의 일반을 추월 안했고, 후착 열차가 그 뒤에 온 급행열차에게 추월 당해 총 2대가 차이나는 상황
                                        #                        -->  마곡나루에서 첫 대피가 이뤄지는 시점. 대피 이뤄지기 전 까지는 마곡나루에서 급행이 일반을 추월 안하다가 처음 추월하게 되는 시점에 trainCnt + 1 상황이 발생
                                        #               case 2 : 출발데이터에서의 누락

                                        elif (abs(trainArrDepOrderDiff) <= (trainCnt+2)) : #(abs(trainArrDepOrderDiff) == trainCnt) | (abs(trainArrDepOrderDiff) == (trainCnt+2)) :

                                            # 4.5.10. 출발역 관련 누락정보를 보정하여 피추월횟수 계수
                                            # 4.5.10.1. 누락된 열차정보 확인


                                            # 4.5.10.2. 역 별 i+1열차 ID를 저장 : S역부터 하나씩 거슬러올라가면서 저장
                                            #arrTrain_ip1_byPreviousStn = [arrTrainNo_atStnS_ip1StnS]
                                            #previousStnList = [overtakeStaList[overtakeStation]]
                                            #stnS_dataChk = False # 최초 False로 초기화. while문을 돌다가 전 역 데이터 관련 문제가 해결되면 True로 갱신되며 while에서 빠져나오게 됨
                                            stationCnt = 0
                                            arrDataGoodStnS_2 = False
                                            arrTrain_ip1_byPreviousStn_2 = []
                                            previousStnList_2 = []

                                            # while문을 활용해 역을 거슬러가면서 검지
                                            stnS_dataChkErrCnt = 0
                                            while (arrDataGoodStnS_2 == False) & (stnS_dataChkErrCnt < max_stnS_dataChkErrCnt) :
                                                # 4.5.10.3. 출발역의 전역에 대해 분석대상열차와 누락된 열차정보를 조회
                                                stationCnt = stationCnt - 1 # 역 추가검토 관련 변수 초기화 : 전 역을 조회하기 위해 -1로 지정
                                                searchingToNextStn = False # 다음역 방향으로 조회하는지 결정하는 변수 초기화 : False를 줘서 전 역 방향으로 조회되도록 설정
                                                departureTableFlag = False # 도착 데이터부터 조회할 수 있도록 parameter를 변경

                                                nextOvertakeStnId = stations[ findNextOvtStnOrder(lineNm=lineNm, bnd=bnd, nextStnTF=searchingToNextStn, currentOvtStnOrder=overtakeStation, ovtStnIdList=overtakeStaList, wholeStnIdList=stations) ]

                                                # 4.5.3.7. 전 역의 i열차와 i+1열차(정보 누락되서 빠졌던 열차)의 출도착시각 얻기
                                                arrDepDataAvailableRes = findingdepArrTimeAtStaSpN(stationCnt, searchingToNextStn, departureTableFlag, stations, overtakeStaList, overtakeStation, lineNm, bnd, myd3_arr_ttable, myd3_dep_ttable, arrTrainNo_atStnS_ip0StnS, arrTrainNo_atStnS_ip1StnS, arrTimeTable_stnS, train, trainCnt)
                                                
                                                # 4.5.3.8. findingdepArrTimeAtStaSpN 함수 결과 저장
                                                if arrDepDataAvailableRes[0] != False :
                                                    arrDepDataAvailableFlag = arrDepDataAvailableRes[0] # 출도착정보 가용여부 Flag
                                                    arrTrainOrderOf_ip0StnS_atStnSmN = arrDepDataAvailableRes[1] # S역 i열차의 S-1역 도착(출발) 순서
                                                    arrTime_atStnSmN_ip0StnS = arrDepDataAvailableRes[2] # S역 i열차의 S-1역 도착(출발) 시각
                                                    arrTrainOrderOf_ip1StnSmN_atStnSmN = arrDepDataAvailableRes[3] # S-1역 i+1열차(누락됐던 열차)의 S-1역 도착(출발) 순서 (S-1역 기준으로 i열차의 후속열차를 재검색)
                                                    arrTime_atStnSmN_ip1StnSmN = arrDepDataAvailableRes[4] # S-1역 i+1열차(누락됐던 열차)의 S-1역 도착(출발) 시간 (S-1역 기준으로 i열차의 후속열차를 재검색)
                                                    arrTimeTable_stnSmN = arrDepDataAvailableRes[5] # S-1역의 열차 관련 시각표
                                                    arrTimeDataStnSmN = arrDepDataAvailableRes[6] # S-1역 역 ID
                                                
                                                    # 4.5.3.9. 현재 역과 전 역의 후속열차 ID 비교
                                                    # 4.5.3.9.1. # S-1역 i+1열차(누락됐던 열차)의 열차ID 저장
                                                    arrTrainNo_atStnSmN_ip1StnSmN = arrTimeTable_stnSmN.index[arrTrainOrderOf_ip1StnSmN_atStnSmN]

                                                    # 4.5.3.9.2. # S역과 S-1역의 i+1열차 ID 비교 : 
                                                    #       S역 i, i+1열차의 S역출발 또는 S+1역 시각표를 보니 그 사이에 다른 열차가 있었기때문에 여기서는 전 역(S-1역)의 i+1열차가 다른 열차여야 함
                                                    if (arrTrainNo_atStnS_ip1StnS != arrTrainNo_atStnSmN_ip1StnSmN) :
                                                        if (arrTime_atStnS_ip1StnS is not pd.NaT) & (arrTime_atStnSmN_ip1StnSmN is not pd.NaT) :
                                                            # 4.5.3.9.3. S역과 S-1역의 i+1열차 ID가 같다면
                                                            #print("OK")

                                                            # 4.5.3.9.4. 출발역 데이터 점검 Flag 초기화
                                                            arrDataGoodStnS_2 = True

                                                            # 4.5.3.9.5. 출발역 도착데이터 누락 열차번호 관련 리스트 초기화 : 추후 이 열차부터 조회를 시작해나갈 수 있도록 열차번호 저장
                                                            arrTrain_ip1_byPreviousStn = appendTwice(targetList=arrTrain_ip1_byPreviousStn, appendingValue=arrTrainNo_atStnSmN_ip1StnSmN) 
                                                            previousStnList = appendTwice(targetList=previousStnList, appendingValue=arrTimeDataStnSmN) 
                                                            
                                                            useTrain_ip1StnSmN_atStnSmN = True
                                                            print("                                    i+1열차 ID가 수정되었습니다: "+str(arrTrainNo_atStnS_ip1StnS)+" --> "+str(arrTrainNo_atStnSmN_ip1StnSmN))

                                                            #arrTrain_ip1_byPreviousStn = [] # 출발역 도착데이터 누락 열차번호 저장용 리스트 초기화
                                                            #previousStnList = [] # 출발역 도착데이터 누락 열차 발생 역 ID  저장용 리스트 초기화

                                                        else :
                                                            # 4.5.10.6. S-1역의 i열차 도착시각은 있으나 i+1열차 도착시각이 없는 경우
                                                            if  (arrTime_atStnSmN_ip0StnS is not pd.NaT) & (arrTime_atStnSmN_ip1StnSmN is pd.NaT) :
                                                                #print("\nS역 도착정보 중 i+1 열차-"+str(arrTrainNo_atStnS_ip0StnS)+" 의 정보가 없음\n")

                                                                # 4.5.10.6.1. S-1역 i열차가 막차인지 확인
                                                                if arrTrainNo_atStnS_ip0StnS in lstcarTrainNo2List :
                                                                    # 4.5.10.6.2. 막차가 맞다면 피추월횟수를 0으로 강제 지정하기
                                                                    updatingNumOfOvertakeRes = updatingNumOfOvertake(statnId=overtakeStaList[overtakeStation], nOvertakeTable=numOvertakeTable, targetTrainNo=arrTime_atStnSmN_ip0StnS, numberOfOvertakeFreq=0, overtakeTrnNoTable=overtakeTrainNoTable, overtakingTrainNo=arrTime_atStnSmN_ip1StnSmN)
                                                                    
                                                                    endOvertakingAnalysis = updatingNumOfOvertakeRes[0] # 피추월횟수가 0이면 True, >=1 이면 False : 1회 이상 추월되는 경우를 확인하기 위해 추월이 안일어나는것을 확인할 때 까지 반복시행함
                                                                    
                                                                    numOvertakeTable = updatingNumOfOvertakeRes[1]
                                                                    overtakeTrainNoTable = updatingNumOfOvertakeRes[2]

                                                                    # 4.5.3.14.3. 도착 순서 저장
                                                                    tempArrSeq = [arrTrainNo_atStnS_ip0StnS]

                                                                    # 4.5.3.14.4. 출발 순서 저장
                                                                    tempDepSeq = [arrTrainNo_atStnS_ip0StnS]

                                                                    # 추월열차 ID 누적해서 저장하기
                                                                    if numOfOvertake >= 1 :
                                                                        overtakedTrnIdList.append(arrTrainNo_atStnS_ip1StnS)

                                                                # 4.5.10.6.3.
                                                                elif arrTime_atStnSmN_ip0StnS > arrTime_atStnSmN_ip1StnSmN :
                                                                    #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                                                    print("here 13-2-1") 
                                                                
                                                                elif arrTime_atStnSmN_ip0StnS == arrTime_atStnSmN_ip1StnSmN :
                                                                    print("\nLast recored of train had been analyzed 2. ====================\n")

                                                                # 4.5.10.6.4.
                                                                else :
                                                                    #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                                                    print("here 13-2-2")
                                                                
                                                            # 4.5.10.7. S-1역의 i열차, i+1열차 모두 도착정보가 없는 경우
                                                            #          이런 경우는 열차 운행정보가 없어 아얘 
                                                            elif (arrTime_atStnSmN_ip0StnS is pd.NaT) & (arrTime_atStnSmN_ip1StnSmN is pd.NaT):
                                                                if numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTime_atStnSmN_ip0StnS )[0][0] ] == -99 :    
                                                                    #print("There's no arr or dep time information of train i and i+1. train_i: "+str(arrTrainNo_atStnS_ip0StnS)+",  train_i+1: "+str(arrTrainNo_atStnS_ip1StnS) )
                                                                    numOvertakeTable[str(overtakeStaList[overtakeStation])][arrTime_atStnSmN_ip0StnS] = np.nan
                                                                    endOvertakingAnalysis = True
                                                                    #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                                                
                                                                elif np.isnan(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTime_atStnSmN_ip0StnS )[0][0] ]) :    
                                                                    #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                                                    endOvertakingAnalysis = True
                                                                
                                                                elif numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTime_atStnSmN_ip0StnS )[0][0] ] == 0 :    
                                                                    print("이미 초기화되었던 이력이 있음 2")  

                                                                else :
                                                                    #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                                                    print("here 112")

                                                            elif (arrTime_atStnSmN_ip0StnS is pd.NaT) & (arrTime_atStnSmN_ip1StnSmN is not pd.NaT) :
                                                                print("\nS-1역 도착정보 중 i 열차-"+str(arrTrainNo_atStnS_ip0StnS)+" 의 정보가 없음\n")
                                                                # 이 경우는 발생될 수 없음. S-1역 도착정보 기준으로 sorting 한 데이터에 대해 분석하므로, S역 i열차가 정보가 없을때는 그 이후에는 모든 데이터에 값이 없을것임.
                                                            
                                                            else :
                                                                print("오류 확인 필요! 3")
                                                    
                                                    else : # if (arrTrainNo_atStnS_ip1StnS == arrTrainNo_atStnSmN_ip1StnSmN) & (arrTime_atStnS_ip1StnS is not pd.NaT) & (arrTime_atStnSmN_ip1StnSmN is not pd.NaT) :
                                                        # 4.5.3.9.5. S역과 S-1역의 i+1열차 ID가 다르다면
                                                        #print("ERR. S역과 S-1역의 i+1열차 ID가 다릅니다.")

                                                        # 4.5.3.9.6. 이전 역 별 i+1열차 id 저장
                                                        arrTrain_ip1_byPreviousStn_2.append( arrTrainNo_atStnSmN_ip1StnSmN )
                                                        previousStnList_2.append( arrTimeDataStnSmN )

                                                        # 4.5.3.9.7. 만약 마지막 두 역의 i+1열차 정보가 맞다면, S역의 정보가 틀렸다고 보는게 합리적일것임
                                                        if (stnS_dataChkErrCnt >= 4) & (len(arrTrain_ip1_byPreviousStn_2) >= 3) :
                                                            #arrTrainNo_atStnS_ip1StnS, arrDataGoodStnS_2, useTrainTF_notUsed = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=arrTrain_ip1_byPreviousStn_2)
                                                            arrTrainNo_atStnS_ip1StnS, feasibleStnId, arrDataGoodStnS_2, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=arrTimeDataStnSmN, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=arrTrain_ip1_byPreviousStn_2, stnIdByPreviousStnList=previousStnList_2, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                                                        
                                                        elif (stnS_dataChkErrCnt >= 4) & (len(arrTrain_ip1_byPreviousStn_2) == 2) :
                                                            if (arrTrain_ip1_byPreviousStn_2[0] == arrTrain_ip1_byPreviousStn_2[1]) :
                                                                #arrTrainNo_atStnS_ip1StnS, arrDataGoodStnS_2, useTrainTF_notUsed = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=arrTrain_ip1_byPreviousStn_2)
                                                                arrTrainNo_atStnS_ip1StnS, feasibleStnId, arrDataGoodStnS_2, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=arrTimeDataStnSmN, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=arrTrain_ip1_byPreviousStn_2, stnIdByPreviousStnList=previousStnList_2, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                                                            else :
                                                                print("S역 누락열차 확인 리스트의 열차 ID 두개가 정보가 서로 다릅니다")

                                                        else : 
                                                            # 4.5.3.9.6. 한개 더 전 역에 대해서 조회할 수 있도록 False 유지
                                                            #stationCnt = stationCnt - 1
                                                            arrDataGoodStnS_2 = False

                                                else : # if arrDepDataAvailableRes[0] != False :
                                                    arrDepDataAvailableFlag = arrDepDataAvailableRes[0]
                                                    #stnS_dataChkErrCnt = arrDepDataAvailableRes[1]

                                                    if (stnS_dataChkErrCnt >= 4) & (len(arrTrain_ip1_byPreviousStn_2) >= 3) :
                                                        #arrTrainNo_atStnS_ip1StnS, arrDataGoodStnS_2, useTrainTF_notUsed = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=arrTrain_ip1_byPreviousStn_2)
                                                        arrTrainNo_atStnS_ip1StnS, feasibleStnId, arrDataGoodStnS_2, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=arrTimeDataStnSmN, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=arrTrain_ip1_byPreviousStn_2, stnIdByPreviousStnList=previousStnList_2, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)

                                                    elif (stnS_dataChkErrCnt >= 4) & (len(arrTrain_ip1_byPreviousStn_2) == 2) :
                                                        if (arrTrain_ip1_byPreviousStn_2[0] == arrTrain_ip1_byPreviousStn_2[1]) :
                                                            #arrTrainNo_atStnS_ip1StnS, arrDataGoodStnS_2, useTrainTF_notUsed = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=arrTrain_ip1_byPreviousStn_2)
                                                            arrTrainNo_atStnS_ip1StnS, feasibleStnId, arrDataGoodStnS_2, useTrainTF_notUsed = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=arrTimeDataStnSmN, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId, trainNoByPreviousStnList=arrTrain_ip1_byPreviousStn_2, stnIdByPreviousStnList=previousStnList_2, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)
                                                        
                                                        else : # HERE
                                                            if previousStnList_2[0] == previousStnList_2[1] :
                                                                
                                                                tempArrtimeTable_stnSpN = returnSortedrtTableByStn(stationIdInteger=previousStnList_2[0], rtTableByStn=arrTableByStn, rtTable=myd3_arr_ttable) 
                                                                arrOrderOfTrain0 = np.where(tempArrtimeTable_stnSpN.index==arrTrain_ip1_byPreviousStn_2[0])[0][0]         # S역 i+1열차의 S-1역 도착순서 저장
                                                                arrOrderOfTrain1 = np.where(tempArrtimeTable_stnSpN.index==arrTrain_ip1_byPreviousStn_2[1])[0][0]     # S-1역 i+1열차의 S-1역 도착순서 저장

                                                                # S-1역의 출발시각표 불러오기
                                                                tempDeptimeTable_stnSpN = returnSortedrtTableByStn(stationIdInteger=previousStnList_2[0], rtTableByStn=depTableByStn, rtTable=myd3_dep_ttable) 
                                                                depOrderOfTrain0 = np.where(tempDeptimeTable_stnSpN.index==arrTrain_ip1_byPreviousStn_2[0])[0][0]         # S역 i+1열차의 S-1역 출발순서 저장
                                                                depOrderOfTrain1 = np.where(tempDeptimeTable_stnSpN.index==arrTrain_ip1_byPreviousStn_2[1])[0][0]     # S-1역 i+1열차의 S-1역 출발순서 저장
                                                                
                                                                # 어떤열차가 정말 앞선 열차인지 확인
                                                                if (tempArrtimeTable_stnSpN.iloc[arrOrderOfTrain0][0] is not pd.NaT) & (tempArrtimeTable_stnSpN.iloc[arrOrderOfTrain1][0] is not pd.NaT) :    # 두 데이터의 시각정보가 모두 유효하다면 (만약 한쪽 시각이 NaT인 경우 비교 불가)
                                                                    if arrOrderOfTrain0 < arrOrderOfTrain1 :    # S역 i+1열차의 정보가 참
                                                                        arrTrainNo_atStnS_ip1StnS = arrTrain_ip1_byPreviousStn_2[0]
                                                                        arrDataGoodStnS_2 = True
                                                                        #useTrain_ip1StnSmN_atStnSmN = True
                                                                        arrTrain_ip1_byPreviousStn_2 = appendTwice(targetList=arrTrain_ip1_byPreviousStn_2, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                        previousStnList_2 = appendTwice(targetList=previousStnList_2, appendingValue=arrTimeDataStnSmN)
                                                                        print("(2차 보정) i+1열차 ID가 확인되었습니다 : "+str(arrTrainNo_atStnS_ip1StnS)) 
                                                                    
                                                                    else :                                      # S-1역 i+1열차의 정보가 참 - 같을 순 없음. 같았으면 위에서 같다고 하고 끝났을것임
                                                                        arrTrainNo_atStnS_ip1StnS = arrTrain_ip1_byPreviousStn_2[1]
                                                                        arrDataGoodStnS_2 = True
                                                                        #useTrain_ip1StnSmN_atStnSmN = True
                                                                        arrTrain_ip1_byPreviousStn_2 = appendTwice(targetList=arrTrain_ip1_byPreviousStn_2, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                        previousStnList_2 = appendTwice(targetList=previousStnList_2, appendingValue=arrTimeDataStnSmN) 
                                                                        print("(2차 보정) i+1열차 ID가 확인되었습니다 : "+str(arrTrainNo_atStnS_ip1StnS)) 
                                                                
                                                                elif (tempDeptimeTable_stnSpN.iloc[depOrderOfTrain0][0] is not pd.NaT) & (tempDeptimeTable_stnSpN.iloc[depOrderOfTrain1][0] is not pd.NaT) :
                                                                    if depOrderOfTrain0 < depOrderOfTrain1 :    # S역 i+1열차의 정보가 참
                                                                        arrTrainNo_atStnS_ip1StnS = arrTrain_ip1_byPreviousStn_2[0]
                                                                        arrDataGoodStnS_2 = True
                                                                        #useTrain_ip1StnSmN_atStnSmN = True
                                                                        arrTrain_ip1_byPreviousStn_2 = appendTwice(targetList=arrTrain_ip1_byPreviousStn_2, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                        previousStnList_2 = appendTwice(targetList=previousStnList_2, appendingValue=arrTimeDataStnSmN)
                                                                        print("(2차 보정) i+1열차 ID가 확인되었습니다 : "+str(arrTrainNo_atStnS_ip1StnS)) 
                                                                    
                                                                    else :                                      # S-1역 i+1열차의 정보가 참 - 같을 순 없음. 같았으면 위에서 같다고 하고 끝났을것임
                                                                        arrTrainNo_atStnS_ip1StnS = arrTrain_ip1_byPreviousStn_2[1]
                                                                        arrDataGoodStnS_2 = True
                                                                        #useTrain_ip1StnSmN_atStnSmN = True
                                                                        arrTrain_ip1_byPreviousStn_2 = appendTwice(targetList=arrTrain_ip1_byPreviousStn_2, appendingValue=arrTrainNo_atStnS_ip1StnS) 
                                                                        previousStnList_2 = appendTwice(targetList=previousStnList_2, appendingValue=arrTimeDataStnSmN)
                                                                        print("(2차 보정) i+1열차 ID가 확인되었습니다 : "+str(arrTrainNo_atStnS_ip1StnS)) 
                                                                
                                                                else :
                                                                    print("S-1역:"+str(arrTimeDataStnSmN)+"의 출도착 시각표 모두에서 S역 i+1열차와 S-1역 i+1열차의 출도착시각이 없습니다.")
                                                                
                                                                    # 4.5.3.9.6. 이전 역 별 i+1열차 id 저장
                                                                    arrTrain_ip1_byPreviousStn_2.append( arrTrainNo_atStnSmN_ip1StnSmN )
                                                                    previousStnList_2.append( arrTimeDataStnSmN )

                                                                    # 4.5.3.9.7. 만약 마지막 두 역의 i+1열차 정보가 맞다면, S역의 정보가 틀렸다고 보는게 합리적일것임
                                                                    if ((stnS_dataChkErrCnt >= 2) & (len(arrTrain_ip1_byPreviousStn_2)%2 == 1)) : #& (arrTrain_ip1_byPreviousStn_2[(len(arrTrain_ip1_byPreviousStn_2)-1)] == arrTrain_ip1_byPreviousStn[(len(arrTrain_ip1_byPreviousStn)-2)]) :
                                                                        
                                                                        #arrTrainNo_atStnS_ip1StnS, arrDataGoodStnS_2, useTrain_ip1StnSmN_atStnSmN = findCandidNaTrainNo(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, previousStnList=arrTrain_ip1_byPreviousStn_2)
                                                                        arrTrainNo_atStnS_ip1StnS, feasibleStnId, arrDataGoodStnS_2, useTrain_ip1StnSmN_atStnSmN = findFeasibleTrainId(trainNo_atStnS_ip1StnS=arrTrainNo_atStnS_ip1StnS, initStnId=arrTimeDataStnSmN, nextStnTF=searchingToNextStn, nextOvtStnId=nextOvertakeStnId,  trainNoByPreviousStnList=arrTrain_ip1_byPreviousStn_2, stnIdByPreviousStnList=previousStnList_2, arrTtable=myd3_arr_ttable, depTtable=myd3_dep_ttable)

                                                                    else : 
                                                                        # 4.5.3.9.6. 한개 더 전 역에 대해서 조회할 수 있도록 False 유지
                                                                        #stationCnt = stationCnt - 1
                                                                        arrDataGoodStnS_2 = False

                                                            else : 
                                                                print("S역 누락열차 확인 리스트의 열차 ID 두개가 정보가 서로 다릅니다")
                                                    
                                                    elif (stnS_dataChkErrCnt >= 6) & (len(arrTrain_ip1_byPreviousStn_2) == 1) :
                                                        print("arrTrain_ip1_byPreviousStn_2 가 1개만 남아 후속열차가 맞게 도출되는것으로 추정됩니다. 계속 분석을 진행합니다.")
                                                        arrDataGoodStnS_2 = True

                                                    else : 
                                                        if len(arrTrain_ip1_byPreviousStn_2) == 0 :
                                                            print("arrTrain_ip1_byPreviousStn_2 가 초기화되지 않았습니다")
                                                    
                                                # 무한루프 방지 변수 초기화
                                                stnS_dataChkErrCnt = stnS_dataChkErrCnt + 1
                                            
                                            if arrDataGoodStnS_2 == True :
                                                # 4.5.3.10. s역부터 다음 피추월역까지의 출도착 정보를 조회하여 출도착 정보가 둘 다 있는 경우를 찾아서 확인
                                                stationCnt = 0 # 역 추가검토 관련 변수 초기화
                                                searchingToNextStn = True # 다음역 방향으로 조회하는지 결정하는 변수 초기화
                                                departureTableFlag = True
                                                depDataGoodStnS_2 = False

                                                nextOvertakeStnId = stations[ findNextOvtStnOrder(lineNm=lineNm, bnd=bnd, nextStnTF=searchingToNextStn, currentOvtStnOrder=overtakeStation, ovtStnIdList=overtakeStaList, wholeStnIdList=stations) ]

                                                # 4.5.3.11. 다음 역의 i열차와 i+1열차(정보 누락되서 빠졌던 열차)의 출도착시각 얻기
                                                arrDepDataAvailableRes = findingdepArrTimeAtStaSpN(stationCnt, searchingToNextStn, departureTableFlag, stations, overtakeStaList, overtakeStation, lineNm, bnd, myd3_arr_ttable, myd3_dep_ttable, arrTrainNo_atStnS_ip0StnS, arrTrainNo_atStnSmN_ip1StnSmN, arrTimeTable_stnS, train, trainCnt)

                                                # 4.5.3.12. findingdepArrTimeAtStaSpN 함수 결과 저장
                                                if arrDepDataAvailableRes[0] != False :
                                                    depDataGoodStnS_2 = arrDepDataAvailableRes[0]
                                                    depTrainOrderOf_ip0StnS_atStnSpN = arrDepDataAvailableRes[1]
                                                    depTime_atStnSpN_ip0StnS = arrDepDataAvailableRes[2]
                                                    depTrainOrderOf_ip1StnS_atStnSpN = arrDepDataAvailableRes[3]
                                                    depTime_atStnSpN_ip1StnSpN = arrDepDataAvailableRes[4]
                                                    depTimeTable_stnSpN = arrDepDataAvailableRes[5] 
                                                    depTimeDataStnSpn = arrDepDataAvailableRes[6]

                                                else : # if arrDepDataAvailableRes[0] != False :
                                                    depDataGoodStnS_2 = arrDepDataAvailableRes[0]
                                                    #stnS_dataChkErrCnt = arrDepDataAvailableRes[1]    
                                                
                                                if arrDataGoodStnS_2 == True & depDataGoodStnS_2 == True :
                                                    arrDepBothDataGood_2 = True
                                                else :
                                                    arrDepBothDataGood_2 = False

                                            else :
                                                print("출발역 도착열차 순서의 일관성이 확보되지 않았음 2")
                                                arrDepBothDataGood_2 = False
                                                # =====================
                                                # 이 행을 지나면 다음 역에 대한 출도착 시각에 대해 조회가 끝난 상태임. 선후행열차의 출도착 관련 네 종류 데이터가 다 있는상태임.
                                            

                                            if arrDepBothDataGood_2 == True : # 정상적으로 알고리즘이 끝났으면

                                                # 4.5.11. 출발데이터 상에서의 순서 차이와 S역 도착 데이터 상에서의 분석대상열차 순서 차이 계산
                                                trainArrDepOrderDiff = depTrainOrderOf_ip0StnS_atStnSpN - depTrainOrderOf_ip1StnS_atStnSpN

                                                # 4.5.12. 출발데이터 상에서의 순서 차이와 S역 도착 데이터 상에서의 분석대상열차 순서 차이가 trainCnt보다 작거나 같으면
                                                #         trainCnt + 0 : 선행 도착이 일반, 후착이 급행이어서 출발할 때 역전이 된 경우
                                                #         trainCnt + 2 : 선행 출발이 급행, 후착이 일반인데 선행 급행은 그 앞의 일반을 추월했고, 후착 열차가 그 뒤에 온 급행열차에게 추월 당해 총 3대 차이가 나는 상황
                                                if (abs(trainArrDepOrderDiff) <= (trainCnt)) | (abs(trainArrDepOrderDiff) == (trainCnt+2)) : #(abs(trainArrDepOrderDiff) == trainCnt) | (abs(trainArrDepOrderDiff) == (trainCnt+2)) :
                                                    
                                                    # 4.5.12.1. 피추월횟수 계산
                                                    numOfOvertake = analyzingOvertake(trainArrDepOrderDiff, arrTrainNo_atStnS_ip0StnS, arrTime_atStnS_ip0StnS, depTime_atStnSpN_ip0StnS, arrTrainNo_atStnS_ip1StnS, arrTime_atStnS_ip1StnS, depTime_atStnSpN_ip1StnSpN) 
                                                
                                                    # 4.5.12.2. 피추월횟수 계산 결과 검토 및 추월당한 경우 추월한 열차정보 기재. 
                                                    #         updatingNumOfOvertake 함수의 결과로서 계속 피추월횟수를 계산해야하는지에 대한 
                                                    #         True(피추월횟수 분석 종료), False(피추월횟수 추가 분석 필요)값을 제공함
                                                    updatingNumOfOvertakeRes = updatingNumOfOvertake(statnId=overtakeStaList[overtakeStation], nOvertakeTable=numOvertakeTable, targetTrainNo=arrTrainNo_atStnS_ip0StnS, numberOfOvertakeFreq=numOfOvertake, overtakeTrnNoTable=overtakeTrainNoTable, overtakingTrainNo=arrTrainNo_atStnSmN_ip1StnSmN)
                                                    
                                                    endOvertakingAnalysis = updatingNumOfOvertakeRes[0] # 피추월횟수가 0이면 True, >=1 이면 False : 1회 이상 추월되는 경우를 확인하기 위해 추월이 안일어나는것을 확인할 때 까지 반복시행함
                                                    
                                                    numOvertakeTable = updatingNumOfOvertakeRes[1]
                                                    overtakeTrainNoTable = updatingNumOfOvertakeRes[2]
                                                    #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )

                                                    #if numOfOvertake > 0 :
                                                    #    print("오류확인! 이거 가능한건지 확인 필요")

                                                    # 4.5.12.3. S역 i번째 도착열차 분석에 대한 임시 Sequence update, 추월열차정보 update
                                                    tempArrSeq, tempDepSeq, overtakedTrnIdList = updatingTempArrDepSeq(funcTempArrSeq=tempArrSeq, funcTempDepSeq=tempDepSeq, funcTrainNo_ip0=arrTrainNo_atStnS_ip0StnS, funcTrainNo_ip1=arrTrainNo_atStnSmN_ip1StnSmN, numOfOvertake=numOfOvertake, funcOvertakedTrnIdList=overtakedTrnIdList)

                                                    # 4.5.9.4. 만약 현재 분석대상열차가 누락열차였고, 그 열차가 추월당한것으로 나타나면 
                                                    if (numOfOvertake == 1) & (natTF_ip0StnS == True) :
                                                        
                                                        # 다음 조회에서 다시 현재 분석대상열차로부터 시작할 수 있도록 nextTrainByForce 로 열차ID 지정
                                                        nextTrainByForce = arrTrainNo_atStnS_ip0StnS

                                                else : # if (abs(trainArrDepOrderDiff) <= (trainCnt)) | (abs(trainArrDepOrderDiff) == (trainCnt+2)) : #(abs(trainArrDepOrderDiff) == trainCnt) | (abs(trainArrDepOrderDiff) == (trainCnt+2)) :
                                                    if depTrainOrderOf_ip0StnS_atStnSpN < depTrainOrderOf_ip1StnS_atStnSpN : # S역의 i열차가 SpN역에서 더 먼저 출발한 경우 : 추월이 일어나지 않은 경우

                                                        # 사이에 낀 열차정보 저장
                                                        depTrainNo_atStnS_btw_ip0_ip1 = depTimeTable_stnSpN.index[depTrainOrderOf_ip0StnS_atStnSpN+1:depTrainOrderOf_ip1StnS_atStnSpN]
                                                        if len(depTrainNo_atStnS_btw_ip0_ip1) >= 1 :
                                                            
                                                            for btwTrainCnt in range(0,len(depTrainNo_atStnS_btw_ip0_ip1)) :
                                                                # 사이에 낀 열차가 피추월횟수 table에서 몇번째 있는지 확인
                                                                numOvertakeDfOdrer_depTrainOrder_atStnS_btw_ip0_ip1 = np.where( numOvertakeTable.index == depTrainNo_atStnS_btw_ip0_ip1[btwTrainCnt] )[0][0]

                                                                # 사이에 낀 열차의 피추월횟수 확인  
                                                                if numOvertakeTable[str(overtakeStnId)].iloc[numOvertakeDfOdrer_depTrainOrder_atStnS_btw_ip0_ip1] == 1 : # 사이에 낀 열차의 피추월횟수가 1인 경우
                                                                    overtakedTrainNo = overtakeTrainNoTable[str(overtakeStnId)].iloc[numOvertakeDfOdrer_depTrainOrder_atStnS_btw_ip0_ip1]
                                                                    
                                                                    # 추월당한 열차 ID가 현재 분석대상인 S역 i열차가 맞는지 확인
                                                                    if overtakedTrainNo == str(arrTrainNo_atStnS_ip0StnS) :
                                                                        # 4.5.12.1. 피추월횟수 계산 : 아마 
                                                                        numOfOvertake = analyzingOvertake(trainArrDepOrderDiff, arrTrainNo_atStnS_ip0StnS, arrTime_atStnS_ip0StnS, depTime_atStnSpN_ip0StnS, arrTrainNo_atStnS_ip1StnS, arrTime_atStnS_ip1StnS, depTime_atStnSpN_ip1StnSpN) 
                                                                    
                                                                        # 4.5.12.2. 피추월횟수 계산 결과 검토 및 추월당한 경우 추월한 열차정보 기재. 
                                                                        #         updatingNumOfOvertake 함수의 결과로서 계속 피추월횟수를 계산해야하는지에 대한 
                                                                        #         True(피추월횟수 분석 종료), False(피추월횟수 추가 분석 필요)값을 제공함
                                                                        updatingNumOfOvertakeRes = updatingNumOfOvertake(statnId=overtakeStaList[overtakeStation], nOvertakeTable=numOvertakeTable, targetTrainNo=arrTrainNo_atStnS_ip0StnS, numberOfOvertakeFreq=numOfOvertake, overtakeTrnNoTable=overtakeTrainNoTable, overtakingTrainNo=arrTrainNo_atStnSmN_ip1StnSmN)
                                                                        
                                                                        endOvertakingAnalysis = updatingNumOfOvertakeRes[0] # 피추월횟수가 0이면 True, >=1 이면 False : 1회 이상 추월되는 경우를 확인하기 위해 추월이 안일어나는것을 확인할 때 까지 반복시행함
                                                                        
                                                                        numOvertakeTable = updatingNumOfOvertakeRes[1]
                                                                        overtakeTrainNoTable = updatingNumOfOvertakeRes[2]

                                                                        # S역 i번째 도착열차 분석에 대한 임시 Sequence update, 추월열차정보 update
                                                                        tempArrSeq, tempDepSeq, overtakedTrnIdList = updatingTempArrDepSeq(funcTempArrSeq=tempArrSeq, funcTempDepSeq=tempDepSeq, funcTrainNo_ip0=arrTrainNo_atStnS_ip0StnS, funcTrainNo_ip1=arrTrainNo_atStnS_ip1StnS, numOfOvertake=numOfOvertake, funcOvertakedTrnIdList=overtakedTrnIdList)
                                                                    
                                                                    else : # 
                                                                        print("논리확인 필요 : 추월한 열차 ID가 분석대상인 S역 i열차가 아님. 추월한열차: "+str(overtakedTrainNo))
                                                                
                                                                # 사이에 낀 열차의 피추월횟수가 1보다 큰 경우 : 두번째 추월한 열차에 대한 분석 시 이 단계에 도달할 수 있음
                                                                elif numOvertakeTable[str(overtakeStnId)].iloc[numOvertakeDfOdrer_depTrainOrder_atStnS_btw_ip0_ip1] > 1 : 
                                                                    # i번째 도착열차와 i+1번째 도착열차 사이에 출발한 열차를 마지막으로 추월한 열차번호 확인하기
                                                                    lastOvertakingTrainNo = overtakeTrainNoTable[str(overtakeStnId)].iloc[numOvertakeDfOdrer_depTrainOrder_atStnS_btw_ip0_ip1]
                                                                    
                                                                    # i번째 도착열차와 i+1번째 도착열차 사이에 출발한 열차를 마지막으로 추월한 열차가 현재 분석대상 열차인지 확인하기
                                                                    if str(arrTrainNo_atStnS_ip0StnS) != str(lastOvertakingTrainNo) :
                                                                        print("논리확인 필요 : 사이에 낀 열차의 피추월횟수가 1 이상임")
                                                                    
                                                                    else : 
                                                                        # 4.5.12.1. 피추월횟수 계산 : 아마 
                                                                        numOfOvertake = analyzingOvertake(trainArrDepOrderDiff, arrTrainNo_atStnS_ip0StnS, arrTime_atStnS_ip0StnS, depTime_atStnSpN_ip0StnS, arrTrainNo_atStnS_ip1StnS, arrTime_atStnS_ip1StnS, depTime_atStnSpN_ip1StnSpN) 
                                                                    
                                                                        # 4.5.12.2. 피추월횟수 계산 결과 검토 및 추월당한 경우 추월한 열차정보 기재. 
                                                                        #         updatingNumOfOvertake 함수의 결과로서 계속 피추월횟수를 계산해야하는지에 대한 
                                                                        #         True(피추월횟수 분석 종료), False(피추월횟수 추가 분석 필요)값을 제공함
                                                                        updatingNumOfOvertakeRes = updatingNumOfOvertake(statnId=overtakeStaList[overtakeStation], nOvertakeTable=numOvertakeTable, targetTrainNo=arrTrainNo_atStnS_ip0StnS, numberOfOvertakeFreq=numOfOvertake, overtakeTrnNoTable=overtakeTrainNoTable, overtakingTrainNo=arrTrainNo_atStnSmN_ip1StnSmN)
                                                                        
                                                                        endOvertakingAnalysis = updatingNumOfOvertakeRes[0] # 피추월횟수가 0이면 True, >=1 이면 False : 1회 이상 추월되는 경우를 확인하기 위해 추월이 안일어나는것을 확인할 때 까지 반복시행함
                                                                        
                                                                        numOvertakeTable = updatingNumOfOvertakeRes[1]
                                                                        overtakeTrainNoTable = updatingNumOfOvertakeRes[2]

                                                                        # S역 i번째 도착열차 분석에 대한 임시 Sequence update, 추월열차정보 update
                                                                        tempArrSeq, tempDepSeq, overtakedTrnIdList = updatingTempArrDepSeq(funcTempArrSeq=tempArrSeq, funcTempDepSeq=tempDepSeq, funcTrainNo_ip0=arrTrainNo_atStnS_ip0StnS, funcTrainNo_ip1=arrTrainNo_atStnS_ip1StnS, numOfOvertake=numOfOvertake, funcOvertakedTrnIdList=overtakedTrnIdList)

                                                                
                                                                elif numOvertakeTable[str(overtakeStnId)].iloc[numOvertakeDfOdrer_depTrainOrder_atStnS_btw_ip0_ip1] == 0 :
                                                                    print("논리확인 필요 : 사이에 낀 열차의 피추월횟수가 0 임")

                                                                else : # if numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[numOvertakeDfOdrer_depTrainOrder_atStnS_btw_ip0_ip1] == 1 

                                                                    # S역의 i+2열차의 순서 저장                                                                  
                                                                    tempArrOrder_ip2StnS_atStnS = np.where(arrTimeTable_stnS.index==arrTrainNo_atStnS_ip1StnS)[0][0]+1

                                                                    # S역의 i번째 도착열차와 i+1번째 도착열차 사이에 출발한 열차가 i+2번째 도착한 열차인 경우 : 즉, i+1번째 도착한 열차를 i+2번째 도착열차가 추월해서 출발한 경우
                                                                    if ( str(depTrainNo_atStnS_btw_ip0_ip1[btwTrainCnt]) == str(arrTimeTable_stnS.index[(tempArrOrder_ip2StnS_atStnS+btwTrainCnt)]) ) :  # 전역 도착순서 사이에 낀 열차가 S역에서 현재 조회중인 i+1열차의 다음차(i+2)열차인 경우
                                                                        # 4.5.12.1. 피추월횟수 계산
                                                                        numOfOvertake = analyzingOvertake(trainArrDepOrderDiff, arrTrainNo_atStnS_ip0StnS, arrTime_atStnS_ip0StnS, depTime_atStnSpN_ip0StnS, arrTrainNo_atStnS_ip1StnS, arrTime_atStnS_ip1StnS, depTime_atStnSpN_ip1StnSpN) 
                                                                    
                                                                        # 4.5.12.2. 피추월횟수 계산 결과 검토 및 추월당한 경우 추월한 열차정보 기재. 
                                                                        #         updatingNumOfOvertake 함수의 결과로서 계속 피추월횟수를 계산해야하는지에 대한 
                                                                        #         True(피추월횟수 분석 종료), False(피추월횟수 추가 분석 필요)값을 제공함
                                                                        updatingNumOfOvertakeRes = updatingNumOfOvertake(statnId=overtakeStaList[overtakeStation], nOvertakeTable=numOvertakeTable, targetTrainNo=arrTrainNo_atStnS_ip0StnS, numberOfOvertakeFreq=numOfOvertake, overtakeTrnNoTable=overtakeTrainNoTable, overtakingTrainNo=arrTrainNo_atStnSmN_ip1StnSmN)
                                                                        
                                                                        endOvertakingAnalysis = updatingNumOfOvertakeRes[0] # 피추월횟수가 0이면 True, >=1 이면 False : 1회 이상 추월되는 경우를 확인하기 위해 추월이 안일어나는것을 확인할 때 까지 반복시행함
                                                                        
                                                                        numOvertakeTable = updatingNumOfOvertakeRes[1]
                                                                        overtakeTrainNoTable = updatingNumOfOvertakeRes[2]
                                                                        
                                                                        # S역 i번째 도착열차 분석에 대한 임시 Sequence update, 추월열차정보 update
                                                                        tempArrSeq, tempDepSeq, overtakedTrnIdList = updatingTempArrDepSeq(funcTempArrSeq=tempArrSeq, funcTempDepSeq=tempDepSeq, funcTrainNo_ip0=arrTrainNo_atStnS_ip0StnS, funcTrainNo_ip1=arrTrainNo_atStnS_ip1StnS, numOfOvertake=numOfOvertake, funcOvertakedTrnIdList=overtakedTrnIdList)

                                                                    else : # if ( str(depTrainNo_atStnS_btw_ip0_ip1[btwTrainCnt]) == str(arrTimeTable_stnS.index[(tempArrOrder_ip2StnS_atStnS+btwTrainCnt)]) ) :
                                                                        # 만약 사이에 낀 열차가 S역에서 정보가 누락되어있는 상태라면, 추월을 안한 것은 확실하므로 단순히 도착정보누락으로 조회가 안된던 것으로 보고 피추월횟수를 계산함
                                                                        if arrTimeTable_stnS.iloc[np.where( arrTimeTable_stnS.index == depTrainNo_atStnS_btw_ip0_ip1[btwTrainCnt] )[0][0]][0] is pd.NaT : 

                                                                            # 4.5.12.1. 피추월횟수 계산
                                                                            numOfOvertake = analyzingOvertake(trainArrDepOrderDiff, arrTrainNo_atStnS_ip0StnS, arrTime_atStnS_ip0StnS, depTime_atStnSpN_ip0StnS, arrTrainNo_atStnS_ip1StnS, arrTime_atStnS_ip1StnS, depTime_atStnSpN_ip1StnSpN) 
                                                                        
                                                                            # 4.5.12.2. 피추월횟수 계산 결과 검토 및 추월당한 경우 추월한 열차정보 기재. 
                                                                            #         updatingNumOfOvertake 함수의 결과로서 계속 피추월횟수를 계산해야하는지에 대한 
                                                                            #         True(피추월횟수 분석 종료), False(피추월횟수 추가 분석 필요)값을 제공함
                                                                            updatingNumOfOvertakeRes = updatingNumOfOvertake(statnId=overtakeStaList[overtakeStation], nOvertakeTable=numOvertakeTable, targetTrainNo=arrTrainNo_atStnS_ip0StnS, numberOfOvertakeFreq=numOfOvertake, overtakeTrnNoTable=overtakeTrainNoTable, overtakingTrainNo=arrTrainNo_atStnSmN_ip1StnSmN)
                                                                            
                                                                            endOvertakingAnalysis = updatingNumOfOvertakeRes[0] # 피추월횟수가 0이면 True, >=1 이면 False : 1회 이상 추월되는 경우를 확인하기 위해 추월이 안일어나는것을 확인할 때 까지 반복시행함
                                                                            
                                                                            numOvertakeTable = updatingNumOfOvertakeRes[1]
                                                                            overtakeTrainNoTable = updatingNumOfOvertakeRes[2]
                                                                            
                                                                            # S역 i번째 도착열차 분석에 대한 임시 Sequence update, 추월열차정보 update
                                                                            tempArrSeq, tempDepSeq, overtakedTrnIdList = updatingTempArrDepSeq(funcTempArrSeq=tempArrSeq, funcTempDepSeq=tempDepSeq, funcTrainNo_ip0=arrTrainNo_atStnS_ip0StnS, funcTrainNo_ip1=arrTrainNo_atStnS_ip1StnS, numOfOvertake=numOfOvertake, funcOvertakedTrnIdList=overtakedTrnIdList)

                                                                        else :
                                                                            print("사이에 낀 "+str(depTrainNo_atStnS_btw_ip0_ip1[btwTrainCnt])+"열차는 "+"S역("+str(overtakeStnId)+")의 i+2 열차도 아니고 피추월횟수가 초기화되어있지 않음. ")
                                                                            if (btwTrainCnt == (len(depTrainNo_atStnS_btw_ip0_ip1)-1)) :
                                                                                print("사이에 낀 모든 열차는 "+"S역("+str(overtakeStnId)+")의 i+2 열차도 아니고 피추월횟수가 초기화되어있지 않음. ")
                                                                                print(numOvertakeTable[str(overtakeStnId)].iloc[numOvertakeDfOdrer_depTrainOrder_atStnS_btw_ip0_ip1])

                                                        else : # if len(depTrainNo_atStnS_btw_ip0_ip1) == 1 :
                                                            print("논리오류 : S역의 i열차와 S역의 i+1 열차 사이에 출발한 열차가 있어야 하는데 없거나 1대 이상임.")  
                                                        
                                                    else : # depTrainOrderOf_ip0StnS_atStnSpN < depTrainOrderOf_ip1StnS_atStnSpN : # S역의 i열차가 SpN역에서 더 늦게 출발한 경우 : 추월이 일어난 경우
                                                           # 특히, 앞서 (abs(trainArrDepOrderDiff) <= (trainCnt)) | (abs(trainArrDepOrderDiff) == (trainCnt+2)) 조건에서 False가 난 경우에 이 부분으로 오므로
                                                           # 추월이 일어났는데 연속 2대 이상의 추월이 발생한 경우에 이 부분으로 들어오게 됨
                                                        
                                                        # 사이에 낀 열차정보 저장
                                                        depTrainNo_atStnS_btw_ip0_ip1 = depTimeTable_stnSpN.index[depTrainOrderOf_ip1StnS_atStnSpN+1:depTrainOrderOf_ip0StnS_atStnSpN]

                                                        if len(depTrainNo_atStnS_btw_ip0_ip1) == 1 :
                                                            
                                                            # S역의 i+2열차의 도착 순서 및 열차Id 저장                                                               
                                                            tempArrOrder_ip2StnS_atStnS = np.where(arrTimeTable_stnS.index==arrTrainNo_atStnS_ip1StnS)[0][0]+1
                                                            
                                                            for btwTrainCnt in range(0,len(depTrainNo_atStnS_btw_ip0_ip1)) :
                                                                # 사이에 낀 열차가 피추월횟수 table에서 몇번째 있는지 확인
                                                                #depTrainNo_atStnS_btw_ip0_ip1[btwTrainCnt]

                                                                # S역의 i번째 도착열차와 i+1번째 도착열차 사이에 출발한 열차가 i+2+btwTrainCnt번째 도착한 열차인 경우 : 즉, i번째 도착한 열차가 i+1번째 도착한 열차 뿐만 아니라 i+2번째 도착열차에게도 추월당한 경우
                                                                if ( str(depTrainNo_atStnS_btw_ip0_ip1[btwTrainCnt]) == str(arrTimeTable_stnS.index[(tempArrOrder_ip2StnS_atStnS+btwTrainCnt)]) ) :  # 전역 도착순서 사이에 낀 열차가 S역에서 현재 조회중인 i+1열차의 다음차(i+2)열차인 경우

                                                                    # 피추월횟수 계산
                                                                    numOfOvertake = analyzingOvertake(trainArrDepOrderDiff, arrTrainNo_atStnS_ip0StnS, arrTime_atStnS_ip0StnS, depTime_atStnSpN_ip0StnS, arrTrainNo_atStnS_ip1StnS, arrTime_atStnS_ip1StnS, depTime_atStnSpN_ip1StnSpN) 
                                                                
                                                                    # 피추월횟수 계산 결과 검토 및 추월당한 경우 추월한 열차정보 기재. 
                                                                    #         updatingNumOfOvertake 함수의 결과로서 계속 피추월횟수를 계산해야하는지에 대한 
                                                                    #         True(피추월횟수 분석 종료), False(피추월횟수 추가 분석 필요)값을 제공함
                                                                    updatingNumOfOvertakeRes = updatingNumOfOvertake(statnId=overtakeStaList[overtakeStation], nOvertakeTable=numOvertakeTable, targetTrainNo=arrTrainNo_atStnS_ip0StnS, numberOfOvertakeFreq=numOfOvertake, overtakeTrnNoTable=overtakeTrainNoTable, overtakingTrainNo=arrTrainNo_atStnSmN_ip1StnSmN)
                                                                    
                                                                    endOvertakingAnalysis = updatingNumOfOvertakeRes[0] # 피추월횟수가 0이면 True, >=1 이면 False : 1회 이상 추월되는 경우를 확인하기 위해 추월이 안일어나는것을 확인할 때 까지 반복시행함
                                                                    
                                                                    numOvertakeTable = updatingNumOfOvertakeRes[1]
                                                                    overtakeTrainNoTable = updatingNumOfOvertakeRes[2]
                                                                    
                                                                    # S역 i번째 도착열차 분석에 대한 임시 Sequence update, 추월열차정보 update
                                                                    tempArrSeq, tempDepSeq, overtakedTrnIdList = updatingTempArrDepSeq(funcTempArrSeq=tempArrSeq, funcTempDepSeq=tempDepSeq, funcTrainNo_ip0=arrTrainNo_atStnS_ip0StnS, funcTrainNo_ip1=arrTrainNo_atStnS_ip1StnS, numOfOvertake=numOfOvertake, funcOvertakedTrnIdList=overtakedTrnIdList)

                                                        
                                                        elif len(depTrainNo_atStnS_btw_ip0_ip1) > 1 :
                                                            if (depTime_atStnSpN_ip0StnS is pd.NaT) | (depTime_atStnSpN_ip1StnSpN is pd.NaT) :
                                                                print("열차정보 중 하나의 시각정보가 없습니다.")
                                                            
                                                            else :
                                                                print("논리오류 : S역의 i열차가 3대 이상 추월당했는지 확인해야 합니다. i열차를 추월한것으로 추정되는 열차목록:"+str(depTrainNo_atStnS_btw_ip0_ip1))

                                                        else :
                                                            print("논리오류 : 사이에 낀 열차가 있어야만 하는데 없다고 나왔습니다.")


                                            else : # if arrDepDataAvailableFlag == True : 
                                                # 해당 열차에는 유효한 데이터가 없으므로, 누락 열차 확인에 사용한 리스트를 초기화시킴
                                                arrTrain_ip1_byPreviousStn = [] # 출발역 도착데이터 누락 열차번호 저장용 리스트 초기화
                                                previousStnList = [] # 출발역 도착데이터 누락 열차 발생 역 ID  저장용 리스트 초기화
                                                trainCnt = trainCnt + 1
                                                
                                                #print("Error!\n")
                                                if numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS )[0][0] ] == -99 :    
                                                    #print("There's no arr or dep time information of train i and i+1. train_i: "+str(arrTrainNo_atStnS_ip0StnS)+",  train_i+1: "+str(arrTrainNo_atStnS_ip1StnS) )
                                                    numOvertakeTable[str(overtakeStaList[overtakeStation])][arrTrainNo_atStnS_ip0StnS] = np.nan
                                                    endOvertakingAnalysis = True
                                                    #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                                
                                                elif np.isnan(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS )[0][0] ]) :    
                                                    #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                                    endOvertakingAnalysis = True

                                                elif numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS )[0][0] ] == 0 :    
                                                        print("이미 초기화되었던 이력이 있음 3")  
                                                        endOvertakingAnalysis = True 
                                                
                                                else :
                                                    #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                                    print("here 13 \n")

                                        # 4.5.13. 출발 데이터상에서 열차 ID가 3 이상 차이가 나는 경우 : 출발역 데이터의 누락으로 인해 도착역 데이터랑 안맞는 경우임 - 추후 업데이트 고민해보기
                                        else : 
                                            if numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS )[0][0] ] == -99 :    
                                                #print("There's no arr or dep time information of train i and i+1. train_i: "+str(arrTrainNo_atStnS_ip0StnS)+",  train_i+1: "+str(arrTrainNo_atStnS_ip1StnS) )
                                                numOvertakeTable[str(overtakeStaList[overtakeStation])][arrTrainNo_atStnS_ip0StnS] = np.nan
                                                endOvertakingAnalysis = True
                                                #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                            
                                            elif np.isnan(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS )[0][0] ]) :    
                                                #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                                endOvertakingAnalysis = True

                                            else :
                                                print("abs(trainArrDepOrderDiff): "+str(abs(trainArrDepOrderDiff))+"\n")
                                                print("\n많이 차이난다는디?\n")
                                                endOvertakingAnalysis = True
                                        
                                        
                                        # 4.5.12. 추월을 당한 경우 다음 열차에 대해 검색할 수 있도록 trainCnt 변수를 1 증가시킴
                                        if useTrain_ip1StnSmN_atStnSmN == False : 
                                            trainCnt = trainCnt + 1 
                                            #useTrain_ip1StnSmN_atStnSmN = False

                                    else : # if arrDepDataAvailableFlag == True : 
                                        # 해당 열차에는 유효한 데이터가 없으므로, 누락 열차 확인에 사용한 리스트를 초기화시킴
                                        arrTrain_ip1_byPreviousStn = [] # 출발역 도착데이터 누락 열차번호 저장용 리스트 초기화
                                        previousStnList = [] # 출발역 도착데이터 누락 열차 발생 역 ID  저장용 리스트 초기화
                                        trainCnt = trainCnt + 1
                                        
                                        #print("Error!\n")
                                        if numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS )[0][0] ] == -99 :    
                                            #print("There's no arr or dep time information of train i and i+1. train_i: "+str(arrTrainNo_atStnS_ip0StnS)+",  train_i+1: "+str(arrTrainNo_atStnS_ip1StnS) )
                                            numOvertakeTable[str(overtakeStaList[overtakeStation])][arrTrainNo_atStnS_ip0StnS] = np.nan
                                            endOvertakingAnalysis = True
                                            #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                        
                                        elif np.isnan(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS )[0][0] ]) :    
                                            #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                            endOvertakingAnalysis = True

                                        elif numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS )[0][0] ] == 0 :    
                                                print("이미 초기화되었던 이력이 있음 4")  
                                                endOvertakingAnalysis = True 
                                        
                                        else :
                                            #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                            print("here 13 \n")

                                    errCntNextTrain = errCntNextTrain + 1
                                
                                if endOvertakingAnalysis == True : # while ( ((train + trainCnt) < len(arrTimeTable_stnS.index)) & (trainCnt < 5) & (endOvertakingAnalysis == False) ) & (errCntNextTrain < 10) :
                                    #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                    #print("Works Well \n")
                                    
                                    if (len(tempArrSeq)>0) & (len(tempDepSeq)>0) :
                                        arrSeq = appendArrDepTrainSeq(tempSeq=tempArrSeq, targetSeqList=arrSeq, analysisTrainArrTime=arrTime_atStnS_ip0StnS) # 도착순서 리스트 병합
                                        depSeq = appendArrDepTrainSeq(tempSeq=tempDepSeq, targetSeqList=depSeq, analysisTrainArrTime=arrTime_atStnS_ip0StnS) # 출발순서 리스트 병합
                                    
                                        tempArrSeq = [] # while문 용 도착순서 리스트 초기화
                                        tempDepSeq = [] # while문 용 도착순서 리스트 초기화
                                    

                                    if (len(arrTrain_ip1_byPreviousStn) >= 2) : 
                                        if (arrTrain_ip1_byPreviousStn[(len(arrTrain_ip1_byPreviousStn)-1)] != arrTrain_ip1_byPreviousStn[(len(arrTrain_ip1_byPreviousStn)-2)])  :
                                            arrTrain_ip1_byPreviousStn = [] # 출발역 도착데이터 누락 열차번호 저장용 리스트 초기화
                                            previousStnList = [] # 출발역 도착데이터 누락 열차 발생 역 ID  저장용 리스트 초기화 

                                else : 
                                    if numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS )[0][0] ] == -99 :    
                                        #print("There's no arr or dep time information of train i and i+1. train_i: "+str(arrTrainNo_atStnS_ip0StnS)+",  train_i+1: "+str(arrTrainNo_atStnS_ip1StnS) )
                                        numOvertakeTable[str(overtakeStaList[overtakeStation])][arrTrainNo_atStnS_ip0StnS] = np.nan
                                        endOvertakingAnalysis = True
                                        #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                    
                                    elif np.isnan(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS )[0][0] ]) :    
                                        #print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                        endOvertakingAnalysis = True
                                    
                                    else :
                                        print("\n"+str(numOvertakeTable[str(overtakeStaList[overtakeStation])].iloc[ np.where( numOvertakeTable.index == arrTrainNo_atStnS_ip0StnS ) ]) + "\n" )
                                        print("here 14 \n")
                                
                                train = train + 1 # 열차 조회변수 1 증가시키기
                            
                            # 역 별 출도착순서 리스트 병합. 
                            arrSeqByStn = pd.concat([arrSeqByStn, df({str(overtakeStaList[overtakeStation]):arrSeq})], axis=1)
                            depSeqByStn = pd.concat([depSeqByStn, df({str(overtakeStaList[overtakeStation]):depSeq})], axis=1) #depSeqByStn.append(depSeq)

                        arrSeqByStn.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-rtPos_arrSeqByOvtStn-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)
                        depSeqByStn.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-rtPos_depSeqByOvtStn-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)

                        
                        if (lineNm == "Line9") & (bnd == 1) :
                            ## ============ 준희 작성 코드 시작 ==========
                            
                            ## arrSeqByOvertakingStn과 depSeqByOvertakingStn 불러오기
                            arrSeqByOvertakingStn = arrSeqByStn
                            depSeqByOvertakingStn = depSeqByStn


                            ## 급행/완행이 역전되는 역의 최종 도착/출발 시각표 저장 list 생성
                            resultInformation1 = []#pd.DataFrame()

                            ## 0905역의 도착 정보 추가하기
                            resultInformation1.append(pd.Series(arrSeqByOvertakingStn.iloc[:,0],name=arrSeqByOvertakingStn.columns[0]+"-arr"))

                            ## arrSeqByOvertakingStn의 i+1번째 역과 depSeqByOvertakingStn의 i번째 역의 도착역 순서 비교하기
                            for i in range(len(arrSeqByOvertakingStn.iloc[1])-1):

                                ## depSeqByOvertakingStn의 i번째 역을 출발한 열차 정보 불러오기
                                tmpDepInformation = list(depSeqByOvertakingStn.iloc[:,i])

                                ## arrSeqByOvertakingStn의 i+1번째 역에 도착한 열차 정보 불러오기
                                tmpArrInformation = list(arrSeqByOvertakingStn.iloc[:,i+1])

                                j=0 # 각 역에서 j번째로 기록된 열차 ID 불러오기
                                while True:
                                    ## 만약 j의 길이가 list의 길이만큼 되면, while문 정지
                                    if j==min(len(tmpDepInformation),len(tmpArrInformation)): break
                                    ## 만약 한 쪽에서 NA가 나오면 while문 정지
                                    if str(tmpDepInformation[j])=="nan" or str(tmpArrInformation[j])=="nan": break

                                    ## tmpDepInformation j번째 열차와 arrSeqByOvertakingStn j번째 열차가 다른지 확인
                                    if tmpDepInformation[j]!=tmpArrInformation[j]: # 만약 다르다면,
                                        k=1 # j번째 열차에서 k번 뒤의 열차 번호 확인하기
                                        while True:
                                            ## 만약, 일치하지 않는 역 번호가 tmpDepInformation와 tmpArrInformation에 대하여 서로 존재하지 않는 경우, 매칭을 할 수 없으므로 건너뛰기
                                            if j+k==min(len(tmpDepInformation),len(tmpArrInformation)):
                                                j += 1
                                                break
                                            ## tmpDepInformation에서 k번째 뒤의 열차번호가 동일한 경우
                                            if tmpDepInformation[j+k]==tmpArrInformation[j]:
                                                ## j==0인 경우에는 nan 삽입하기
                                                if j==0:
                                                    for n in range(k): tmpArrInformation.insert(j+n,np.nan)
                                                    j = j+k
                                                    break
                                                ## tmpArrInformation에 j번째부터 j+k-1번째까지 tmpDepInformation에 기록된 열차 정보 삽입
                                                for n in range(k): tmpArrInformation.insert(j+n,tmpDepInformation[j+n])

                                                break
                                            ## tmpArrInformation에서 k번째 뒤의 열차번호가 동일한 경우
                                            elif tmpDepInformation[j]==tmpArrInformation[j+k]:
                                                ## j==0인 경우에는 nan 삽입하기
                                                if j==0:
                                                    for n in range(k): tmpDepInformation.insert(j+n,np.nan)
                                                    j = j+k
                                                    break
                                                ## tmpDepInformation에 j번째부터 j+k-1번째까지 tmpArrInformation에 기록된 열차 정보 삽입
                                                for n in range(k): tmpDepInformation.insert(j+n,tmpArrInformation[j+n])

                                                break
                                            else: k += 1

                                    ## 열차 번호가 같은 경우, 다음 열차로 분석 진행
                                    else: j += 1

                                resultInformation1.append(pd.Series(tmpDepInformation,name=depSeqByOvertakingStn.iloc[:,i].name+"-dep"))
                                resultInformation1.append(pd.Series(tmpArrInformation,name=arrSeqByOvertakingStn.iloc[:,i+1].name+"-arr"))

                            ## 0934역의 출발 정보 추가하기
                            resultInformation1.append(pd.Series(depSeqByOvertakingStn.iloc[:,-1],name=depSeqByOvertakingStn.columns[-1]+"-dep"))

                            ## resultInformation1 list를 data frame으로 변환하기
                            resultInformation1 = pd.DataFrame(resultInformation1).transpose()
                            #print(resultInformation1.columns)

                            ## resultInformation1의 정보를 csv로 저장
                            resultInformation1.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-rtPos_arrDepSeqByOvtStn-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)

                            # 2. 역별 누락된 열차 정보를 정리하여 csv 파일로 저장하기
                            #    역별로 열차 순서를 비교하였을 때, 없는 열차번호를 저장하는 list 생성
                            errorInformation1 = []#pd.DataFrame()

                            for i in range(len(resultInformation1.columns)):
                                ## 열차 번호 검증작업
                                ### 현재 열차 번호에 +2한 값이 이후 같은 등급 열차 번호와 같은지를 확인
                                ### 만약 같으면 이상 없음, 다르면 +2한 값은 없는 정보이므로 error에 저장 후 다음 열차 번호로 넘어가기

                                ## 역별 열차 도착 정보를 tmpInformation에 저장하기
                                tmpInformation = resultInformation1.iloc[:,i]

                                ## 역별 누락 열차 정보를 입력할 list 생성
                                errorTrainIDList = []

                                for j in range(len(tmpInformation)):
                                    ## 2.1. 완행 열차의 경우: 열차 번호가 9400 미만이다
                                    if tmpInformation[j] < 9400:
                                        ## tmpTrainID 이후에 나오는 완행열차의 ID가 tmpTrainID+2여야 한다.
                                        tmpTrainID = tmpInformation[j]

                                        ## 이후에 처음 나오는 완행열차 ID를 compTrainID에 저장
                                        k = 1
                                        while True:
                                            if j+k >= len(tmpInformation)-1 or tmpInformation[j+k] < 9400: break
                                            else: k += 1
                                        compTrainID = tmpInformation[j+k]

                                        ## tmpTrainID와 compTrainID가 같은지 확인해서 같으면 통과
                                        if j+k >= len(tmpInformation)-1 or tmpTrainID+2 == compTrainID: continue

                                        ## tmpTrainID와 compTrainID가 다르면 tmpTrainID+2부터 compTrainID 사이에 없는 ID를 errorInformation에 저장하기
                                        else:
                                            tmpErr = list(range(int(tmpTrainID+2),int(compTrainID),2))
                                            errorTrainIDList.append(tmpErr)


                                    ## 2.2. 주말 공휴일 완행 열차의 경우: 열차 번호가 9500 미만이다
                                    elif tmpInformation[j] >= 9400 and tmpInformation[j] < 9500:
                                        ## tmpTrainID 이후에 나오는 완행열차의 ID가 tmpTrainID+2여야 한다.
                                        tmpTrainID = tmpInformation[j]

                                        ## 이후에 처음 나오는 완행열차 ID를 compTrainID에 저장
                                        k = 1
                                        while True:
                                            if j+k >= len(tmpInformation)-1 or (tmpInformation[j+k] >= 9400 and tmpInformation[j+k] < 9500): break
                                            else: k += 1
                                        compTrainID = tmpInformation[j+k]

                                        ## tmpTrainID와 compTrainID가 같은지 확인해서 같으면 통과
                                        if j+k >= len(tmpInformation)-1 or tmpTrainID+2 == compTrainID: continue

                                        ## tmpTrainID와 compTrainID가 다르면 tmpTrainID+2부터 compTrainID 사이에 없는 ID를 errorInformation에 저장하기
                                        else:
                                            tmpErr = list(range(int(tmpTrainID+2),int(compTrainID),2))
                                            errorTrainIDList.append(tmpErr)
                                

                                    ## 2.3. 급행 열차의 경우: 열차 번호가 9500 이상이다
                                    elif tmpInformation[j] >= 9500:
                                        ## tmpTrainID 이후에 나오는 급행열차의 ID가 tmpTrainID+2여야 한다.
                                        tmpTrainID = tmpInformation[j]

                                        ## 이후에 처음 나오는 급행열차 ID를 compTrainID에 저장
                                        k = 1
                                        while True:
                                            if j+k >= len(tmpInformation)-1 or tmpInformation[j+k] >= 9500: break
                                            else: k += 1
                                        compTrainID = tmpInformation[j+k]

                                        ## tmpTrainID와 compTrainID가 같은지 확인해서 같으면 통과
                                        if j+k >= len(tmpInformation)-1 or tmpTrainID+2 == compTrainID: continue

                                        ## tmpTrainID와 compTrainID가 다르면 tmpTrainID+2부터 compTrainID 사이에 없는 ID를 errorInformation에 저장하기
                                        else:
                                            tmpErr = list(range(int(tmpTrainID+2),int(compTrainID),2))
                                            errorTrainIDList.append(tmpErr)

                                    else: continue                                              ## NaN이므로 건너뜀

                                errorTrainIDList = sum(errorTrainIDList,[])                     ## errorTrainIDList를 1차원으로 변환
                                errorInformation1.append(pd.Series(errorTrainIDList,name=resultInformation1.columns[i]))    ## errorInformation1 list에 errorTrainIDList를 저장하기
                            
                            errorInformation1 = pd.DataFrame(errorInformation1).transpose()     ## errorInformation1을 data frame 형식으로 변환하기
                            #print(errorInformation1)

                            ## errorInformation1의 정보를 csv 파일로 저장
                            errorInformation1.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-rtPos_seqErrInfoByOvtStn-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)

                            # 3. 0902역부터 0938역까지의 실제 열차 도착 정보를 입력하기

                            # 0902역부터 0938역까지의 실제 열차 도착 정보를 입력할 list 생성
                            resultInformation_arr = []      # 도착정보
                            resultInformation_dep = []      # 출발정보

                            # resultInformation1에 적혀 있는 급행/완행 역전 역의 ID의 위치를 가져올 변수 reverseStnIDIndex 정의
                            reverseStnIDIndex = -1
                            for stnID in range(901,938+1):
                                ## 3.1. stnID가 첫 번재 추월 가능 역보다 전에 있는 경우
                                if stnID < int(resultInformation1.columns[0][:-4]):
                                    ## 3.1.1 stnID가 901(개화역)인 경우: 급행열차가 지나가지 않으므로, 급행 열차 번호를 제외해야 한다.
                                    if stnID == 901:
                                        tmpResultInformationSeries = resultInformation1.iloc[:,0]
                                        tmpResultInformationList = list(tmpResultInformationSeries[tmpResultInformationSeries<9500])
                                        resultInformation_arr.append(pd.Series(tmpResultInformationList,name=str(stnID)))
                                        resultInformation_dep.append(pd.Series(tmpResultInformationList,name=str(stnID)))
                                        continue
                                    
                                    resultInformation_arr.append(resultInformation1.iloc[:,0].dropna().rename(str(stnID)))
                                    resultInformation_dep.append(resultInformation1.iloc[:,0].dropna().rename(str(stnID)))

                                ## 3.2. stnID가 마지막 추월 가능 역보다 뒤에 있는 경우
                                elif stnID > int(resultInformation1.columns[-1][:-4]):
                                    resultInformation_arr.append(resultInformation1.iloc[:,-1].dropna().rename(str(stnID)))
                                    resultInformation_dep.append(resultInformation1.iloc[:,-1].dropna().rename(str(stnID)))

                                ## 3.3. stnID가 급행/완행 역전 역의 ID인 경우: 해당 정보를 각각 arr와 dep에 저장
                                elif stnID == int(resultInformation1.columns[reverseStnIDIndex+1][:-4]):
                                    reverseStnIDIndex += 1
                                    resultInformation_arr.append(pd.Series(list(resultInformation1.iloc[:,reverseStnIDIndex].dropna()),name=str(stnID)))
                                    reverseStnIDIndex += 1
                                    resultInformation_dep.append(pd.Series(list(resultInformation1.iloc[:,reverseStnIDIndex].dropna()),name=str(stnID)))

                                ## 3.4. stnID가 급행/완행 역전 역의 ID보다 큰 경우
                                elif stnID > int(resultInformation1.columns[reverseStnIDIndex][:-4]):
                                    ## 마지막으로 지나친 추월 가능 역에서의 출발 데이터를 열차 도착 순서로 사용
                                    ### 이때 생각할 수 있는 문제는, 다음 추월 가능역에 찍혀있는 이전에 추가된 열차의 정보는 이 중간 역들에서는 구하지 못한 값이 되는 것인가 이 부분은 잘 모르겠다.
                                    resultInformation_arr.append(pd.Series(list(resultInformation1.iloc[:,reverseStnIDIndex].dropna()),name=str(stnID)))
                                    resultInformation_dep.append(pd.Series(list(resultInformation1.iloc[:,reverseStnIDIndex].dropna()),name=str(stnID)))
                                
                                else:
                                    print("Error 1: No matching data from reverse stations")
                                    exit()

                            ## resultInformation을 data frame으로 변환
                            resultInformation_arr = pd.DataFrame(resultInformation_arr).transpose()
                            resultInformation_dep = pd.DataFrame(resultInformation_dep).transpose()

                            ## resultInofrmation을 csv 파일로 저장
                            resultInformation_arr.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-rtPos_arrSeqByStn-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)
                            resultInformation_dep.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-rtPos_depSeqByStn-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)

                            ## ============ 준희 작성 코드 종료 ==========

                    # 5. 파일 저장
                    # 5.1. 열차 정보 합치기
                    myd3_arr_ttable = pd.merge(trainsInfo, myd3_arr_ttable, left_on='trainNo2', right_on=myd3_arr_ttable.index) # 열차번호 기준의 열차정보 병합
                    myd3_dep_ttable = pd.merge(trainsInfo, myd3_dep_ttable, left_on='trainNo2', right_on=myd3_dep_ttable.index) # 열차번호 기준의 열차정보 병합
                    myd3_occ = pd.merge(trainsInfo, myd3_occ, left_on='trainNo2', right_on=myd3_occ.index) # 열차번호 기준의 열차정보 병합

                    if overtakeAnalysisTF == True :
                        myd3_numOvertakeTable = pd.merge(trainsInfo, numOvertakeTable, left_on='trainNo2', right_on=numOvertakeTable.index) # 열차번호 기준의 열차정보 병합
                        myd3_overtakeTrainNoTable = pd.merge(trainsInfo, overtakeTrainNoTable, left_on='trainNo2', right_on=overtakeTrainNoTable.index) # 열차번호 기준의 열차정보 병합

                    #myd3_occ = myd3_occ.apply(lambda x : x/dt.timedelta(seconds=1) if x!="NaN" else "NaN" if x!="NaT" else "NaN") #import datetime as dt
                    #print("myd3_staOccTime in seconds \n")
                    #print(myd3_occ)
                    
                    # 5.2. 열차 데이터 저장하기
                    myd3_arr_ttable.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_arr_ttable-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)
                    myd3_dep_ttable.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_dep_ttable-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)
                    myd3_occ.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_staOccTime-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)
                    
                    if overtakeAnalysisTF == True :
                        myd3_numOvertakeTable.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_numOvertakeTable-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)
                        myd3_overtakeTrainNoTable.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_overtakeTrainNoTable-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", encoding='euc-kr', index=False)

                else :
                    print(oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_staOccTime.csv_and_occDensity.png_had_not_been_created."+"\n") 
                    f = open(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_staOccTime_csv_and_occDensity_png_had_not_been_created-pc_"+pcNm+"-"+fwDate+".txt", mode='wt', encoding='utf-8')
                    printer = oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_staOccTime.csv_and_occDensity.png_had_not_been_created."+"\n"
                    f.write(printer)
                    f.close() 
                
                
                # 6. 열차 별 기점 역 출발시각을 0초로 변환한 열차 Diagram 형태의 운행시간 분포 그래프 작성 : 신분당선 강남방향에 대해서만 작동함

                if ((lineNm == "LineS") & (bnd == 0)) | ((lineNm == "Line9") & (bnd == 1)) | (lineNm=="LineA") : #신분당선 강남방향 또는 9호선 종합운동장방향
                    # 6.1. keySta 값 가져오기
                    keySta = int( keyStations['statnNm'][np.where(keyStations['lineNm']==lineNm)[0]] )


                    # 6.2. 노선별 keyStation 기준으로 출발 데이터 정렬
                    # 6.2.1. 노선별 keyStation 기준으로 출발 데이터 정렬
                    myd3_dep_ttable_sorted = myd3_dep_ttable.sort_values(keySta, ascending=True)

                    # 6.2.2. 출발데이터 - 정보가 NA인 열차를 분석에서 제외
                    dep_nonNaRow_id = np.where(myd3_dep_ttable_sorted.T.isnull().sum()==0)
                    dep_nonNaRow_id = dep_nonNaRow_id[0]
                    dep_nonNaRow = myd3_dep_ttable_sorted.index[dep_nonNaRow_id]
                    #print(dep_nonNaRow)

                    # 6.2.3 도착데이터 - 노선별 keyStation 기준으로 출발 데이터 정렬
                    myd3_arr_ttable_sorted = myd3_arr_ttable.sort_values(keySta, ascending=True)

                    # 6.2.4. 도착데이터 - 정보가 NA인 열차를 분석에서 제외
                    arr_nonNaRow_id = np.where(myd3_arr_ttable_sorted.T.isnull().sum()==0)
                    #print(arr_nonNaRow_id)
                    arr_nonNaRow_id = arr_nonNaRow_id[0]
                    arr_nonNaRow = myd3_arr_ttable_sorted.index[arr_nonNaRow_id]
                    #print(arr_nonNaRow)


                    # 6.3. 도착, 출발 데이터셋에서 NA가 없는 행 교집합 찾기
                    nonNa_trainId = df(set(dep_nonNaRow).intersection(arr_nonNaRow))

                    
                    if len(nonNa_trainId.columns)>0 :
                        nonNa_trainId.columns = ['trainId']

                        # 6.4. 도착,출발 데이터셋에서 모두 NA가 없는 열차들이 각 데이터셋에서 몇번째 행에 있는지를 저장
                        nonNa_trainId['arr_row_id'] = nonNa_trainId.trainId.apply(lambda x: np.where(myd3_arr_ttable_sorted.index == x)[0][0])
                        nonNa_trainId['dep_row_id'] = nonNa_trainId.trainId.apply(lambda x: np.where(myd3_dep_ttable_sorted.index == x)[0][0])

                        nonNa_trainId

                        # 6.5. 각 열차의 기점역 출발시각을 기준으로 도착,출발시각 데이터를 변환
                        # 6.5.1. NA가 없는 행을 추출한 dataframe 생성
                        myd3_dep_ttable_nonNa = myd3_dep_ttable_sorted.iloc[nonNa_trainId['dep_row_id'],7:len(myd3_dep_ttable_sorted.columns)] # 8번째 column부터 역 코드가 들어있고, 앞에 7개에는 열차번호 등의 정보가 포함되어있음
                        myd3_arr_ttable_nonNa = myd3_arr_ttable_sorted.iloc[nonNa_trainId['arr_row_id'],7:len(myd3_arr_ttable_sorted.columns)]

                        # 6.5.2. 도착, 출발 데이터의 공통 열 추출
                        commonSta = df(set(myd3_app_ttable.columns[7:len(myd3_arr_ttable_sorted.columns)]).intersection(myd3_dep_ttable.columns[7:len(myd3_dep_ttable_sorted.columns)]))
                        commonSta.columns = ["commonSta"]
                        commonSta = commonSta.sort_values('commonSta', ascending=True).reset_index(drop=True)
                        sortingSta = commonSta.commonSta[0]

                        # 6.5.3. Departure and Arrival time Difference 계산
                        # 6.5.3.1. 출발 시각 차이 계산
                        myd3_dep_ttable_nonNa = myd3_dep_ttable_nonNa.sort_values(sortingSta, ascending=True)
                        if bnd == 0 : 
                            myd3_depTdiff = myd3_dep_ttable_nonNa.apply(lambda x: x - myd3_dep_ttable_nonNa.iloc[:,len(myd3_dep_ttable_nonNa.columns)-1]) # 각 시각을 각 열차의 출발역 출발시각 값으로 빼줌
                        elif bnd == 1 :
                            myd3_depTdiff = myd3_dep_ttable_nonNa.apply(lambda x: x - myd3_dep_ttable_nonNa.iloc[:,0]) # 각 시각을 각 열차의 출발역 출발시각 값으로 빼줌
                        else :
                            print("bnd is out of bound. bnd:", bnd)

                        # 6.5.3.2. 도착 시각 차이 계산
                        myd3_arr_ttable_nonNa = myd3_arr_ttable_nonNa.sort_values(sortingSta, ascending=True)
                        if bnd == 0 : 
                            myd3_arrTdiff = myd3_arr_ttable_nonNa.apply(lambda x: x - myd3_dep_ttable_nonNa.iloc[:,len(myd3_dep_ttable_nonNa.columns)-1])  # 각 시각을 각 열차의 출발역 출발시각 값으로 빼줌. 가장 빠른 도착시각으로 빼지 않는 이유는 그림 그릴 때 기준값을 통일하기 위해서임. 출발시각을 쓴 이유는 첫 기점역의 도착시각 정보가 없는 경우가 많아서임
                        elif bnd == 1 :
                            if lineNm == "Line9" :
                                myd3_arrTdiff = myd3_arr_ttable_nonNa.apply(lambda x: x - myd3_dep_ttable_nonNa.iloc[:,1])  # 9호선 중앙보훈병원방향 (bnd1)은 급행열차가 김포공항에서 출발하므로 .iloc에서 참조할 열 번호가 1(두번째 열) 임.
                            else :
                                myd3_arrTdiff = myd3_arr_ttable_nonNa.apply(lambda x: x - myd3_dep_ttable_nonNa.iloc[:,0])  # 각 시각을 각 열차의 출발역 출발시각 값으로 빼줌. 가장 빠른 도착시각으로 빼지 않는 이유는 그림 그릴 때 기준값을 통일하기 위해서임. 출발시각을 쓴 이유는 첫 기점역의 도착시각 정보가 없는 경우가 많아서임
                            
                        else :
                            print("bnd is out of bound. bnd:", bnd)


                        if len(myd3_dep_ttable_nonNa.index) == sum(myd3_dep_ttable_nonNa.index == myd3_arr_ttable_nonNa.index):
                            print("Dep and Arr has same index")
                        else : 
                            print("There are something wrong at indeces of Dep and Arr dataset")
                        
                        # 6.6. 열차 별 기점역 출발시각 정보로 보정된 역 별 출도착시각을 병합
                        trains = myd3_depTdiff.T.columns#[0:2]
                        train = trains[0]

                        dia = df()

                        for train in trains:
                            mydia_dep = myd3_depTdiff[myd3_depTdiff.index==train]
                            mydia_arr = myd3_arrTdiff[myd3_arrTdiff.index==train]
                            mydia_dep2 = myd3_dep_ttable_nonNa[myd3_depTdiff.index==train]
                            mydia_arr2 = myd3_arr_ttable_nonNa[myd3_arrTdiff.index==train]

                            mydia = pd.concat([mydia_dep.T, mydia_arr.T], axis=0, sort=False)
                            mydia2 = pd.concat([mydia_dep2.T, mydia_dep2.T], axis=0, sort=False)

                            mydia.reset_index(drop=False, inplace=True)
                            mydia = mydia.sort_values(train, ascending=True)
                            mydia.columns = ['statnId', train]

                            mydia2.reset_index(drop=False, inplace=True)
                            mydia2 = mydia2.sort_values(train, ascending=True)
                            mydia2.columns = ['statnId', train]

                            # bnd checking needed
                            if lineNm == "LineS" :
                                mydia.statnId = mydia.statnId.apply(lambda x: np.abs(x-6819))
                                mydia.reset_index(drop=True, inplace=True)

                                mydia2.statnId = mydia2.statnId.apply(lambda x: np.abs(x-6819))
                                mydia2.reset_index(drop=True, inplace=True)


                            if train == trains[0]:
                                dia = mydia
                                dia2 = mydia2
                            else:
                                dia = pd.concat([dia, mydia[train]], axis=1)
                                #dia = pd.merge(dia, mydia, on='statnId', how='inner')
                                dia2 = pd.concat([dia2, mydia2[train]], axis=1)

                            # LineS_rtPos_merged = pd.merge(LineS_rtPos, LineS_rtPos_from_rtArr, how='outer', on=['trainNo', 'recptnDt', 'trainSttus', 'statnId', 'statnNm', 'statnTnm', 'updnLine'])
                            #dia

                        # 6.7. 역이름-ID 매칭 데이터프레임을 참조하여 역 ID를 부여
                        for i in range(0,len(operDist.index)):
                            dia = dia.replace({'statnId':operDist.bnd0_statnId.iloc[i]}, {'statnId':operDist.bnd0_arrDist.iloc[i]})
                            dia2 = dia2.replace({'statnId':operDist.bnd0_statnId.iloc[i]}, {'statnId':operDist.bnd0_arrDist.iloc[i]})

                        dia

                        # 7. 열차별 기점역 출발시각을 기준으로 다이아 그려보기
                        # 7.1. 모든 열차에 대해 다이아 그려보기
                        trains = myd3_depTdiff.T.columns[0:2]

                        plt.plot(dia[trains], dia['statnId'])
                        #print(TT_dia)
                        plt.savefig(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_relativeTTbyTrain-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".png", dpi=300)
                        plt.close() # plt.close 안하면 뒷 그림에 영향을 주므로 반드시 쓸 것

                        #dia.to_csv(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_dia1-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv")


                        # 7.1. 특정 열차에 대해 다이아 그려보기
                        trains = myd3_depTdiff.T.columns[2:4]
                        
                        #myPltDf = pd.concat([dia2[trains], dia2['statnId']], axis=1) #to_datetime(dia2[trains], format="%Y-%m-%d %H:%M:%S")
                        #myPltDf

                        if(len(trains)>0):
                            plt.plot(dia2[trains], dia2['statnId'])
                            #plt.plot(myPltDf[trains].astype("datetime64"), myPltDf['statnId'])

                            #dia2.to_csv(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_dia2-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv")

                            plt.savefig(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_absTTbyTrain-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".png", dpi=600)
                            plt.close()
                        else :
                            f = open(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_absTTbyTrain_out_of_bounds-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".txt", mode='wt', encoding='utf-8')
                            f.write(oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_absTTbyTrain_out_of_bounds \n")
                            f.close()

                    else : # if len(nonNa_trainId.columns)>0 :
                        f = open(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_absTT_allTrainHasNA-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".txt", mode='wt', encoding='utf-8')
                        f.write("All of the train has NA data \n")
                        f.close()
                    


                    # 8. 보유중인 모든 데이터에 대한 다이어그램 그리기

                    arrTrains = myd3_arr_ttable_sorted.index  # myd3_arr_ttable_sorted.trainNo2 를 쓸 수 없음. index를 가져가야 아래에서 참조가 가능한 구조임.
                    stations = myd3_arr_ttable_sorted.columns[7:len(myd3_arr_ttable_sorted.columns)] #8열부터 역 정보가 시작됨
                    nSta = len(stations)-1 # 총 역 갯수. 단 index로 사용하기 위해, array번호체계가 0부터 시작하므로 1을 더 빼줌
                    #print(stations)
                    mydia_LongTable = df()#index=range(0,1), columns=['trainNo', 'arrT', 'arrSta', 'depT', 'depSta'])

                    print("nSta:"+str(nSta))

                    for train in arrTrains: #range(0,len(arrTrains)):
                        #print("train: ", myd3_arr_ttable_sorted.trainNo2[train])#, " sta:", stations[(nSta-j)])
                        temp_mydia_LongTable = df()#index=range(0,1), columns=['trainNo', 'arrT', 'arrSta', 'depT', 'depSta'])  
                        for j in range(0,nSta):
                            #print("j:",j)
                            # 8.1. 출발역 도착시각 정보 저장하기
                            if bnd==0 : 
                                #print("train: ", myd3_arr_ttable_sorted.trainNo2[train], " sta:", stations[(nSta-j)])
                                staT0 = stations[(nSta-j)]
                                staT1 = stations[(nSta-j-1)]
                            else :
                                #print("train: ", myd3_arr_ttable_sorted.trainNo2[train], " sta:", stations[j])
                                staT0 = stations[j]
                                staT1 = stations[j+1]

                            tempArrT0 = myd3_arr_ttable_sorted.loc[train,staT0] # 출발역 도착시각 정보를 저장
                            if bnd==0 : 
                                tempArrSta0 = stations[(nSta-j)] # 역 정보 저장
                            else :
                                tempArrSta0 = stations[j] # 역 정보 저장

                            # 8.2. 출발역 출발시각 정보 저장하기
                            if tempArrT0 is not pd.NaT: # 출발역 도착정보가 있으면,
                                tempDepT0 = myd3_dep_ttable_sorted.loc[train,staT0] # 출발역 출발시각 정보를 저장
                                tempDepSta0 = staT0 # 역 정보 저장
                                #print("tempDepT0: ", tempDepT0)

                                # 8.3. 출발역 도착-출발시각 정보 저장
                                # 8.3.1. 출발역 출발시각 정보가 있으면
                                if tempDepT0 is not pd.NaT: 
                                    if j < (len(stations)-1): # j 범위 체크 
                                        tempDfSta0 = df({"trainNo2":[myd3_arr_ttable_sorted.trainNo2[train], myd3_arr_ttable_sorted.trainNo2[train]], "recptnDt":[tempArrT0, tempDepT0], "statnId":[tempArrSta0, tempDepSta0], "Dist":[tempArrSta0, tempDepSta0], "trainSttus":[1, 2] })
                                        ##print("-1-------\n")
                                        ##print(tempDfSta0)
                                        #tempDfSta0 = df(tempDfSta0)

                                        # 8.3.1.1. 출발역 도착-출발시각 저장
                                        temp_mydia_LongTable = pd.concat([temp_mydia_LongTable, tempDfSta0], axis=0) 
                                        #print("-2-------\n")
                                        #print(temp_mydia_LongTable)

                                        # 8.3.1.2. 도착역 도착시각 정보 저장하기
                                        tempArrT1 = myd3_arr_ttable_sorted.loc[train,staT1] 
                                        tempArrSta1 = staT1 # 역 정보 저장

                                    # 8.3.2. 도착역 도착시각 정보가 있으면
                                    if tempArrT1 is not pd.NaT: 
                                        # 출발역 출발시각 + 도착역 도착시각 (운행중) 저장
                                        tempDfSta0Sta1 = df({"trainNo2":[myd3_arr_ttable_sorted.trainNo2[train], myd3_arr_ttable_sorted.trainNo2[train]], "recptnDt":[tempDepT0, tempArrT1], "statnId":[tempDepSta0, tempArrSta1],"Dist":[tempDepSta0, tempArrSta1], "trainSttus":[2, 1] })   
                                        ##print("-3-------\n")
                                        ##print(tempDfSta0Sta1)
                                        # temp_mydia_LongTable에 데이터 병합
                                        temp_mydia_LongTable = pd.concat([temp_mydia_LongTable, tempDfSta0Sta1], axis=0) 

                                        #print("-4-------\n")
                                        #print(temp_mydia_LongTable)
                                    
                                    #else:
                                        #print("도착역 도착시각 정보 없음. trainNo:", train, " station:", stations[(j+1)], "\n")

                                    # 8.3.3. j가 마지막이면 : 출발역 정보에서 끝.
                                    else: 
                                        tempDfSta0 = df({"trainNo2":[myd3_arr_ttable_sorted.trainNo2[train], myd3_arr_ttable_sorted.trainNo2[train]], "recptnDt":[tempArrT0, tempDepT0], "statnId":[tempArrSta0, tempDepSta0], "Dist":[tempArrSta0, tempDepSta0], "trainSttus":[1, 2] })
                                        ##print("-1-1-------\n")
                                        ##print(tempDfSta0)

                                        temp_mydia_LongTable = pd.concat([temp_mydia_LongTable, tempDfSta0], axis=0) 
                                        #print("-2-1------\n")
                                        #print(temp_mydia_LongTable)
                                
                                #else: # 출발역 출발시각 정보가 없으면 : 도착시각만 있음 --> 데이터 병합 안함
                                    #print("출발역 출발시각 정보 없음. trainNo:", train, " station:", stations[(nSta-j)], "\n")
                            
                            else : # 8.3.4. 출발역 도착정보가 없으면
                                tempDepT0 = myd3_dep_ttable_sorted.loc[train,staT0] # 출발역 출발시각 정보를 저장
                                tempDepSta0 = staT0 # 역 정보 저장

                                # 출발역~도착역 주행시간 정보 저장하기
                                if tempDepT0 is not pd.NaT: # 출발역 출발시각 정보가 있으면
                                    if j < (len(stations)-1): # j 범위 체크 
                                        tempArrT1 = myd3_arr_ttable_sorted.loc[train,staT1] # 도착역 도착시각 정보 저장하기
                                        tempArrSta1 = staT1 # 역 정보 저장

                                        if tempArrT1 is not pd.NaT: # 도착역 도착시각 정보가 있으면
                                            # 출발역-도착역 운행시간 정보 저장
                                            tempDfSta0Sta1 = df({"trainNo2":[myd3_arr_ttable_sorted.trainNo2[train], myd3_arr_ttable_sorted.trainNo2[train]], "recptnDt":[tempDepT0, tempArrT1], "statnId":[tempDepSta0, tempArrSta1],"Dist":[tempDepSta0, tempArrSta1], "trainSttus":[2, 1]})   
                                            ##print("-3-1-------\n")
                                            ##print(tempDfSta0)

                                            # temp_mydia_LongTable에 데이터 병합
                                            temp_mydia_LongTable = pd.concat([temp_mydia_LongTable, tempDfSta0Sta1], axis=0) 
                                            #print("-4-1------\n")
                                            #print(temp_mydia_LongTable)
                                        
                                        #else:
                                            #print("도착역 도착시각 정보 없음. trainNo:", train, " station:", stations[(j+1)], "\n")
                                        
                                    #else:
                                        #print("마지막 역입니다. trainNo:", train, " station:", stations[(nSta-j)], "\n")

                                #else:
                                    #print("출발역 출발시각 정보 없음. trainNo:", train, " station:", stations[(j)], "\n")
                            
                            #del(tempArrT0, tempArrSta0, tempDepT0, tempDepSta0, tempArrT1, tempArrSta1)


                        #print("\n+++++++++++\n")
                        #print(temp_mydia_LongTable)
                        # 8.4. 데이터 중복 제거 + 열차 상태정보(접근,도착,출발)기준 Sorting 후 데이터 수신시각 기준으로 다시 Sorting
                        if len(temp_mydia_LongTable.index)>0 :
                            temp_mydia_LongTable = temp_mydia_LongTable.sort_values('trainSttus', ascending=True)
                            temp_mydia_LongTable = temp_mydia_LongTable.sort_values('recptnDt', ascending=True)
                            temp_mydia_LongTable = temp_mydia_LongTable.drop_duplicates(["trainNo2", "recptnDt", "statnId", "Dist", "trainSttus"]) 
                        
                        # 8.5. 역차 별로 합쳐진 역 도착-출발, 출발역-도착역 구간 출발-도착 정보를 병합 (행 방향으로)
                        mydia_LongTable = pd.concat([mydia_LongTable, temp_mydia_LongTable], axis=0)
                        del(temp_mydia_LongTable)
                    


                    print("\n=========\n")
                    #print(mydia_LongTable)

                    # 8.6. 역 정보를 기준으로 Distance 열에 있는 값을 거리로 치환
                    for i in range(0,len(operDist.index)):
                        if bnd == 0 :
                            mydia_LongTable = mydia_LongTable.replace({'Dist':operDist.statnId.iloc[i]}, {'Dist':operDist.bnd0_arrDist.iloc[i]})
                        else :
                            mydia_LongTable = mydia_LongTable.replace({'Dist':operDist.statnId.iloc[i]}, {'Dist':operDist.bnd1_arrDist.iloc[i]})


                    print(mydia_LongTable) #.iloc[0:100,:])

                    # 8.7. 실시간 위치데이터의 Dist(이정-postmile) 조정 : 도착은 역 상류 0.4km, 출발은 역 하류 0.2km
                    #print(mydia_LongTable)
                    #print("\n------\n")
                    mydia_LongTable = adjustDist(mydia_LongTable, 1, -0.4)
                    mydia_LongTable = adjustDist(mydia_LongTable, 2, 0.2)

                    mydia_LongTable.to_csv(path_or_buf=outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_mydia_LongTable-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".csv", index=False)
    

                    axisT = datetime.strptime(tTableDate+" 04:00:00", "%Y-%m-%d %H:%M:%S") #str(oDate)
                    min30 = datetime.strptime("00:30:00", "%H:%M:%S")

                    #nTstamp = 21*2+1
                    #timeAxis = np.arange(nTstamp)

                    #for i in range(0,nTstamp):
                    #    timeAxis[i] = axisT + dt.timedelta(minutes=30)*i
                    #timeAxis

                    # 8.8. 전체 데이터를 활용한 Diagram 그리기
                    ## For문 돌릴 열차 ID 값 가져오기
                    trains = mydia_LongTable.trainNo2.unique()

                    for train in trains:
                        myTrainRows = np.where(mydia_LongTable.trainNo2 == train)[0]
                        diaByTrain = mydia_LongTable.iloc[myTrainRows,:]
                        diaByTrain = diaByTrain.sort_values('recptnDt', ascending=True).reset_index(drop=True, inplace=False)
                        diaByTrain = diaByTrain.sort_values('Dist', ascending=True).reset_index(drop=True, inplace=False)
                        plt.plot(diaByTrain.recptnDt, diaByTrain.Dist)

                    #plt.xticks(np.arange(min(diaByTrain.recptnDt), max(diaByTrain.recptnDt)+1, dt.timedelta(minutes=30)))  
                    #plt.xaxis.set_major_formatter(ticker.FormatStrFormatter('%m-%d %H:%M'))
                    #plt.show()
                    plt.savefig(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_Diagram_wholeTrains-"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".png", dpi=300)
                    plt.close()

                    ## 8.9. 일부열차만 발췌해서 보기
                    trains = mydia_LongTable.trainNo2.unique()
                    fromTrain = 60
                    toTrain = 90

                    if len(trains[fromTrain:toTrain])>0 :
                        for train in trains[fromTrain:toTrain]:
                            myTrainRows = np.where(mydia_LongTable.trainNo2 == train)[0]
                            diaByTrain = mydia_LongTable.iloc[myTrainRows,:]
                            diaByTrain = diaByTrain.sort_values('recptnDt', ascending=True).reset_index(drop=True, inplace=False)
                            diaByTrain = diaByTrain.sort_values('Dist', ascending=True).reset_index(drop=True, inplace=False)
                            plt.plot(diaByTrain.recptnDt, diaByTrain.Dist)
                            plt.plot(diaByTrain.recptnDt, diaByTrain.Dist)
                        #plt.show()
                        plt.savefig(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_Diagram_Trains_from_"+str(trains[fromTrain])+"_to_"+str(trains[len(trains)-1])+"_"+nFiles+"_files-pc_"+pcNm+"-"+fwDate+".png", dpi=300)
                        plt.close()
                    else :
                        print(oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_Diagram_Trains_out_of_bounds")
                        f = open(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_Diagram_Trains_out_of_bounds-pc_"+pcNm+"-"+fwDate+".txt", mode='wt', encoding='utf-8')
                        f.write(oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_Diagram_Trains_out_of_bounds"+"\n") 
                        f.close()

                del(myd3) # 변수 삭제

                if 'myd3_app_ttable' in locals():
                    del(myd3_app_ttable)
                if 'myd3_arr_ttable' in locals():
                    del(myd3_arr_ttable)
                if 'myd3_dep_ttable' in locals():
                    del(myd3_dep_ttable)
                if 'myd3_occ' in locals():
                    del(myd3_occ)
                if 'trainsInfo' in locals():
                    del(trainsInfo)
        
        else : #if(len(myd3.index)>0):
            # Pivottable을 못만들 정도로 데이터가 없는 경우 해당일의 열차시각표를 생성하지 않고 넘어감
            print(oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_does_not_have_timetable_data_due_to_the_lack_of_the_data. len(myd3.index):"+str(len(myd3.index)))
            f = open(outputPath+"/"+oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_does_not_have_timetable_data_due_to_the_lack_of_the_data-pc_"+pcNm+"-"+fwDate+".txt", mode='wt', encoding='utf-8')
            f.write(oDate+"-"+lineNm+"-bnd"+str(bnd)+"-"+dataType+"_does_not_have_timetable_data_due_to_the_lack_of_the_data. len(myd3.index):"+str(len(myd3.index))+"\n") 
            f.close()
            del(myd3)



def addMigumData(myLine_rtData, rtArrResDataPath, oDate, lineNm, statnIdNm):
    # 1. 입력받은 rtArrResDataPath 에 포함된 파일 목록 가져오기
    rtArrDataDir = os.listdir(rtArrResDataPath)

    # 2. 파일 목록 중 .csv 파일만 추리기
    rtArrDataDir = [file for file in rtArrDataDir if file.endswith(".csv")]

    # 3. 파일 목록 중 분석대상날짜에 맞는 미금역 데이터를 가지고있는 rtArr 파일을 찾음
    rtArrDataList = [file for file in rtArrDataDir if file.startswith(oDate+"-"+lineNm+"-bndA-rtPos_from_rtArr2-")]

    # 4. 파일 읽기
    if len(rtArrDataList)==1:
        RtPosFromRtArr = pd.read_csv(rtArrResDataPath+"/"+rtArrDataList[0]) 

        # 5. 미금역 데이터 발췌
        migumRtPosFromRtArr = RtPosFromRtArr[["trainNo", "directAt", "recptnDt", "trainSttus", "statnId", "statnNm", "lstcarAt", "statnTnm", "statnTid", "updnLine", 'old_statnId', 'old_statnTid']].iloc[np.where(RtPosFromRtArr.statnNm == "미금")[0]]
        migumRtPosFromRtArr = migumRtPosFromRtArr.iloc[np.where(migumRtPosFromRtArr.trainSttus!=0)[0],:]
        migumRtPosFromRtArr = rtDataSortingAndDropDuplicates(migumRtPosFromRtArr, "trainNo")
        

        # 6. 미금역 도착정보 trainSttus 중복 데이터 제거
        trains = migumRtPosFromRtArr.trainNo.unique() # 해당일 영업 편성번호 추출

        tempMigumData = df()
        #train = trains[5]
        for train in trains:
            tempTrain = migumRtPosFromRtArr.iloc[np.where(migumRtPosFromRtArr.trainNo==train)[0],:] # 1개 편성 데이터 선택
            tempTrain["updnLine2"] = tempTrain["updnLine"].copy()
            updnLineChange = tempTrain.updnLine.iloc[0]
            updnLineChange2 = 0

            # 미금역에 해당 편성 열차가 도착한 순서를 undnLine2 column에 부여
            for i in range(0,len(tempTrain.index)):
                if updnLineChange != tempTrain.updnLine.iloc[i]:
                    updnLineChange = tempTrain.updnLine.iloc[i]
                    updnLineChange2 = updnLineChange2 + 1
                tempTrain.updnLine2.iloc[i] = updnLineChange2
            #tempTrain.to_csv("tempTrain-oDate-190715-fwDate-200702.csv")
            
            # 미금역에 해당 편성 열차가 도착한 순서의 고유값을 도출
            tempTrainArrialOrder = tempTrain.updnLine2.unique()

            # 미금역에 해당 편성 열차가 도착한 순서를 기준으로 trainSttus가 중복으로 기재된 데이터를 검색해서 가장 최신 정보만 사용하도록 처리
            tempTrain3 = df()
            #arrivalOrder = tempTrainArrialOrder[0]
            for arrivalOrder in tempTrainArrialOrder:
                tempTrain2 = tempTrain.iloc[np.where(tempTrain.updnLine2==arrivalOrder)[0],:] # 1개 편성 데이터 선택
                tempTrain2 = tempTrain2.sort_values(by='trainSttus', ascending=True) # 열차 상태정보 기준 Sorting
                tempTrain2 = tempTrain2.sort_values(by='recptnDt', ascending=False) # 열차 정보 수신시각을 내림차순으로 Sorting : trainSttus 기준으로 가장 마지막에 수신된 값만 남기고 삭제하기 위해 - python 에서는 drop_duplicates를 하면 고유값 중 iloc이 낮은 순서(가장 위쪽 데이터)를 남기고 나머지를 지우는 듯 함
                tempTrain2 = tempTrain2.drop_duplicates(["trainSttus"]) # 열차 상태정보 고유값만 남기고 데이터 제거
                tempTrain2 = tempTrain2.sort_values(by='recptnDt', ascending=True) # 다시 시간순서로 sorting
                tempTrain3 = pd.concat([tempTrain3, tempTrain2], axis=0)
            
            tempMigumData = pd.concat([tempMigumData, tempTrain3], axis=0)

        # 처리 완료 된 미금역 도착정보 데이터를 updnLine2 정보를 제외하고 저장
        migumRtPosFromRtArr = tempMigumData[["trainNo", "directAt", "recptnDt", "trainSttus", "statnId", "statnNm", "lstcarAt", "statnTnm", "statnTid", "updnLine", 'old_statnId', 'old_statnTid']]
        #migumRtPosFromRtArr.to_csv("migumRtPosFromRtArr-oDate-190715-fwDate-200703.csv")

        # 7. StatnTid 수정
        for i in range(0,len(statnIdNm.index)):
            # 6.1. 역 이름이 statnIdNm[i]와 같은 역을 검색하고, 그 행의 상대index 번호를 twoDigitStatnTidList에 저장
            twoDigitStatnTidList = np.where(migumRtPosFromRtArr.statnTnm == statnIdNm.statnNm.iloc[i])[0]
            # 6.2. 해당 행의 역 ID를 교체
            if len(twoDigitStatnTidList)>0 :
                migumRtPosFromRtArr.statnTid.iloc[twoDigitStatnTidList] = statnIdNm.new_statnId.iloc[i]
                migumRtPosFromRtArr.old_statnTid.iloc[twoDigitStatnTidList] = statnIdNm.statnId.iloc[i]
        
        # 8. 급행정보, 막차정보 열 수정
        migumRtPosFromRtArr = migumRtPosFromRtArr.replace({'directAt':"None"}, {'directAt':0})
        migumRtPosFromRtArr = migumRtPosFromRtArr.replace({'directAt':"급행"}, {'directAt':1})
        migumRtPosFromRtArr.lstcarAt = 0 # 신분당선은 막차 정보가 아얘 없는것 같음

        # 9. rtPos에서 필요한 열만 추린 데이터 밑에 미금역 데이터를 덧붙임
        myLine_rtData = pd.concat([myLine_rtData, migumRtPosFromRtArr], axis=0)  # (아래에 덧붙임)

    else:
        print("Length of rtArrDataList is greater than 1: "+str(len(rtArrDataList)))


    # 8. 미금역 데이터 반환
    return myLine_rtData


# ===========================================================================

def main(): # python3 realtimeStationTimetable-190712.py tlsyslab_jh1_admin Line1 190705 week 2
    #from google.colab import auth
    #auth.authenticate_user()
    #from google.colab import drive
    #drive.mount('/content/2020R01')
    #!cd 2020R01/My\ Drive/2020R01_RT_Rail_Delay_Predict-191111; ls -al;

    # 0. 데이터 전처리 준비
    # 0.1. basePath 설정
    basePath = "T:/06_seoulSubway_realtimeData"

    # 0.2. 읽어들일 노선(Line) 이름 설정
    lineNm = "LineWS" #first # "Line2" # python 실행 코드의 첫번째 parameter로 입력받음
    dataType = "rtPos"

    if dataType == "rtPos" :
        dataTypeFull = "rtPosition"
    elif dataType == "rtArr" :
        dataTypeFull = "rtArrival"


    fwDate = datetime.now()
    fwDate = fwDate.strftime('%y%m%d') #yymmdd format
    print("fwDate : "+fwDate)
    
    # 0.3. 시각표 읽기 관련 parameter 설정
    tTableWeek = 'week'
    tTableBnd = '0'
    tTableDate = '2019-08-16'
    tTableHour = '23' # 23 or 2

    # 0.4. 노선 별 KeyStation 정보 만들기 : 나중엔 다른 위치로 이동해도 될듯
    # 신분당-정자, 2호선-한양대, 5호선-왕십리, 9호선-가양, 공철-김포공항, 분당선-선릉, 경중-가좌 # 끝 4자리 기준
    keyStations = df({"lineNm":["LineS", "Line2", "LIne5", "Line9", "LineA","LineBD", "LineKJ","LineWS"], "statnNm":[6812, 209, 540, 907, 6505, 5215, 5315,4711]}) 

    # 0.5. 실시간 도착정보 데이터 들어있는 경로 설정
    dataPath = basePath+"/1_rawData/"+dataTypeFull+"/"+lineNm+"/oDate-20200701-20201031"
    rtPosResDataPath = basePath+"/2_data"+"/rtPosition/"+lineNm
    rtArrResDataPath = basePath+"/2_data"+"/rtArrival/"+lineNm

    print("dataPath : "+dataPath)

    # 0.6. rawData 전처리 결과 파일 저장할 경로 설정
    outputPath = basePath+"/2_data/"+dataTypeFull+"/"+lineNm+"/oDate-20200701-20201031"
    print("outputPath : "+outputPath)
    if not os.path.exists(outputPath): # 결과파일 저장할 폴더가 없을 경우 생성
        print("outputPath doesn't exist. Python code will make the outputPath folder.")
        os.makedirs(outputPath)


    # 0.7. rawData 폴더 내 파일목록 확인
    # 0.7.1. 폴더 내 csv파일 추출
    mydir = os.listdir(dataPath) # file 목록 : 폴더도 포함한 상태 포함
    print("\n---------\n mydir which is including directories: ")
    print(mydir)

    mydir = [file for file in mydir if file.endswith(".csv")] # file 목록 중 csv파일만 추출

    print("\n---------\n mydir: ")
    print(mydir)

    # 0.7.2. 주어진 폴더 안에서 날짜 별 수집PC별 파일 수집 현황정보 추출 및 정리하기
    dataByDayByPc = df() # 빈 데이터프레임 생성
    dataByDayByPc = checkRawdataFileList(lineNm, dataType, dataPath, mydir, dataByDayByPc) # rawdata 파일 목록 확인 함수 실행
    
    # 0.7.3. 데이터 수집 정보로부터 수집일 추출 
    dataObsDates = dataByDayByPc['oDate'].unique() # 각 데이터의 날짜 별 값의 unique값을 찾아 분석 대상일을 정리 - data observed dates

    # 0.7.4. 데이터 수집 정보 저장
    dataByDayByPc.to_csv(path_or_buf=basePath+"/"+"dataByDayByPc-"+lineNm+"-from_"+dataObsDates[0]+"_to_"+dataObsDates[len(dataObsDates)-1]+"-"+fwDate+".csv", index=False) # 중복제거 확인용
    
    # 0.7.5. 역 별 운행방향 별 이정 및 역 운행순서 정보 읽기
    stationOrderChk = True
    if lineNm=="LineA":
        operDist = pd.read_csv(basePath+"/3_operDistCsv/Oper_Distance_LineA-210730.csv", encoding='euc-kr')  # 200211.csv")
    elif lineNm == "Line9":
        operDist = pd.read_csv(basePath+"/3_operDistCsv/Oper_Distance_Line9-210621.csv", encoding='euc-kr')  # 200211.csv")
    elif lineNm == "LineS":
        operDist = pd.read_csv(basePath+"/3_operDistCsv/Oper_Distance_LineS-210730.csv", encoding='euc-kr')  # 200211.csv")
    elif lineNm == "LineWS":
        operDist = pd.read_csv(basePath+"/3_operDistCsv/Oper_Distance_LineWS-220107.csv", encoding='euc-kr')  # 200211.csv")
    else :
        print("Oper_Distance 파일이 없는 노선을 분석하고있습니다. 해당 노선의 Oper_Distance 파일을 만들어주세요")
        stationOrderChk = False

    # 0.8. 파일 읽기
    # 0.8.1. 데이터 수집일에 대해 데이터 읽어나가기 시작
    if stationOrderChk == True :

        nDates = len(dataObsDates)

        for date in range(0,(nDates)):    
            print(mydir[date]+" || 날짜가 순차적으로 시행되지 않으니 참고해주세요.")

            # 0.8.2. 해당 날짜의 데이터가 몇개인지 확인
            nDataOntheDate = np.where(dataByDayByPc['oDate']==dataObsDates[date])[0]
            print(nDataOntheDate)

            # 0.8.3. 데이터 읽기
            if len(nDataOntheDate) == 1 : # 해당 날짜의 rawdata가 1개이면
                mydata = pd.read_csv(dataPath+"/"+mydir[nDataOntheDate[0]]) # 바로 읽기
                myLine_rtData = mydata

                # 0.8.4. 기초 변수 생성
                ## file 이름 sample : 2019-10-13_Line1_rtTrainPos_merged_4356-files_utlsyslab_10_admin.csv
                oDate = dataByDayByPc['oDate'][nDataOntheDate[0]] #mydir[nDataOntheDate][0:10] # YYYY-MM-DD
                pcNm = dataByDayByPc['pcNm'][nDataOntheDate[0]] #findPcNm(mydir[nDataOntheDate])
                nFiles = dataByDayByPc['nFiles'][nDataOntheDate[0]] # findNumberOfFiles(mydir[nDataOntheDate])
                

            elif len(nDataOntheDate) == 2 : # 해당 날짜의 rawdata가 2개면
                ## 각각 데이터를 읽은 후
                mydata1 = pd.read_csv(dataPath+"/"+mydir[nDataOntheDate[0]])
                mydata2 = pd.read_csv(dataPath+"/"+mydir[nDataOntheDate[1]])
                
                ## 하나로 합침 (아래에 덧붙임) : 아래에서 열차정보 수신시각과 열차ID등으로 중복 행 제거 할 것이기 때문
                mydata = pd.concat([mydata1, mydata2], axis=0)  
                myLine_rtData = mydata

                # 0.8.4. 기초 변수 생성
                ## file 이름 sample : 2019-10-13_Line1_rtTrainPos_merged_4356-files_utlsyslab_10_admin.csv
                oDate = dataByDayByPc['oDate'][nDataOntheDate[0]] #mydir[nDataOntheDate][0:10] # YYYY-MM-DD
                pcNm = str(dataByDayByPc['pcNm'][nDataOntheDate[0]]+"+"+dataByDayByPc['pcNm'][nDataOntheDate[1]]) #findPcNm(mydir[nDataOntheDate])
                nFiles = dataByDayByPc['nFiles'][nDataOntheDate[0]]+"+"+dataByDayByPc['nFiles'][nDataOntheDate[1]] # findNumberOfFiles(mydir[nDataOntheDate])

            else : 
                print("nDataOntheDate is 0 or greater than 2: ", str(nDataOntheDate))

            #myLine_rtData.to_csv(path_or_buf=basePath+"/"+"concat_sample-200614-v1.csv") # concat 확인용
            
            # 1. 데이터 전처리
            # 1.1. 다음을 기준으로 데이터 중복 행을 제거
            ## 급행여부(directAt), 데이터 수신시각(recptnDt), 역이름(statnNm), 열차번호(trainNo), 열차운행상태(trainSttus), 상하행구분(updnLine)
            myLine_rtData = myLine_rtData.drop_duplicates(["directAt", "recptnDt", "statnNm", "trainNo", "trainSttus", "updnLine"]) 

            #myLine_rtData.to_csv(path_or_buf=basePath+"/"+"concat_sample-200614-v2.csv") # 중복제거 확인용

            # 1.2. 데이터 수신시각을 datetime형식으로 변환 후 다시 string으로 저장 : 규격을 맞추기 위해 
            myLine_rtData['recptnDt'] = pd.to_datetime(myLine_rtData['recptnDt'], format="%Y-%m-%d %H:%M:%S") 
            myLine_rtData['recptnDt'] = myLine_rtData['recptnDt'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
            #myLine_rtData = myLine_rtData.sort_values(by='recptnDt', ascending=True)


            # 1.3. 역이름-ID 매칭 
            statnIdNm = creatingNewStatnId(myLine_rtData, lineNm)
            myLine_rtData = changingShortStatnId(myLine_rtData, statnIdNm)


            # 1.5. 분석에 필요한 데이터만 추출해서 저장
            myLine_rtData = myLine_rtData[["trainNo", "directAt", "recptnDt", "trainSttus", "statnId", "statnNm", "lstcarAt", "statnTnm", "statnTid", "updnLine", 'old_statnId', 'old_statnTid']] # 데이터 발췌
            myLine_rtData.reset_index(drop=True, inplace=True) # index 초기화
            #print(myLine_rtData)

            # 1.6. 신분당선 미금역 누락기간 데이터 보정
            if lineNm == "LineS" :
                myLine_rtData = addMigumData(myLine_rtData, rtArrResDataPath, oDate, lineNm, statnIdNm)
                #myLine_rtData.to_csv(basePath+"/"+"rtArrMigumAdded-rtPos-oDate-190715-fwDate-200702.csv")

            # 2. 실시간 열차위치데이터를 통해 열차 접근,도착,출발 위치정보 table 생성
            creatingRtTimetable(myLine_rtData, lineNm, dataType, dataTypeFull, keyStations, operDist, stationOrderChk, basePath, dataPath, outputPath, oDate, pcNm, nFiles, fwDate, tTableDate)
    
    else :
        print("Oper_Distance 파일이 없는 노선을 분석하고있습니다. 해당 노선의 Oper_Distance 파일을 만들어주세요. 시각표 생성 시행 안함")

# python filename.py
if __name__ == "__main__":
    main()
