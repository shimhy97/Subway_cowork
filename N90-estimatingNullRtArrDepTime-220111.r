
# 실시간으로 수집된 실시간 지하철 위치정보 데이터를 creatingTrainOperLogTable-201019-v1.py 를 통해 table형태로 변환한 후,
# 이 코드를 통해 도착/출발 중 한쪽 정보가 누락된 경우에 한해 보정을 시행함
# 주요 경로를 잘 설정해야 함
# 
# 이 코드는 우성종 연구원이 교통카드-열차운행실적 매칭 코드의 중간에 삽입했던 부분을 발췌하여 독립적으로 작동될 수 있도록 오윤석이 수정하였음
# 본 코드의 파일 읽기를 마치는 부분까지는 오윤석 연구원이, 그 이후는 우성종 연구원이 주축이 되어 개발하였음
# effective on 2020-10-20
# 문의 : ysoh0223@korea.ac.kr(오윤석), seongjongwoo@korea.ac.kr(우성종)

## ================================================================================

chkLineNm <- function(mystr, lineNm){
  mystr# = "2019-10-13_Line1_rtTrainPos_merged_4356-files_utlsyslab_jh1_admin.csv"
  end <- str_locate(mystr, "-bnd")[1]-1 #mystr.find("-files")
  start <- end-7+str_locate(substr(mystr, end-7, end), "-")[1] #mystr.find("_",end-5,end)+1
  myLineNm <- substr(mystr, start, end) #mystr[start:end]
  
  if (myLineNm == lineNm) {
    #print(paste0(myLineNm))
    return (myLineNm)  
  } else {
    print(paste0("lineNm of reading file is not matched : ", mystr))
    return ("err")
  }
}

## ================================================================================

findBndId <- function(mystr){
  mystr# = "2019-07-13-LineS-bnd0-rtPos_mydia_LongTable-2316+5070_files-pc_0+10-200624.csv"
  end <- str_locate(mystr, "bnd")[2] #mystr.find("-files")
  myLineNm <- substr(mystr, end-2, end+1)
  
  if (is.na(end) != TRUE) {
    #print(paste0(myLineNm))
    return (myLineNm)  
  } else {
    print(paste0("findBndId Error: ", mystr))
    return ("err")
  }
}

## ================================================================================

findDirectAtId <- function(mystr){
  mystr# = "2019-07-13-LineS-bnd0-rtPos_mydia_LongTable-2316+5070_files-pc_0+10-200624.csv"
  end <- str_locate(mystr, "bnd")[2] #mystr.find("-files")
  mydirectAt <- substr(mystr, end+3, end+3)
  
  if (mydirectAt == "G" | mydirectAt == "E") {
    #print(paste0(myLineNm))
    return (mydirectAt)  
  } else {
    #print(paste0("findDirectAtId Error: ", mystr))
    return ("G") # 나중엔 "Err"로 return해야 함
  }
}

## ================================================================================

findTypeOfInputData <- function(mystr, dataType){
  mystr # = "2020-01-17-Line9-bnd0-rtPos_arr_adjTtable-1556_files-pc_9-200928.csv"
  
  # 파일확장명과 파일생성날짜 규칙을 고려하면 뒤에서 11번째 자리에 pc이름 관련정보의 끝 정보가 있음
  start <- str_locate(mystr, paste0(dataType,"_"))[2]
  filesStrLoc <- str_locate(mystr, paste0("_files"))[1]-1
  end <- start+str_locate(substr(mystr, start, filesStrLoc), "-")[1]-1   # start = mystr.find("_",end-4,end)+1 : python의 .find는 end-4, end사이에서 검색한 뒤 mystr 전체 index를 기준으로 값을 return : 1을 더 뺀건 python과 r의 substr과 python의 variable[from:to] 의 indexing 방식 차이 때문임
  
  if (is.na(start)==T) {
    print(paste0("Error had been occurred: ", mystr))
    return ("err")
    
  } else if (start > 0) {
    myInputDataType = substr(mystr, start+1, end-1) # myPcNm = mystr[start:end]
    #print(paste0(myInputData))
    
    # inputData 형식이 logTable인지 확인하기 위해 한번더 확인
    end2 <- str_locate(myInputDataType, "_ttable")[1]-1 #mystr.find("-files")
    if (is.na(end2)==T) {
      end3 <- str_locate(myInputDataType, "_adjTtable")[1]-1 #mystr.find("-files") 
      if (is.na(end3)==T) {
        return (myInputDataType)
      } else {
        return ("adjLogTable")
      }
    } else {
      return ("logTable")
    }
    
  } else {
    print(paste0('pcNm has out of range:' , mystr))
    return ("err")
  }
}

## ================================================================================

findTypeOfLogTable <- function(mystr){
  mystr# = "2019-10-13_Line1_rtTrainPos_merged_4356-files_utlsyslab_jh1_admin.csv"
  end <- str_locate(mystr, "_ttable")[1]-1 #mystr.find("-files")
  if (is.na(end)==F) {
    start <- end-5+str_locate(substr(mystr, end-5, end), "_")[1] #mystr.find("_",end-5,end)+1
    myLogTableType <- substr(mystr, start, end) #mystr[start:end]
    #print(paste0(myLogTableType,"_ttable"))
    return (myLogTableType)
  } else {
    #print(paste0("It's not logTable file: ", mystr))
    return (NA)
  }
}

## ================================================================================

findTypeOfAdjLogTable <- function(mystr){
  mystr# = "2019-10-13_Line1_rtTrainPos_merged_4356-files_utlsyslab_jh1_admin.csv"
  end <- str_locate(mystr, "_adjTtable")[1]-1 #mystr.find("-files")
  if (is.na(end)==F) {
    start <- end-5+str_locate(substr(mystr, end-5, end), "_")[1] #mystr.find("_",end-5,end)+1
    myLogTableType <- substr(mystr, start, end) #mystr[start:end]
    #print(paste0(myLogTableType,"_ttable"))
    return (myLogTableType)
  } else {
    #print(paste0("It's not logTable file: ", mystr))
    return (NA)
  }
}

## ================================================================================

findNumberOfFiles <- function(mystr){
  mystr# = "2019-10-13_Line1_rtTrainPos_merged_4356-files_utlsyslab_jh1_admin.csv"
  end <- str_locate(mystr, "-files")[1]-1 #mystr.find("-files")
  start <- end-5+str_locate(substr(mystr, end-5, end), "_")[1] #mystr.find("_",end-5,end)+1
  myNfiles <- substr(mystr, start, end) #mystr[start:end]
  #print(paste0(myNfiles,"_files"))
  return (myNfiles)
}

## ================================================================================

checkRawdataFileList <- function(lineNm, dataType, mydir, dataByDayByPc) {
  rawFiles = length(mydir) # 파일 갯수 추출
  print(paste0("Total number of data in the folder: ",rawFiles))
  
  # 2대 이상의 pc에서 수집된 일일 데이터를 병합하기 위해 날짜 별 데이터 정보를 정리하는 코드 생성
  
  date <- 1
  for (date in 1:rawFiles) {
    
    # 주어진 column 이름에 대해 빈 데이터프레임 만들기
    # 참고 : https://specialscene.tistory.com/43
    tempData = data.frame()
    #colnames(tempData) = c('fDate', 'dataType', 'lineNm', 'pcNm', 'nFiles')
    
    # 폴더 내 파일이름으로 부터 정보를 추출
    oDate = substr(mydir[date], 0,10) # YYYY-MM-DD
    myLineNm <- chkLineNm(mydir[date], lineNm)
    if (myLineNm != "err") {
      myInputDataType <- findTypeOfInputData(mydir[date], dataType) # 입력 데이터 형식 확인
      myLogTableType <- findTypeOfLogTable(mydir[date]) # 입력데이터가 실시간 접근|도착|출발시각표라면 그 종류를 저장
      myAdjLogTableType <- findTypeOfAdjLogTable(mydir[date]) # 입력데이터가 실시간 접근|도착|출발시각표라면 그 종류를 저장
      myBnd <- findBndId(mydir[date])
      myDirectAt <- findDirectAtId(mydir[date])
      
      
    } else {
      myInputDataType <- "err"
      myLogTableType <- "err"
      myBnd <- "err"
      myDirectAt <- "err"
      
    }
    # 추출한 정보를 임시 데이터프레임에 저장
    tempData <- data.frame('oDate'=oDate, 'lineNm'=myLineNm, 'bnd'=myBnd, 'directAt'=myDirectAt ,'inputDataType'=myInputDataType, 'logTableType'=myLogTableType, 'adjLogTableType'=myAdjLogTableType)
    dataByDayByPc <- rbind(dataByDayByPc, tempData) #pd.concat([dataByDayByPc, tempData], axis=0) 
  } 
  
  #dataByDayByPc.reset_index(drop=True, inplace=True)
  #print(dataByDayByPc)
  return (dataByDayByPc)
}

# ===================================

#myLogTable <- as.data.frame(readLogTableByBndByDirectAt(date, lineNm, bnd, directAt, logTableType))

#myDate <- date
#tempLineNm <- lineNm
#tempBnd <- bnd
#tempDirectAt <- directAt
#tempLogTableType <- logTableType

readLogTableByBndByDirectAt <- function(myDate, tempLineNm, tempBnd, tempDirectAt, tempLogTableType) {
  candidateFile <- vector()
  candidateFile <- which(dataByDayByPc$oDate==as.character(myDate) & 
                           dataByDayByPc$lineNm==tempLineNm &
                           dataByDayByPc$bnd==tempBnd &
                           dataByDayByPc$directAt==tempDirectAt & 
                           dataByDayByPc$inputDataType == "logTable" &
                           dataByDayByPc$logTableType == tempLogTableType)
  
  if (length(candidateFile)==1) {
    # 파일 읽기
    
    if(tempLineNm==lineNm){
      myLogTable <- fread(paste0(rtPosResDataPath,"/",mydir[candidateFile]),tz = "")
    }else if(tempLineNm==lineNm_sub){
      myLogTable <- fread(paste0(rtPosResDataPathLineSub,"/",mydir[candidateFile]),tz = "")
    }
    
    return(myLogTable)
    
  } else if (length(candidateFile)>1) {
    print(paste0("There are at least 2 logTable data: "))
    dataByDayByPc[candidateFile, ]
    
  } else {
    print(paste0("There is no matched logTable data: "))
  }
  
}

# ===================================

#myLogTable <- as.data.frame(readLogTableByBndByDirectAt(date, lineNm, bnd, directAt, logTableType))

#myDate <- date
#tempLineNm <- lineNm
#tempBnd <- bnd
#tempDirectAt <- directAt
#tempLogTableType <- logTableType

chkLogTableOrderInDir <- function(myDate, tempLineNm, tempBnd, tempDirectAt, tempLogTableType) {
  candidateFile <- vector()
  
  if(tempLineNm == lineNm){
    candidateFile <- which(dataByDayByPc$oDate==as.character(myDate) & 
                             dataByDayByPc$lineNm==lineNm &
                             dataByDayByPc$bnd==tempBnd &
                             dataByDayByPc$directAt==tempDirectAt & 
                             dataByDayByPc$inputDataType == "logTable" &
                             dataByDayByPc$logTableType == tempLogTableType)
    
  }else if(tempLineNm == lineNm_sub){
    candidateFile <- which(dataByDayByPc$oDate==as.character(myDate) & 
                             dataByDayByPc$lineNm==lineNm_sub &
                             dataByDayByPc$bnd==tempBnd &
                             dataByDayByPc$directAt==tempDirectAt & 
                             dataByDayByPc$inputDataType == "logTable" &
                             dataByDayByPc$logTableType == tempLogTableType)
    
  }
  
  if (length(candidateFile)==1) {
    return(candidateFile)
    
  } else if (length(candidateFile)>1) {
    print(paste0("There are at least 2 logTable data: "))
    dataByDayByPc[candidateFile, ]
    
  } else {
    print(paste0("There is no matched logTable data: "))
  }
  
}

## ================================================================================

#as.data.frame(readrtDataByBndByDirectAt(myDate=date, tempLineNm=lineNm, tempBnd=bnd, tempDirectAt=directAt, rtDataType="depSeqByStn"))

#myDate <- date
#tempLineNm <- lineNm
#tempBnd <- bnd
#tempDirectAt <- directAt
#rtDataType <- "depSeqByStn"

readrtDataByBndByDirectAt <- function(myDate, tempLineNm, tempBnd, tempDirectAt, rtDataType) {
  candidateFile <- vector()
  candidateFile <- which(dataByDayByPc$oDate==as.character(myDate) & 
                           dataByDayByPc$lineNm==tempLineNm &
                           dataByDayByPc$bnd==tempBnd &
                           dataByDayByPc$directAt==tempDirectAt & # 현재는 rtLogTable을 만드는 파이썬 코드에서 9호선 급행,완행 시각표를 구분하지 않고 G(일반)으로 통일해서 파일을 저장하고 있음. 이 때문에 파일 읽는 과정에서는 G, E 구분이 되지 않고있으며 G만 작동 가능함. 
                           dataByDayByPc$inputDataType == rtDataType)
  
  if (length(candidateFile)==1) {
    
    # 파일 읽기
    if (( (rtDataType == "depSeqByStn") | (rtDataType == "arrSeqByStn") | 
          (rtDataType == "depSeqByOvtStn") | (rtDataType == "arrSeqByOvtStn") | 
          (rtDataType == "arrDepSeqByOvtStn") | (rtDataType == "seqErrInfoByOvtStn")) == TRUE) {
      
      if(tempLineNm==lineNm){
        myLogTable <- fread(paste0(rtPosResDataPath,"/",mydir[candidateFile]), header=TRUE,tz = "") # 실시간 출도착순서 관련 데이터면 첫 행을 헤더정보로 읽음  
      }else if(tempLineNm==lineNm_sub){
        myLogTable <- fread(paste0(rtPosResDataPathLineSub,"/",mydir[candidateFile]), header = TRUE,tz = "")
      }
      
    } else if (( (rtDataType == "logTable") | (rtDataType == "adjLogTable") ) == TRUE) {
      print(paste0("실시간 출도착시각표 정보는 별도의 readLogTableByBndByDirectAt, readAdjLogTableByBndByDirectAt 함수를 사용해주세요. 현재 dataType:",dataType," | oDate: ",oDate," | ",lineNm," | ", bnd," | ",Sys.time()))
      
    } else {
      
      if(tempLineNm==lineNm){
        myLogTable <- fread(paste0(rtPosResDataPath,"/",mydir[candidateFile]), header=TRUE,tz = "") # 실시간 출도착순서 관련 데이터면 첫 행을 헤더정보로 읽음  
      }else if(tempLineNm==lineNm_sub){
        myLogTable <- fread(paste0(rtPosResDataPathLineSub,"/",mydir[candidateFile]), header = TRUE,tz = "")
      }
      
    }
    
    return(myLogTable)
    
  } else if (length(candidateFile)>1) {
    print(paste0("There are at least 2 ",dataType," data: "))
    dataByDayByPc[candidateFile, ]
    
  } else {
    print(paste0("There is no matched ",dataType," data: "))
  }
  
}


# ===================================

#library(KoNLP)
# library(reshape2)
# library(data.table)
# library(bit64)
# library(stringr)
# library(lubridate)
# options(scipen=100)

TimeDiffSwitch <- FALSE


# ===================================
#####          M A I N          #####
# ===================================


# 분석대상노선 정보 지정하기
# lineNm <- "Line9" #lineNm_list[l]#Available : LineS or Line9
# bndNm <- "bnd1"
directAt <- "G"           # 현재는 rtLogTable을 만드는 파이썬 코드에서 9호선 급행,완행 시각표를 구분하지 않고 G(일반)으로 통일해서 파일을 저장하고 있음. 이 때문에 파일 읽는 과정에서는 G, E 구분이 되지 않고있으며 G만 작동 가능함.
dataType = "rtPos"

if (lineNm == "Line9") {
  myline <- "9호선" #Available : 9호선 or 신분당선
  linename <- "L9" #Available : L9 or SBD
  stn_id <- c(4101:4138)

} else if (lineNm == "LineS") {
  myline <- "신분당선" #Available : 9호선 or 신분당선
  linename <- "SBD" #Available : L9 or SBD
  stn_id <- c(4307:4319)

} else if (lineNm=="LineWS") {
  myline <- "우이신설선" #Available : 9호선 or 신분당선
  linename <- "WS" #Available : L9 or SBD
  stn_id <- c(4701:4713)
  
} else {
  print(paste0("현재 노선명이 9호선이나 신분당선, 우이신설선이 아닙니다. current lineNm:",lineNm))
}


# ===================================

# # 추출하고자하는 열차 노선 관련 정보 지정
# myline_list <- c("9호선") #c("9호선","신분당선")
# lineNm_list <- c("Line9") #,"LineS")
# linename_list <- c("L9") #,"SBD")
# Line_start_list <- c(4100,4300)
# Line_end_list <- c(4199,4399)
# Linesub_start_list <- c(4200,9999)
# Linesub_end_list <- c(4299,9999)
# 
# lineNm_sub <- "LineA" # 9호선 전용 변수 : 김포공항 평면환승 보정을 위해 공항철도 정보를 사용해야 함
# 
# 
# # 9호선(급행운영노선) 추월역 관련련 변수 지정
# if (((lineNm=="Line9") & (bndNm=="bnd1"))==TRUE) {
#   passedstn <- c(4105,4107,4112,4116,4120,4124,4125,4128,4131,4134)#피추월이 일어나는 역  
# } else {
#   passedstn <- c(4105,4107,4112,4116,4120,4124,4128,4131,4134)#피추월이 일어나는 역  
# }
# 
# # 급행 정차역 관련 변수 지정
# ExpStopStn <- c(4102,4105,4107,4110,4113,4115,4117,4120,4123,4125,4127,4129,4130,4133,4136,4138)#급행열차 정차역

# delay_standard <- data.frame(nOvt=c(0,1,2,-99),delay_G=c(90,239,388,NA),delay_E=c(90,239,388,NA))#delay=c(50,140,230))
# # 피추월횟수 1회일때 230초는 피추월횟수 1회인 열차들의 역구내점유시간의 Q3(75percentile)인 239초를 사용


# 출도착로그 보정 시 적용하는 보정시간 (초, 일반역 및 급행정차역)

#occupied_calibrate_time <- 90

# 출도착로그 보정 시 적용하는 보정시간 (초, 급행열차 미정차역에)
# ExpPassStn_occupied_calibrate_time <- 13  # 11초는 급행열차 통과역의 역구내 점유시간의 75퍼센타일인 13초를 사용

#Update on : 2021-06-11
# 김포공항역(4102)에서 출발하는 급행열차의 출발시각을 도착시각으로부터 추정할 때는 120초를 적용함(윤석이형 의견 반영)  
E_TRNID_OccupiedTime_in_GimpoStn <- 120

# ===================================
# 
# # 기본경로 지정
# path <- getwd()
# 
# # T 드라이브의 wd 경로 지정
# tDriveDvidingTripPath <- "/mnt/TLSYSL-RD1-2/Smart_Card_Data/202103_수도권교통카드데이터_KRRI/3_wds/3-01_Dividing_Trips"
# 
# # 주요 경로 저장
# rtPos_path <- "/mnt/TLSYSL-RD1-2/06_seoulSubway_realtimeData/2_data/rtPosition"
# 
# # 1.3. 읽어들일 실시간 logData 파일 위치 설정
# if (dataType == "rtPos") {
#   dataTypeFull = "rtPosition"
# }else if (dataType == "rtArr"){
#   dataTypeFull = "rtArrival"
# } 
# 
# rtBasePath =  "/mnt/TLSYSL-RD1-2/06_seoulSubway_realtimeData"                                              #"C:/Users/ysoh0223/Desktop/2020W01-정차시간예측_200101/platformCrwodness"
# rawLogDataPath = paste0(basePath,"/1_rawData/",dataTypeFull,"/",lineNm)                         # 실시간 수집 데이터 raw 파일 위치
# logDataPath = paste0(basePath,"/2_data/",dataTypeFull,"/",lineNm)                               # 실시간 수집 데이터 전처리 완료 파일 위치 : dataType에 의해 가변하는 변수
# 
# #!!!!!!!!! LogTable폴더 명이 바뀔 경우, 경로 수정 필요!!
# rtPosResDataPath = paste0(basePath,"/2_data","/rtPosition/",lineNm,"/oDate-20210301-20210331")  # 실시간 수집 데이터 전처리 완료 파일 위치 : rtPos 기반 데이터 
# rtPosResDataPathLineSub = paste0(basePath,"/2_data","/rtPosition/",lineNm_sub,"/oDate-20210301-20210331")

#rtArrResDataPath = paste0(basePath,"/2_data","/rtArrival/",lineNm)    
timeDiffDoorToATSPath = "/mnt/TLSYSL-RD1-2/11_subwayOperation_doorOpenClose_KRRI-210430/3_timeDiff_doorTime_trainOper/04_statisticAnalysis_results"


# 이 코드의 실행 결과로 실행되는 파일의 파일생성일 날짜
# fwDate <- "210616"#"201019" # paste0(strftime(Sys.Date(), format="%y%m%d")) #"191231" # 이 코드에서 결과로 도출될 파일들에 사용될 버전 날짜

# ===================================


# 1.8. 공휴일 정보 벡터 만들기
holidays <- data.frame("oDate"=c("2019-08-15","2019-09-12", "2019-09-13", "2019-09-14", "2019-10-03", "2019-10-09", "2019-12-25", "2020-01-01","2020-01-24", "2020-01-25","2020-01-27","2020-04-15","2020-04-30","2020-05-05","2020-06-06"),
                       "description"=c("광복절",    "추석연휴",    "추석연휴",    "추석연휴",    "개천절",     "한글날",      "성탄절",      "신정",      "설연휴",     "설연휴",     "대체휴일", "21대국회의원선거","석가탄신일","어린이날","현충일"))


# 2.1. 읽어올 데이터 경로의 파일 정보에 대한 dataframe 만들기
mydir <- dir(paste0(rtPosResDataPath)) 
mydir <- mydir[endsWith(mydir, ".csv")]

dataByDayByPc = data.frame()
dataByDayByPc = checkRawdataFileList(lineNm, dataType, mydir, dataByDayByPc)

# 2.2. 읽어야 할 날짜 정보를 정리
# 2.2.1. Directory 안의 파일들의 날짜에 대한 고유값을 도출
dataObsDates <- unique(dataByDayByPc$oDate) #dataByDayByPc['oDate'].unique()
oDateAdjLogList <- dataObsDates


# 2.3. 날짜별로 logTable파일 읽기
# 2.3.1. 읽어들일 logTable 종류 지정
logTableTypes <- c("arr", "dep")
bnds <- c("bnd0", "bnd1") #, "bznd1")

if(TimeDiffSwitch == TRUE){
  # 2.3.2. 출입문 개폐시간 ~ ATS 지상자간 시간 차이 통계자료 불러오기
  timeDiffGeneral <- read.csv(paste0(timeDiffDoorToATSPath,"/timeDiff-line9-G-mode.csv"))
  names(timeDiffGeneral)[-(1:2)] <- substr(names(timeDiffGeneral)[-(1:2)],2,5)
  
  timeDiffExpress <- read.csv(paste0(timeDiffDoorToATSPath,"/timeDiff-line9-E-mode.csv"))
  names(timeDiffExpress)[-(1:2)] <- substr(names(timeDiffExpress)[-(1:2)],2,5)
}

v <- 1
for(v in 1:length(oDateAdjLogList)){
  
  ###### 입력 부분 ######
  oDates <- oDateAdjLogList[v]
  oDateAdjLog <- paste0(substr(oDates,3,4),substr(oDates,6,7),substr(oDates,9,10)) # 분석에 사용할 데이터의 관측일
  
  ## 폴더 내 파일 명 확인해서 원하는 날짜의 bnd0, bnd1 열차 출발,도착로그 가져오기 (2020-06-24)
  oDateAdjLog_char <- ymd(oDateAdjLog,tz = 'Asia/Seoul')
  oDateAdjLog_char <- as.character(oDateAdjLog_char) 

  stnID_table <- read.csv(paste0(rtBasePath,"/8_rtData_crawling/2020W01-05-04_실시간위치_도착정보_수집/2019W03-03-04-02_Crawling","/자료출처별_역코드_매칭테이블-200618.csv"),fileEncoding = "EUC-KR")
  
  stnID_table_main <- stnID_table[(stnID_table$lineNmKor %in% myline),]
  
  stn_id <- as.integer(as.character(stnID_table_main$tcdStatnId))
  
  n_sta <- length(stn_id)#max(length(unique(get(paste0(lineNm,"_bnd0"))$CARDSTNID)), length(unique(get(paste0(lineNm,"_bnd1"))$CARDSTNID)))
  sta_id <- stn_id#unique(c(get(paste0(lineNm,"_bnd0"))$CARDSTNID, get(paste0(lineNm,"_bnd1"))$CARDSTNID))
  sta_id <- stn_id[order(stn_id)]#sta_id[order(sta_id)]
  
  
  # 양방향 출도착 정보, 피추월횟수 읽어오기 및 변수 할당하기
  bnd <- bnds[1]
  for (bnd in bnds) {
    
    logTableType <- logTableTypes[1]
    for (logTableType in logTableTypes) {
      
      # 현재 파일이름 정보 저장 (adjLogTable 저장할 때 수집 pc 관련 정보를 유지하기 위해)
      currentFileNmStr <- mydir[chkLogTableOrderInDir(myDate=oDateAdjLog_char, tempLineNm=lineNm, tempBnd=bnd, tempDirectAt=directAt, tempLogTableType=logTableType)]
      currentFileVarNm <- paste0(logTableType,"_table_Nm_",bnd)
      assign(currentFileVarNm, currentFileNmStr)
      
      # 2.3.2. 변수이름 생성
      
      logTableNm <- paste0(logTableType,"LogTable_",bnd)
      shortLogTableNm <- paste0(logTableType,"_log_",bnd)
      trnInfoTableNm <- paste0(logTableType,"_log_",bnd,"_trnInfo")
      
      # 2.3.3. 날짜, 노선명, 방향, 급행여부, logTable 종류에 따라 logTable을 읽기
      # 2.3.3.1. 빈 데이터프레임 만들기 readLogTableByBndByDirectAt 함수를 이용해 
      myLogTable <- data.frame()
      
      # 2.3.3.5. adjLogTable이 없는 경우 최초 수집된 시각표를 사용
      myLogTable <- as.data.frame(readLogTableByBndByDirectAt(oDateAdjLog_char, lineNm, bnd, directAt, logTableType))
       
      # 2.3.4. 열 정보 변경
      # 2.3.4.1 trainNo2 열 이름을 TRNID로 변경
      colnames(myLogTable)[which(colnames(myLogTable)=="trainNo2")] <- "TRNID"
      
      if (lineNm == "LineS") {
        # 2.3.4.2 신분당선 : 열차번호가 3자리인 열차를 4자리로 변경하고, 역 코드체계를 CARDSTNID 4자리 character로 변경
        myLogTable$TRNID <- sapply(myLogTable$TRNID, function (x) {ifelse(nchar(x)==3,paste0("0",x),paste(x))})
        colnames(myLogTable)[8:ncol(myLogTable)] <- as.character(as.numeric(colnames(myLogTable)[8:ncol(myLogTable)])-2500) #as.character(seq(4307,4319,by=1)) #as.character(as.numeric(colnames(myLogTable)[8:ncol(myLogTable)]))
        # operDistDF <- read.csv(paste0(rtBasePath,"/3_operDistCsv/Oper_Distance_LineS-200419.csv"),fileEncoding = "EUC-KR")
        
      } else if (lineNm == "Line9") {
        # 2.3.4.3. 9호선 : 교통카드 역 코드 체계로 변경
        # 2.3.4.3.1. operDist 파일 활용하기 위해 파일 읽기
        operDistDF <- read.csv(paste0(rtBasePath,"/3_operDistCsv/Oper_Distance_Line9-210621.csv"),fileEncoding = "EUC-KR")
        
        # 2.3.4.3.2. operDist의 statnId열(rtPos의 역 ID)을 statnId2(tcd의 역 ID)로 변경
        colnames(myLogTable)[8:ncol(myLogTable)] <- sapply(substr(colnames(myLogTable)[8:ncol(myLogTable)],2,99), 
                                                           function(x) { operDistDF$statnId2[which(substr(operDistDF$statnId,start = 2,stop = 99) == x)]  }) 
      }
      
      # 2.3.5. logTable의 시간 정보를 POSIXct 형태로 변경
      myLogTable[,c(8:ncol(myLogTable))] <- lapply(myLogTable[,c(8:ncol(myLogTable))], 
                                                         function(x) {as.POSIXct(x, format="%Y-%m-%d %H:%M:%S")} )
      # 2.3.6.생성한 변수이름에 읽은 파일 할당
      assign(logTableNm, myLogTable)
      
      # 2.3.7. 성종이가 짠 코드와 연계할 변수 정의 및 변수 할당
      shortLogTable <- myLogTable[,c(1,8:length(myLogTable[1,]))] # 열차 시각 정보만 저장한 부분
      trnInfoTable <- myLogTable[,c(1:7)]                      # 열차의 ID등의 정보만 저장한 부분
      
      assign(shortLogTableNm, shortLogTable)                    # 열차정보 제외한 시각정보 저장
      assign(trnInfoTableNm, trnInfoTable)                      # 열차정보 저장

      # 2.3.8. 임시변수 삭제
      rm(logTableNm, myLogTable, shortLogTableNm, shortLogTable, trnInfoTableNm, trnInfoTable, currentFileVarNm, currentFileNmStr)
      
      # 2.3.9. 방향 별 피추월횟수 읽기
      if (lineNm=="Line9"){ # bnd for문 안에 포함시킴
        myOvtTableByStn <- as.data.frame(readrtDataByBndByDirectAt(myDate=oDateAdjLog_char, tempLineNm=lineNm, tempBnd=bnd, tempDirectAt=directAt, rtDataType="numOvertakeTable"))
        colnames(myOvtTableByStn)[which(colnames(myOvtTableByStn)=="trainNo2")] <- "TRNID"
        colnames(myOvtTableByStn)[8:ncol(myOvtTableByStn)] <- sapply(substr(colnames(myOvtTableByStn)[8:ncol(myOvtTableByStn)],2,99),
                                                                     function(x) { operDistDF$statnId2[which(substr(operDistDF$statnId,start = 2,stop = 99) == x)]  })
        myOvtTableByStn <- myOvtTableByStn[,c(1,8:ncol(myOvtTableByStn))]
        
        myOvtTableNm <- paste0("overtake_",bnd)  # 변수 이름 생성성
        assign(myOvtTableNm, myOvtTableByStn)    # 변수 할당
        rm(myOvtTableNm, myOvtTableByStn)        # 변수 삭제
      }
    } # for (logTableType in logTableTypes) 
    
  }
  
  # 2.3.9. 방향 별 피추월횟수 파일이 없는 경우, 파일 만들기
  if (lineNm!="Line9"){ # bnd for문 종료 후에 해당 문구 추가
    overtake_bnd0 = dep_log_bnd0
    overtake_bnd1 = dep_log_bnd1
    overtake_bnd0[,2:ncol(overtake_bnd0)] = 0
    overtake_bnd1[,2:ncol(overtake_bnd1)] = 0
  }
  
  
  ####################### 2020-06-29 월요일부터 시작 ########################
  ##### 열차 출발, 도착 로그 테이블 결측치보정!! ####
  #### !!! 현재 코드는 전체 테이블에 대해 NA의 위치를 참고하여 보정하는 형식임. 따라서, 출발 로그와 도착로그의 행수가 맞지않으면, 코드가 제대로 작동하지 않으므로
  #### !!! 출발 로그와 도착 로그 테이블의 행이 다른 날짜가 있는 경우, 결측값 수정을 열차별로 하도록함.
  
  n_trn_dep_bnd0 <- nrow(dep_log_bnd0)
  n_trn_arr_bnd0 <- nrow(arr_log_bnd0)
  n_trn_dep_bnd1 <- nrow(dep_log_bnd1)
  n_trn_arr_bnd1 <- nrow(arr_log_bnd1)
  n_trn_overtake_bnd0 <- nrow(overtake_bnd0)
  n_trn_overtake_bnd1 <- nrow(overtake_bnd1)
  

  ## 한쪽방향만 대피선이 있는 역이 존재하는 경우, 데이터 프레임을 맞춰주기 위해 피추월횟수가 0인 컬럼을 넣어줌
  ### Update on : 2021-08-06 성종
  
  #BND1 먼저 확인
  dummyStnBnd1 <- names(overtake_bnd0)[!(names(overtake_bnd0) %in% names(overtake_bnd1))]
  # overtake_bnd1 <- overtake_bnd1[,-ncol(overtake_bnd1)]  
  if(length(dummyStnBnd1)>0){
    overtake_bnd1[,dummyStnBnd1] <- 0
  }
  #BND0 확인
  dummyStnBnd0 <- names(overtake_bnd1)[!(names(overtake_bnd1) %in% names(overtake_bnd0))]
  
  if(length(dummyStnBnd0)>0){
    overtake_bnd0[,dummyStnBnd0m] <- 0
  }
  
  
  if( (((n_trn_dep_bnd0 == n_trn_arr_bnd0))&((n_trn_dep_bnd1 == n_trn_arr_bnd1)))&
     ((n_trn_dep_bnd0 == n_trn_overtake_bnd0)&(n_trn_dep_bnd1 == n_trn_overtake_bnd1)) ){
    ### 열차 출발 로그 테이블 결측치 보정
    #### 각각의 출발, 도착 로그 테이블에 결측값있는경우 상호보완하여 결측값을 보정
    
    
    dep_log_bnd0_rvsd <- dep_log_bnd0[order(dep_log_bnd0$TRNID),]
    arr_log_bnd0_rvsd <- arr_log_bnd0[order(arr_log_bnd0$TRNID),]
    dep_log_bnd0_rvsd_temp <- dep_log_bnd0_rvsd
    arr_log_bnd0_rvsd_temp <- arr_log_bnd0_rvsd
    
    dep_log_bnd1_rvsd <- dep_log_bnd1[order(dep_log_bnd1$TRNID),]
    arr_log_bnd1_rvsd <- arr_log_bnd1[order(arr_log_bnd1$TRNID),]
    dep_log_bnd1_rvsd_temp <- dep_log_bnd1_rvsd
    arr_log_bnd1_rvsd_temp <- arr_log_bnd1_rvsd
    
    overtake_bnd0_rvsd <- overtake_bnd0[order(overtake_bnd0$TRNID),]
    overtake_bnd1_rvsd <- overtake_bnd1[order(overtake_bnd1$TRNID),]

    
    #### 결측값(NA)의 위치를 찾은 후, 나머지 로그테이블에서 관측값 후 저장 
    #### Deleay Table 생성 후, 보정시간 적용                              2020-06-29 임의로 90초 사용
    #### (출발 시각 추정값) = (도착 시각 관측값) + (보정 시간)
    #### (도착 시각 추정값) = (출발 시각 관측값) - (보정 시간)
    
    dep_log_bnd0_NA <- is.na(dep_log_bnd0_rvsd)
    arr_log_bnd0_NA <- is.na(arr_log_bnd0_rvsd)
    
    dep_log_bnd1_NA <- is.na(dep_log_bnd1_rvsd)
    arr_log_bnd1_NA <- is.na(arr_log_bnd1_rvsd)
    
    
    dep_log_bnd0_rvsd_temp[dep_log_bnd0_NA] <- arr_log_bnd0_rvsd[dep_log_bnd0_NA] 
    arr_log_bnd0_rvsd_temp[arr_log_bnd0_NA] <- dep_log_bnd0_rvsd[arr_log_bnd0_NA]
    
    dep_log_bnd1_rvsd_temp[dep_log_bnd1_NA] <- arr_log_bnd1_rvsd[dep_log_bnd1_NA] 
    arr_log_bnd1_rvsd_temp[arr_log_bnd1_NA] <- dep_log_bnd1_rvsd[arr_log_bnd1_NA]
    
    
    dep_log_bnd0_rvsd <- dep_log_bnd0_rvsd_temp
    arr_log_bnd0_rvsd <- arr_log_bnd0_rvsd_temp
    
    dep_log_bnd1_rvsd <- dep_log_bnd1_rvsd_temp
    arr_log_bnd1_rvsd <- arr_log_bnd1_rvsd_temp
    

    ## 방향별로 열차타입 구분하여 피추월 횟수별 열차타입별 보정시간 계산
    {
      
      ## 방향별 구분
      
      ##BND0
      {
        ### 열차타입 구분
        G_TRNID_bnd0 <- as.integer(arr_log_bnd0_trnInfo$TRNID[(arr_log_bnd0_trnInfo$directAt %in% 0)]) 
        E_TRNID_bnd0 <- as.integer(arr_log_bnd0_trnInfo$TRNID[(arr_log_bnd0_trnInfo$directAt %in% 1)]) 
        if (lineNm=="LineS"){
          G_TRNID_bnd0[G_TRNID_bnd0<1000] = paste0(0,G_TRNID_bnd0[G_TRNID_bnd0<1000])
          E_TRNID_bnd0[E_TRNID_bnd0<1000] = paste0(0,E_TRNID_bnd0[E_TRNID_bnd0<1000])
        }
        
        G_overtake_bnd0_rvsd_base <- overtake_bnd0_rvsd[(overtake_bnd0_rvsd$TRNID %in% G_TRNID_bnd0),]
        E_overtake_bnd0_rvsd_base <- overtake_bnd0_rvsd[(overtake_bnd0_rvsd$TRNID %in% E_TRNID_bnd0),]
        
        G_overtake_bnd0_rvsd <- G_overtake_bnd0_rvsd_base
        E_overtake_bnd0_rvsd <- E_overtake_bnd0_rvsd_base
        
        n_cur_col <- 2
        for(n_cur_col in 2:ncol(G_overtake_bnd0_rvsd)){
          temp_column <- G_overtake_bnd0_rvsd[,n_cur_col]
          temp_column[temp_column %in% 0] <- delay_standard$delay_G[delay_standard$nOvt == 0]
          temp_column[temp_column %in% 1] <- delay_standard$delay_G[delay_standard$nOvt == 1]
          temp_column[temp_column %in% 2] <- delay_standard$delay_G[delay_standard$nOvt == 2]
          temp_column[temp_column %in% -99] <- delay_standard$delay_G[delay_standard$nOvt == -99]
          
          G_overtake_bnd0_rvsd[,n_cur_col] <- temp_column
          
          rm(temp_column)
        }
        
        for(n_cur_col in 2:ncol(E_overtake_bnd0_rvsd)){
          temp_column <- E_overtake_bnd0_rvsd[,n_cur_col]
          temp_column[temp_column %in% 0] <- delay_standard$delay_E[delay_standard$nOvt == 0]
          temp_column[temp_column %in% 1] <- delay_standard$delay_E[delay_standard$nOvt == 1]
          temp_column[temp_column %in% 2] <- delay_standard$delay_E[delay_standard$nOvt == 2]
          temp_column[temp_column %in% -99] <- delay_standard$delay_E[delay_standard$nOvt == -99]
          
          E_overtake_bnd0_rvsd[,n_cur_col] <- temp_column
          
          rm(temp_column)
        }
        
        ### 계산 후 병합
        overtake_bnd0_calc <- rbind(G_overtake_bnd0_rvsd,E_overtake_bnd0_rvsd)
        overtake_bnd0_calc <- overtake_bnd0_calc[order(overtake_bnd0_calc$TRNID),]
        
      }##BND0
      
      ### 전체 역에 대해 보정시간표 계산
      {
        
        calibration_table_bnd0 <- matrix(nrow = nrow(dep_log_bnd0_rvsd) ,ncol = ncol(dep_log_bnd0_rvsd))
        calibration_table_bnd0 <- as.data.frame(calibration_table_bnd0)
        names(calibration_table_bnd0) <- names(dep_log_bnd0_rvsd)
        calibration_table_bnd0$TRNID <- overtake_bnd0_calc$TRNID
        
        cur_col<-2
        for(cur_col in 2:ncol(overtake_bnd0_calc)){
          
          temp_colname <- names(overtake_bnd0_calc)[cur_col]
          
          calibration_table_bnd0[,(names(calibration_table_bnd0) %in% temp_colname)] <- overtake_bnd0_calc[,cur_col]
          
        }
        
        {
          #### 급행 열차 : 급행 정차 X 역에 대해 별도의 보정시간부여
          
          ##### 급행미정차역
          ExpNonStopStn <- stn_id[-which(stn_id %in% ExpStopStn)]
          
          ##### 급행미정차역 & 추월 가능역
          ExpNonStopStn_Overtake <- ExpNonStopStn[(ExpNonStopStn %in% passedstn)]
          ##### 급행미정차역 & 추월 불가능역
          ExpNonStopStn_NonOvertake <- ExpNonStopStn[-which(ExpNonStopStn %in% passedstn)]
          
          ##### 급행정차역 & 추월 가능역
          ExpStopStn_Overtake <- ExpStopStn[(ExpStopStn %in% passedstn)]
          ##### 급행정차역 & 추월 불가능역
          ExpStopStn_NonOvertake <- ExpStopStn[-which(ExpStopStn %in% passedstn)]
          
          #length(ExpStopStn_Overtake) + length(ExpStopStn_NonOvertake)
          #length(ExpStopStn)
          #length(ExpNonStopStn_Overtake) + length(ExpNonStopStn_NonOvertake)
          #length(ExpNonStopStn)
          
          #급행 정차역 & 추월 가능역
          ## -> 피추월횟수에 따라 위에서 반영 완료
          
          # 급행 정차역 & 추월 불가능역
          {
            ## 급행열차
            
            ### BND0
            calibration_table_bnd0[(calibration_table_bnd0$TRNID %in% E_TRNID_bnd0),
                                   names(calibration_table_bnd0) %in% ExpStopStn_NonOvertake] <- delay_standard$delay_E[delay_standard$nOvt==0]  #occupied_calibrate_time
            
            ## 일반열차
            
            ### BND0
            calibration_table_bnd0[(calibration_table_bnd0$TRNID %in% G_TRNID_bnd0),
                                   names(calibration_table_bnd0) %in% ExpStopStn_NonOvertake] <- delay_standard$delay_G[delay_standard$nOvt==0]
            
          }
          
          # 급행 미정차역 & 추월 가능역
          {
            ## 급행열차
            
            ### BND0
            calibration_table_bnd0[(calibration_table_bnd0$TRNID %in% E_TRNID_bnd0),
                                   names(calibration_table_bnd0) %in% ExpNonStopStn_Overtake] <- ExpPassStn_occupied_calibrate_time
            
            ## 일반열차
            ## -> 피추월횟수에 따라 위에서 calibration_table을 만들면서 반영 완료 : G_overtake_bnd0_rvsd 생성하는 부분
            #calibration_table_bnd0[(calibration_table_bnd0$TRNID %in% G_TRNID_bnd0),names(calibration_table_bnd0) %in% ExpNonStopStn_Overtake]
            
            
          }
          
          # 급행 미정차역 & 피추월 불가능역
          
          {
            ## 급행열차
            
            ### BND0
            calibration_table_bnd0[(calibration_table_bnd0$TRNID %in% E_TRNID_bnd0),
                                   names(calibration_table_bnd0) %in% ExpNonStopStn_NonOvertake] <- ExpPassStn_occupied_calibrate_time
            
            ## 일반열차
            
            ### BND0
            calibration_table_bnd0[(calibration_table_bnd0$TRNID %in% G_TRNID_bnd0),
                                   names(calibration_table_bnd0) %in% ExpNonStopStn_NonOvertake] <- delay_standard$delay_G[delay_standard$nOvt==0]
            
            
          }
          
          
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% G_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpStopStn_Overtake)]
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% E_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpStopStn_Overtake)]
          
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% G_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpStopStn_NonOvertake)]
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% E_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpStopStn_NonOvertake)]
          
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% G_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpNonStopStn_Overtake)]
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% E_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpNonStopStn_Overtake)]
          
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% G_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpNonStopStn_NonOvertake)]
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% E_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpNonStopStn_NonOvertake)]
        }
        
      }### 전체 역에 대해 보정시간표 계산
      
      
      
      ##BND1
      {
        ### 열차타입 구분
        G_TRNID_bnd1 <- as.integer(arr_log_bnd1_trnInfo$TRNID[(arr_log_bnd1_trnInfo$directAt %in% 0)]) 
        E_TRNID_bnd1 <- as.integer(arr_log_bnd1_trnInfo$TRNID[(arr_log_bnd1_trnInfo$directAt %in% 1)]) 
        if (lineNm=="LineS"){
          G_TRNID_bnd1[G_TRNID_bnd1<1000] = paste0(0,G_TRNID_bnd1[G_TRNID_bnd1<1000])
          E_TRNID_bnd1[E_TRNID_bnd1<1000] = paste0(0,E_TRNID_bnd1[E_TRNID_bnd1<1000])
        }
        
        G_overtake_bnd1_rvsd_base <- overtake_bnd1_rvsd[(overtake_bnd1_rvsd$TRNID %in% G_TRNID_bnd1),]
        E_overtake_bnd1_rvsd_base <- overtake_bnd1_rvsd[(overtake_bnd1_rvsd$TRNID %in% E_TRNID_bnd1),]
        
        G_overtake_bnd1_rvsd <- G_overtake_bnd1_rvsd_base
        E_overtake_bnd1_rvsd <- E_overtake_bnd1_rvsd_base
        
        for(n_cur_col in 2:ncol(G_overtake_bnd1_rvsd)){
          temp_column <- G_overtake_bnd1_rvsd[,n_cur_col]
          temp_column[temp_column %in% 0] <- delay_standard$delay_G[delay_standard$nOvt == 0]
          temp_column[temp_column %in% 1] <- delay_standard$delay_G[delay_standard$nOvt == 1]
          temp_column[temp_column %in% 2] <- delay_standard$delay_G[delay_standard$nOvt == 2]
          temp_column[temp_column %in% -99] <- delay_standard$delay_G[delay_standard$nOvt == -99]
          
          G_overtake_bnd1_rvsd[,n_cur_col] <- temp_column
          
          rm(temp_column)
        }
        
        for(n_cur_col in 2:ncol(E_overtake_bnd1_rvsd)){
          temp_column <- E_overtake_bnd1_rvsd[,n_cur_col]
          temp_column[temp_column %in% 0] <- delay_standard$delay_E[delay_standard$nOvt == 0]
          temp_column[temp_column %in% 1] <- delay_standard$delay_E[delay_standard$nOvt == 1]
          temp_column[temp_column %in% 2] <- delay_standard$delay_E[delay_standard$nOvt == 2]
          temp_column[temp_column %in% -99] <- delay_standard$delay_E[delay_standard$nOvt == -99]
          
          E_overtake_bnd1_rvsd[,n_cur_col] <- temp_column
          
          rm(temp_column)
        }
        
        ### 계산 후 병합
        overtake_bnd1_calc <- rbind(G_overtake_bnd1_rvsd,E_overtake_bnd1_rvsd)
        overtake_bnd1_calc <- overtake_bnd1_calc[order(overtake_bnd1_calc$TRNID),]
        
      }##BND1
      
      ### 전체 역에 대해 보정시간표 계산
      {
        calibration_table_bnd1 <- matrix(nrow = nrow(dep_log_bnd1_rvsd) ,ncol = ncol(dep_log_bnd1_rvsd))
        calibration_table_bnd1 <- as.data.frame(calibration_table_bnd1)
        names(calibration_table_bnd1) <- names(dep_log_bnd1_rvsd)
        calibration_table_bnd1$TRNID <- overtake_bnd1_calc$TRNID
        
        cur_col<-2
        for(cur_col in 2:ncol(overtake_bnd1_calc)){
          
          temp_colname <- names(overtake_bnd1_calc)[cur_col]
          
          calibration_table_bnd1[,(names(calibration_table_bnd1) %in% temp_colname)] <- overtake_bnd1_calc[,cur_col]
          
        }
        
        {
          #### 급행 열차 : 급행 정차 X 역에 대해 별도의 보정시간부여
          
          ##### 급행미정차역
          ExpNonStopStn <- stn_id[-which(stn_id %in% ExpStopStn)]
          
          ##### 급행미정차역 & 추월 가능역
          ExpNonStopStn_Overtake <- ExpNonStopStn[(ExpNonStopStn %in% passedstn)]
          ##### 급행미정차역 & 추월 불가능역
          ExpNonStopStn_NonOvertake <- ExpNonStopStn[-which(ExpNonStopStn %in% passedstn)]
          
          ##### 급행정차역 & 추월 가능역
          ExpStopStn_Overtake <- ExpStopStn[(ExpStopStn %in% passedstn)]
          ##### 급행정차역 & 추월 불가능역
          ExpStopStn_NonOvertake <- ExpStopStn[-which(ExpStopStn %in% passedstn)]
          
          #length(ExpStopStn_Overtake) + length(ExpStopStn_NonOvertake)
          #length(ExpStopStn)
          #length(ExpNonStopStn_Overtake) + length(ExpNonStopStn_NonOvertake)
          #length(ExpNonStopStn)
          
          #급행 정차역 & 추월 가능역
          ## -> 피추월횟수에 따라 위에서 반영 완료
          
          # 급행 정차역 & 추월 불가능역
          {
            ## 급행열차
            
            ### BND1
            calibration_table_bnd1[(calibration_table_bnd1$TRNID %in% E_TRNID_bnd1),
                                   names(calibration_table_bnd1) %in% ExpStopStn_NonOvertake] <- delay_standard$delay_E[delay_standard$nOvt==0]
            
            
            ## 일반열차
            
            ### BND1
            calibration_table_bnd1[(calibration_table_bnd1$TRNID %in% G_TRNID_bnd1),
                                   names(calibration_table_bnd1) %in% ExpStopStn_NonOvertake] <- delay_standard$delay_G[delay_standard$nOvt==0]
            
            
          }
          
          # 급행 미정차역 & 추월 가능역
          {
            ## 급행열차
            
            ### BND1
            calibration_table_bnd1[(calibration_table_bnd1$TRNID %in% E_TRNID_bnd1),
                                   names(calibration_table_bnd1) %in% ExpNonStopStn_Overtake] <- ExpPassStn_occupied_calibrate_time
            
            
            ## 일반열차
            ## -> 피추월횟수에 따라 위에서 반영 완료
            #calibration_table_bnd0[(calibration_table_bnd0$TRNID %in% G_TRNID_bnd0),names(calibration_table_bnd0) %in% ExpNonStopStn_Overtake]
            
            
          }
          
          # 급행 미정차역 & 피추월 불가능역
          
          {
            ## 급행열차
            
            ### BND1
            calibration_table_bnd1[(calibration_table_bnd1$TRNID %in% E_TRNID_bnd1),
                                   names(calibration_table_bnd1) %in% ExpNonStopStn_NonOvertake] <- ExpPassStn_occupied_calibrate_time
            
            
            ## 일반열차
            
            ### BND1
            calibration_table_bnd1[(calibration_table_bnd1$TRNID %in% G_TRNID_bnd1),
                                   names(calibration_table_bnd1) %in% ExpNonStopStn_NonOvertake] <- delay_standard$delay_G[delay_standard$nOvt==0]
            
            
          }
          
          
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% G_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpStopStn_Overtake)]
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% E_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpStopStn_Overtake)]
          
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% G_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpStopStn_NonOvertake)]
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% E_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpStopStn_NonOvertake)]
          
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% G_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpNonStopStn_Overtake)]
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% E_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpNonStopStn_Overtake)]
          
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% G_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpNonStopStn_NonOvertake)]
          #calibration_table_bnd0[calibration_table_bnd0$TRNID %in% E_TRNID_bnd0,(names(calibration_table_bnd0) %in% ExpNonStopStn_NonOvertake)]
        }
        
      }### 전체 역에 대해 보정시간표 계산
    }
    
    
    
    
    
    
    #### 기존 결측값에 보정시간 적용
    
    ##### BND0
    if(sum(dep_log_bnd0_rvsd$TRNID != calibration_table_bnd0$TRNID)==0){
      if (lineNm=="LineS"){
        dep_log_bnd0_NA_calibrate <- dep_log_bnd0_NA[,2:ncol(dep_log_bnd0_NA)] * calibration_table_bnd0[,2:ncol(calibration_table_bnd0)]
        dep_log_bnd0_NA_calibrate <- cbind(TRNID=0,dep_log_bnd0_NA_calibrate)
      }else{
        dep_log_bnd0_NA_calibrate <- dep_log_bnd0_NA * calibration_table_bnd0
      }
    }else{
      print(paste0("HAS ERROR IN CALCULATING CALIBRATE TABLE"))
    }
    
    if(sum(arr_log_bnd0_rvsd$TRNID != calibration_table_bnd0$TRNID)==0){
      if (lineNm=="LineS"){
        arr_log_bnd0_NA_calibrate <- arr_log_bnd0_NA[,2:ncol(arr_log_bnd0_NA)] * calibration_table_bnd0[,2:ncol(calibration_table_bnd0)]
        arr_log_bnd0_NA_calibrate <- cbind(TRNID=0,arr_log_bnd0_NA_calibrate)
      }else{
        arr_log_bnd0_NA_calibrate <- arr_log_bnd0_NA * calibration_table_bnd0
      }
    }else{
      print(paste0("HAS ERROR IN CALCULATING CALIBRATE TABLE"))
    }
  
    ##### BND1
    if(sum(dep_log_bnd1_rvsd$TRNID != calibration_table_bnd1$TRNID)==0){
      if (lineNm=="LineS"){
        dep_log_bnd1_NA_calibrate <- dep_log_bnd1_NA[,2:ncol(dep_log_bnd1_NA)] * calibration_table_bnd1[,2:ncol(calibration_table_bnd1)]
        dep_log_bnd1_NA_calibrate <- cbind(TRNID=0,dep_log_bnd1_NA_calibrate)
      }else{
        dep_log_bnd1_NA_calibrate <- dep_log_bnd1_NA * calibration_table_bnd1
      }
      
      #  Update on : 2021-06-11
      # 김포공항역(4102)에서 출발하는 급행열차의 출발시각을 도착시각으로부터 추정할 때는 120초를 적용함(윤석이형 의견 반영)  
      # 출발시각 운영실적 테이블 중, 결측값이 있는 급행열차에만 새로운 보정시간 적용
      whichEtrnWhich <- which((calibration_table_bnd1$TRNID) %in% E_TRNID_bnd1)
      dep_log_bnd1_NA_calibrate$`4102`[whichEtrnWhich][!(dep_log_bnd1_NA_calibrate$`4102`[whichEtrnWhich] %in% 0)] <- E_TRNID_OccupiedTime_in_GimpoStn
      #dep_log_bnd1_NA_calibrate[102:110,]
      #test11 <-dep_log_bnd1_NA_calibrate$`4102`[whichEtrnWhich]
    }else{
      print(paste0("HAS ERROR IN CALCULATING CALIBRATE TABLE"))
    }
    
    if(sum(arr_log_bnd1_rvsd$TRNID != calibration_table_bnd1$TRNID)==0){
      if (lineNm=="LineS"){
        arr_log_bnd1_NA_calibrate <- arr_log_bnd1_NA[,2:ncol(arr_log_bnd1_NA)] * calibration_table_bnd1[,2:ncol(calibration_table_bnd1)]
        arr_log_bnd1_NA_calibrate <- cbind(TRNID=0,arr_log_bnd1_NA_calibrate)
      }else{
        arr_log_bnd1_NA_calibrate <- arr_log_bnd1_NA * calibration_table_bnd1
      }
    }else{
      print(paste0("HAS ERROR IN CALCULATING CALIBRATE TABLE"))
    }
    
    #dep_log_bnd0[2,1:10]
    #dep_log_bnd0_rvsd[2,1:10]
    
    #head(dep_log_bnd0_rvsd[,-1] + dep_log_bnd0_NA_calibrate[,-1])
    #head(dep_log_bnd0_rvsd[,-1])
    #head(dep_log_bnd0_NA_calibrate[,-1])
    
    
    ## 도착 추정시, 출발 시각 - 보정시간
    ## 출발 추정시, 도착 시각 + 보정시간
    
    dep_log_bnd0_rvsd[,-1] <- (dep_log_bnd0_rvsd[,-1] + dep_log_bnd0_NA_calibrate[,-1])
    
    arr_log_bnd0_rvsd[,-1] <- (arr_log_bnd0_rvsd[,-1] - arr_log_bnd0_NA_calibrate[,-1])
    
    dep_log_bnd1_rvsd[,-1] <- (dep_log_bnd1_rvsd[,-1] + dep_log_bnd1_NA_calibrate[,-1]) 
    arr_log_bnd1_rvsd[,-1] <- (arr_log_bnd1_rvsd[,-1] - arr_log_bnd1_NA_calibrate[,-1])
    
    #dep_log_bnd0_rvsd[9,]
    dep_log_bnd0_rvsd_backup <- dep_log_bnd0_rvsd
    arr_log_bnd0_rvsd_backup <- arr_log_bnd0_rvsd
    dep_log_bnd1_rvsd_backup <- dep_log_bnd1_rvsd
    arr_log_bnd1_rvsd_backup <- arr_log_bnd1_rvsd
    
    log_list <- c("dep_log_bnd0","arr_log_bnd0","dep_log_bnd1","arr_log_bnd1")
    #cur_log <- 1
    
    #### 출발, 도착 추정값이 해당 열차 시간 순서가 논리적으로 맞는지 확인 ( Sta A-1 time < Sta A time < Sta A+1 time)
    #cur_log <- 2
    
    #### 운행방향별 열차별로 오류 유형에 대해 정리
    #### -> 카드데이터와 매칭 완료 후, 카드데이터마다 매칭된 열차에 대한 오류 유형을 부여할 예정
    cur_log<-1
    for(cur_log in 1:length(log_list)){
      
      trn_err_type <- NULL
      
      temp_Nm_log <- log_list[cur_log]
      temp_bnd <- substr(temp_Nm_log,start = (nchar(temp_Nm_log)-3) ,stop = nchar(temp_Nm_log))
      
      TRNID <- get(paste0(temp_Nm_log,"_rvsd"))[,1] 
      temp_log_org <- get(paste0(temp_Nm_log))[,-1]
      temp_log_table <- get(paste0(temp_Nm_log,"_rvsd"))[,-1] 
      temp_log_NA_origin <- get(paste0(temp_Nm_log,"_NA"))[,-1] 
      
      # if(lineNm == "LineS"){
      #   if(temp_bnd=="bnd0"){
      #     bnd_min_which <- n_sta
      #     bnd_max_which <- 1
      #   }else{
      #     bnd_min_which <- 1
      #     bnd_max_which <- n_sta
      #   }
      # }else if(lineNm == "Line9"){
      #   if(temp_bnd=="bnd0"){
      #     bnd_min_which <- n_sta
      #     bnd_max_which <- 1
      #   }else{
      #     bnd_min_which <- 1
      #     bnd_max_which <- n_sta
      #   }
      # }
      
      ## 기존 시각표의 NA포함 여부 확인
      ## 보정된 시각표의 시간 흐름이 논리적으로 타당한지 확인
      # if(bnd_min_which != bnd_max_which){
        
        cur_trn_err <- 2
        
        for(cur_trn_err in 1:nrow(temp_log_table)){
          ### 열차 번호 가져오기
          temp_TRNID <- TRNID[cur_trn_err]
          
          ### 열차별 시각표상 NA가 있는지 확인 
          temp_NA_error <- ifelse((sum(is.na(temp_log_table[cur_trn_err,]))==0), 0, 1)
          
          ### 열차별 수정된 시각표가 시간흐름에 위배되지않는지 확인
          temp_trn_squence <- frank(x = as.numeric(temp_log_table[cur_trn_err,]),na.last = NA)
          
          ### 열차 로그가 모두 NA인 경우, 생략
          if(sum(temp_trn_squence)!=0){            
            max_sq <- max(temp_trn_squence)
            min_sq <- min(temp_trn_squence)
            
            temp_exct_sequence <- min_sq:max_sq
            
            temp_sequence_error <- ifelse((sum(temp_trn_squence != temp_exct_sequence)==0),0,1)
            
            temp_err_type <- (temp_NA_error)  +  (temp_sequence_error*2)
            
            
            temp_trn_err <- data.frame(BND = temp_bnd,TRNID = temp_TRNID,Including_NA = temp_NA_error, Sequence_error = temp_sequence_error, Error_type = temp_err_type)
            
            
            trn_err_type <- rbind(trn_err_type,temp_trn_err)
            
            rm(temp_TRNID,temp_NA_error,temp_trn_squence,max_sq,min_sq,temp_exct_sequence,temp_sequence_error,temp_err_type,temp_trn_err)
          }#if(sum(temp_trn_squence)!=0){     
          
        }#for(cur_trn_err in 1:nrow(temp_log_table)){
        
        assign(paste0(temp_Nm_log,"_err_type"),trn_err_type,)
        
        rm(trn_err_type)
        
        # 2020-07-02 이전 작업 || 열차 추정값으로 인해 시간 흐름 위배 경우가 생기는 경우, 추정값 제거
        # 시간 흐름의 위배 여부만 확인 후, 값 수정은 따로 진행하지 않기로함
        ## 이전 역의 시간 차이 계산
        #prev_temp_log_table <- temp_log_table[,-1]
        #prev_temp_log_table <- cbind(prev_temp_log_table,as.POSIXct(NA))
        #time_diff_table <- temp_log_table - prev_temp_log_table
        #
        #all_NA <- is.na(prev_temp_log_table) * is.na(temp_log_table) 
        #one_NA <- (is.na(prev_temp_log_table) * (!is.na(temp_log_table))) + ((!is.na(prev_temp_log_table)) * (is.na(temp_log_table))) 
        
        #both <- all_NA + one_NA
        
        #both[both >=2] 
        
        #time_diff_table[one_NA == 1]
        
        

      # }else if(bnd_min_which == bnd_max_which){
      #   
      #   print(paste0("HAS ERROR IN CHECKING THE TRAIN LOGICAL TIME SEQUENCE1 ||",temp_Nm_log))
      #   
      # }
      
      
      #assign((paste0(temp_Nm_log,"_rvsd")),cbind(TRNID,temp_log_table))
      
      ## 논리적 오류보다 NA 개수가 작으면 오류 출력
      ### 위 작업을 통해 논리적 오류를 제거하기 때문에 이전보다 NA가 더 많아짐
      #if(sum(is.na(get(paste0(temp_Nm_log,"_rvsd_backup")))) > sum(is.na(get(paste0(temp_Nm_log,"_rvsd"))))){
      #  
      #  print(paste0("HAS ERROR IN CHECKING THE TRAIN LOGICAL TIME SEQUENCE2 ||",temp_Nm_log))
      #  
      #  break()
      #}
      
      
    }#for(cur_log in 1:length(log_list)){
    
    
    ## 에러 발견된 데이터 rds로 저장
    ### !!!추정값 반영된 시각표를 기준으로 저장하였음
    err_save<-3
    for(err_save in 1:length(log_list)){
      temp_Nm_log <- log_list[err_save]
      which_temp_log_err_trn <- which((apply(get(paste0(temp_Nm_log,"_err_type"))[,-c(1,2)], MARGIN = 1, FUN = sum))!=0)
      saveRDS(get(paste0(temp_Nm_log,"_err_type")),file = paste0(rtPosResDataPath,"/",oDateAdjLog_char,"-",lineNm,"-",substr(temp_Nm_log,9,12),"-rtPos_",substr(temp_Nm_log,1,3),"TrnErrType-",fwDate,".RDS")) ## 출발, 도착로그 오류 유형 (승,하차 재차인원 만들때 사용)
      saveRDS(get(temp_Nm_log)[which_temp_log_err_trn,],file = paste0(rtPosResDataPath,"/",oDateAdjLog_char,"-",lineNm,"-",substr(temp_Nm_log,9,12),"-rtPos_",substr(temp_Nm_log,1,3),"ErrTrnLog-",fwDate,".RDS")) ## 1개이상의 오류가 발견되면 저장
      rm(temp_Nm_log,which_temp_log_err_trn)
    }
    
    ## 출발 및 도착 로그 에러종류 저장
    
    n_dep_bnd0_NA <- sum(dep_log_bnd0_err_type$Including_NA==1)
    n_dep_bnd0_sequence_err <- sum(dep_log_bnd0_err_type$Sequence_error==1)
    n_dep_bnd0_both <- sum(dep_log_bnd0_err_type$Error_type==3)
    n_dep_bnd0_all_recorded <- sum(dep_log_bnd0_err_type$Error_type==0)
    n_dep_bnd0_total_trn <- nrow(dep_log_bnd0_err_type)
    
    n_arr_bnd0_NA <- sum(arr_log_bnd0_err_type$Including_NA==1)
    n_arr_bnd0_sequence_err <- sum(arr_log_bnd0_err_type$Sequence_error==1)
    n_arr_bnd0_both <- sum(arr_log_bnd0_err_type$Error_type==3)
    n_arr_bnd0_all_recorded <- sum(arr_log_bnd0_err_type$Error_type==0)
    n_arr_bnd0_total_trn <- nrow(arr_log_bnd0_err_type)
    
    n_dep_bnd1_NA <- sum(dep_log_bnd1_err_type$Including_NA==1)
    n_dep_bnd1_sequence_err <- sum(dep_log_bnd1_err_type$Sequence_error==1)
    n_dep_bnd1_both <- sum(dep_log_bnd1_err_type$Error_type==3)
    n_dep_bnd1_all_recorded <- sum(dep_log_bnd1_err_type$Error_type==0)
    n_dep_bnd1_total_trn <- nrow(dep_log_bnd1_err_type)
    
    n_arr_bnd1_NA <- sum(arr_log_bnd1_err_type$Including_NA==1)
    n_arr_bnd1_sequence_err <- sum(arr_log_bnd1_err_type$Sequence_error==1)
    n_arr_bnd1_both <- sum(arr_log_bnd1_err_type$Error_type==3)
    n_arr_bnd1_all_recorded <- sum(arr_log_bnd1_err_type$Error_type==0)
    n_arr_bnd1_total_trn <- nrow(arr_log_bnd1_err_type)
    
    log_err_type_summary <- data.frame(type = c("Err1-including_NA","Err2-sequence_error","Err3-both","all_recorded","total"), depLog_bnd0_obs = c(n_dep_bnd0_NA,n_dep_bnd0_sequence_err,n_dep_bnd0_both,n_dep_bnd0_all_recorded,n_dep_bnd0_total_trn),depLog_bnd0_ratio = c(n_dep_bnd0_NA,n_dep_bnd0_sequence_err,n_dep_bnd0_both,n_dep_bnd0_all_recorded,n_dep_bnd0_total_trn)/n_dep_bnd0_total_trn, arrLog_bnd0_obs = c(n_arr_bnd0_NA,n_arr_bnd0_sequence_err,n_arr_bnd0_both,n_arr_bnd0_all_recorded,n_arr_bnd0_total_trn), arrLog_bnd0_ratio = c(n_arr_bnd0_NA,n_arr_bnd0_sequence_err,n_arr_bnd0_both,n_arr_bnd0_all_recorded,n_arr_bnd0_total_trn)/n_arr_bnd0_total_trn, depLog_bnd1_obs = c(n_dep_bnd1_NA,n_dep_bnd1_sequence_err,n_dep_bnd1_both,n_dep_bnd1_all_recorded,n_dep_bnd1_total_trn), depLog_bnd1_ratio = c(n_dep_bnd1_NA,n_dep_bnd1_sequence_err,n_dep_bnd1_both,n_dep_bnd1_all_recorded,n_dep_bnd1_total_trn)/n_dep_bnd1_total_trn, arrLog_bnd1_obs = c(n_arr_bnd1_NA,n_arr_bnd1_sequence_err,n_arr_bnd1_both,n_arr_bnd1_all_recorded,n_arr_bnd1_total_trn), arrLog_bnd1_ratio = c(n_arr_bnd1_NA,n_arr_bnd1_sequence_err,n_arr_bnd1_both,n_arr_bnd1_all_recorded,n_arr_bnd1_total_trn)/n_arr_bnd1_total_trn)
    
    print(paste0("SUCESS(",myline,") || oDate ",oDateAdjLog," || ",Sys.time()))
    write.csv(log_err_type_summary,file = paste0(rtPosResDataPath,"/",oDateAdjLog_char,"-",lineNm,"-","bndA","-rtPos_trnLog_errType_summary-",fwDate,".csv"),row.names = F)
    
    
    rm(n_arr_bnd0_all_recorded,n_arr_bnd0_both,n_arr_bnd0_NA,n_arr_bnd0_sequence_err,n_arr_bnd0_total_trn,n_dep_bnd0_all_recorded,n_dep_bnd0_both,n_dep_bnd0_NA,n_dep_bnd0_sequence_err,n_dep_bnd0_total_trn,n_arr_bnd1_all_recorded,n_arr_bnd1_both,n_arr_bnd1_NA,n_arr_bnd1_sequence_err,n_arr_bnd1_total_trn,n_dep_bnd1_all_recorded,n_dep_bnd1_both,n_dep_bnd1_NA,n_dep_bnd1_sequence_err,n_dep_bnd1_total_trn)
  }else{
    
    print(paste0("HAS ERROR IN REVISING NA VALUE FOR DEPARTURE || ARRIVAL LOG TABLE || OVERTAKE TABLE : oDate ",oDateAdjLog,"(",myline,")"))
    break()
  }
  
  
  #paste0(substr(x = arr_table_Nm_bnd0,start = 1,stop = 32),"adjTtable",substr(x = arr_table_Nm_bnd0,start = 39,stop = (nchar(arr_table_Nm_bnd0)-10)),fwDate,".csv")
  arr_log_bnd0_rvsd_res <- merge(arr_log_bnd0_trnInfo, arr_log_bnd0_rvsd, by.x="TRNID", by.y="TRNID")
  dep_log_bnd0_rvsd_res <- merge(dep_log_bnd0_trnInfo, dep_log_bnd0_rvsd, by.x="TRNID", by.y="TRNID")
  arr_log_bnd1_rvsd_res <- merge(arr_log_bnd1_trnInfo, arr_log_bnd1_rvsd, by.x="TRNID", by.y="TRNID")
  dep_log_bnd1_rvsd_res <- merge(dep_log_bnd1_trnInfo, dep_log_bnd1_rvsd, by.x="TRNID", by.y="TRNID")
  
  if(TimeDiffSwitch == TRUE){
    ## 실제 ATS 지상자와 출입문 개폐시간의 시간 차이 통계 데이터를 통해 출입문 개폐시간 추정에 사용 (9호선 열차별 역별 7일간 데이터)
    ## trueDwTime 관련
    ## 역별, 운행 방향별, 열차 등급별로 시간 차이가 다름
    ## 2021-05-26 성종 Update
    #tempStn <- "4101"
    
    stnList <- as.character(stnID_table_main$tcdStatnId)
    #tempStn <-"4103"
    for(tempStn in stnList){
      
      arr_log_bnd0_rvsd_res[(arr_log_bnd0_rvsd_res$TRNID %in% G_TRNID_bnd0),tempStn] <- arr_log_bnd0_rvsd_res[(arr_log_bnd0_rvsd_res$TRNID %in% G_TRNID_bnd0),tempStn] + timeDiffGeneral[((timeDiffGeneral$BND %in% "bnd0")&(timeDiffGeneral$TYPE %in% "arr")),tempStn]
      arr_log_bnd0_rvsd_res[(arr_log_bnd0_rvsd_res$TRNID %in% E_TRNID_bnd0),tempStn] <- arr_log_bnd0_rvsd_res[(arr_log_bnd0_rvsd_res$TRNID %in% E_TRNID_bnd0),tempStn]  + timeDiffExpress[((timeDiffExpress$BND %in% "bnd0")&(timeDiffExpress$TYPE %in% "arr")),tempStn]
      
      arr_log_bnd1_rvsd_res[(arr_log_bnd1_rvsd_res$TRNID %in% G_TRNID_bnd1),tempStn] <- arr_log_bnd1_rvsd_res[(arr_log_bnd1_rvsd_res$TRNID %in% G_TRNID_bnd1),tempStn] + timeDiffGeneral[((timeDiffGeneral$BND %in% "bnd1")&(timeDiffGeneral$TYPE %in% "arr")),tempStn]
      arr_log_bnd1_rvsd_res[(arr_log_bnd1_rvsd_res$TRNID %in% E_TRNID_bnd1),tempStn] <- arr_log_bnd1_rvsd_res[(arr_log_bnd1_rvsd_res$TRNID %in% E_TRNID_bnd1),tempStn]  + timeDiffExpress[((timeDiffExpress$BND %in% "bnd1")&(timeDiffExpress$TYPE %in% "arr")),tempStn]
      
      dep_log_bnd0_rvsd_res[(dep_log_bnd0_rvsd_res$TRNID %in% G_TRNID_bnd0),tempStn]  <- dep_log_bnd0_rvsd_res[(dep_log_bnd0_rvsd_res$TRNID %in% G_TRNID_bnd0),tempStn] - timeDiffGeneral[((timeDiffGeneral$BND %in% "bnd0")&(timeDiffGeneral$TYPE %in% "dep")),tempStn]
      dep_log_bnd0_rvsd_res[(dep_log_bnd0_rvsd_res$TRNID %in% E_TRNID_bnd0),tempStn]   <- dep_log_bnd0_rvsd_res[(dep_log_bnd0_rvsd_res$TRNID %in% E_TRNID_bnd0),tempStn] - timeDiffExpress[((timeDiffExpress$BND %in% "bnd0")&(timeDiffExpress$TYPE %in% "dep")),tempStn]
      
      dep_log_bnd1_rvsd_res[(dep_log_bnd1_rvsd_res$TRNID %in% G_TRNID_bnd1),tempStn] <- dep_log_bnd1_rvsd_res[(dep_log_bnd1_rvsd_res$TRNID %in% G_TRNID_bnd1),tempStn] - timeDiffGeneral[((timeDiffGeneral$BND %in% "bnd1")&(timeDiffGeneral$TYPE %in% "dep")),tempStn]
      dep_log_bnd1_rvsd_res[(dep_log_bnd1_rvsd_res$TRNID %in% E_TRNID_bnd1),tempStn] <- dep_log_bnd1_rvsd_res[(dep_log_bnd1_rvsd_res$TRNID %in% E_TRNID_bnd1),tempStn]  - timeDiffExpress[((timeDiffExpress$BND %in% "bnd1")&(timeDiffExpress$TYPE %in% "dep")),tempStn]
      
    }
  }
  ## 보정된 열차 로그 테이블은 급행열차 미정차역의 급행열차 로그가 기록되어있음
  write.csv(arr_log_bnd0_rvsd_res,file = paste0(rtPosResDataPath,"/",substr(x = arr_table_Nm_bnd0,start = 1,stop = 32+etcWord),"adjTtable",substr(x = arr_table_Nm_bnd0,start = 39+etcWord,stop = (nchar(arr_table_Nm_bnd0)-10)),fwDate,".csv"),row.names = F)
  write.csv(dep_log_bnd0_rvsd_res,file = paste0(rtPosResDataPath,"/",substr(x = dep_table_Nm_bnd0,start = 1,stop = 32+etcWord),"adjTtable",substr(x = dep_table_Nm_bnd0,start = 39+etcWord,stop = (nchar(dep_table_Nm_bnd0)-10)),fwDate,".csv"),row.names = F)
  
  write.csv(arr_log_bnd1_rvsd_res,file = paste0(rtPosResDataPath,"/",substr(x = arr_table_Nm_bnd1,start = 1,stop = 32+etcWord),"adjTtable",substr(x = arr_table_Nm_bnd1,start = 39+etcWord,stop = (nchar(arr_table_Nm_bnd1)-10)),fwDate,".csv"),row.names = F)
  write.csv(dep_log_bnd1_rvsd_res,file = paste0(rtPosResDataPath,"/",substr(x = dep_table_Nm_bnd1,start = 1,stop = 32+etcWord),"adjTtable",substr(x = dep_table_Nm_bnd1,start = 39+etcWord,stop = (nchar(dep_table_Nm_bnd1)-10)),fwDate,".csv"),row.names = F)

}

















## Update on : 2021-07-29
## 9호선인 경우, 공항철도 출도착로그 보정 테이블 생성
### 상단 코드를 공항철도(LineA)에 대해 1회 더 실행
### 9호선 코드 수정시 아래도 수정해야함

if(lineNm=="Line9"){
  
  # 2.1. 읽어올 데이터 경로의 파일 정보에 대한 dataframe 만들기
  mydir <- dir(paste0(rtPosResDataPathLineSub)) 
  mydir <- mydir[endsWith(mydir, ".csv")]
  
  dataByDayByPc = data.frame()
  dataByDayByPc = checkRawdataFileList(lineNm_sub, dataType, mydir, dataByDayByPc)
  
  # 2.2. 읽어야 할 날짜 정보를 정리
  # 2.2.1. Directory 안의 파일들의 날짜에 대한 고유값을 도출
  dataObsDates <- unique(dataByDayByPc$oDate) #dataByDayByPc['oDate'].unique()
  oDateAdjLogList <- dataObsDates
  
  
  # 2.3. 날짜별로 logTable파일 읽기
  # 2.3.1. 읽어들일 logTable 종류 지정
  logTableTypes <- c("arr", "dep")
  bnds <- c("bnd0", "bnd1") #, "bznd1")


  v <- 1
  for(v in 1:length(oDateAdjLogList)){
    
    ###### 입력 부분 ######
    oDates <- oDateAdjLogList[v]
    oDateAdjLog <- paste0(substr(oDates,3,4),substr(oDates,6,7),substr(oDates,9,10)) # 분석에 사용할 데이터의 관측일
    
    ## 폴더 내 파일 명 확인해서 원하는 날짜의 bnd0, bnd1 열차 출발,도착로그 가져오기 (2020-06-24)
    oDateAdjLog_char <- ymd(oDateAdjLog,tz = 'Asia/Seoul')
    oDateAdjLog_char <- as.character(oDateAdjLog_char) 
    
    stnID_table <- read.csv(paste0(rtBasePath,"/8_rtData_crawling/2020W01-05-04_실시간위치_도착정보_수집/2019W03-03-04-02_Crawling","/자료출처별_역코드_매칭테이블-200618.csv"),fileEncoding = "EUC-KR")
    
    stnID_table_main <- stnID_table[(stnID_table$lineNmKor %in% myline_sub),]
    
    stn_id <- as.integer(as.character(stnID_table_main$tcdStatnId))
    
    n_sta <- length(stn_id)#max(length(unique(get(paste0(lineNm,"_bnd0"))$CARDSTNID)), length(unique(get(paste0(lineNm,"_bnd1"))$CARDSTNID)))
    sta_id <- stn_id#unique(c(get(paste0(lineNm,"_bnd0"))$CARDSTNID, get(paste0(lineNm,"_bnd1"))$CARDSTNID))
    sta_id <- stn_id[order(stn_id)]#sta_id[order(sta_id)]
    
    
    # 양방향 출도착 정보, 피추월횟수 읽어오기 및 변수 할당하기
    bnd <- bnds[1]
    for (bnd in bnds) {
      
      logTableType <- logTableTypes[1]
      for (logTableType in logTableTypes) {
        
        # 현재 파일이름 정보 저장 (adjLogTable 저장할 때 수집 pc 관련 정보를 유지하기 위해)
        currentFileNmStr <- mydir[chkLogTableOrderInDir(myDate=oDateAdjLog_char, tempLineNm=lineNm_sub, tempBnd=bnd, tempDirectAt=directAt, tempLogTableType=logTableType)]
        currentFileVarNm <- paste0(logTableType,"_table_Nm_",bnd)
        assign(currentFileVarNm, currentFileNmStr)
        
        # 2.3.2. 변수이름 생성
        
        logTableNm <- paste0(logTableType,"LogTable_",bnd)
        shortLogTableNm <- paste0(logTableType,"_log_",bnd)
        trnInfoTableNm <- paste0(logTableType,"_log_",bnd,"_trnInfo")
        
        # 2.3.3. 날짜, 노선명, 방향, 급행여부, logTable 종류에 따라 logTable을 읽기
        # 2.3.3.1. 빈 데이터프레임 만들기 readLogTableByBndByDirectAt 함수를 이용해 
        myLogTable <- data.frame()
        
        # 2.3.3.5. adjLogTable이 없는 경우 최초 수집된 시각표를 사용
        myLogTable <- as.data.frame(readLogTableByBndByDirectAt(oDateAdjLog_char, lineNm_sub, bnd, directAt, logTableType))
        
        # 2.3.4. 열 정보 변경
        # 2.3.4.1 trainNo2 열 이름을 TRNID로 변경
        colnames(myLogTable)[which(colnames(myLogTable)=="trainNo2")] <- "TRNID"
        
        if (lineNm_sub == "LineA") {
          # 2.3.4.3. 9호선 : 교통카드 역 코드 체계로 변경
          # 2.3.4.3.1. operDist 파일 활용하기 위해 파일 읽기
          operDistDF <- read.csv(paste0(rtBasePath,"/3_operDistCsv/Oper_Distance_LineA-210730.csv"),fileEncoding = "EUC-KR")
          
          # 2.3.4.3.2. operDist의 statnId열(rtPos의 역 ID)을 statnId2(tcd의 역 ID)로 변경
          colnames(myLogTable)[8:ncol(myLogTable)] <- sapply(substr(colnames(myLogTable)[8:ncol(myLogTable)],2,99), 
                                                             function(x) { operDistDF$statnId2[which(substr(operDistDF$statnId,start = 2,stop = 99) == x)]  }) 
        }
        
        # 2.3.5. logTable의 시간 정보를 POSIXct 형태로 변경
        myLogTable[,c(8:ncol(myLogTable))] <- lapply(myLogTable[,c(8:ncol(myLogTable))], 
                                                     function(x) {as.POSIXct(x, format="%Y-%m-%d %H:%M:%S")} )
        # 2.3.6.생성한 변수이름에 읽은 파일 할당
        assign(logTableNm, myLogTable)
        
        # 2.3.7. 성종이가 짠 코드와 연계할 변수 정의 및 변수 할당
        shortLogTable <- myLogTable[,c(1,8:length(myLogTable[1,]))] # 열차 시각 정보만 저장한 부분
        trnInfoTable <- myLogTable[,c(1:7)]                      # 열차의 ID등의 정보만 저장한 부분
        
        assign(shortLogTableNm, shortLogTable)                    # 열차정보 제외한 시각정보 저장
        assign(trnInfoTableNm, trnInfoTable)                      # 열차정보 저장
        
        # 2.3.8. 임시변수 삭제
        rm(logTableNm, myLogTable, shortLogTableNm, shortLogTable, trnInfoTableNm, trnInfoTable, currentFileVarNm, currentFileNmStr)
        
      } # for (logTableType in logTableTypes) 
      
      # 2.3.9. 방향 별 피추월횟수 읽기
      ## LineA에서는 피추월 횟수를 모두 0으로 가정
      ## Update on : 2021-07-29 
      
      # myOvtTableByStn <- as.data.frame(readrtDataByBndByDirectAt(myDate=oDateAdjLog_char, tempLineNm=lineNm_sub, tempBnd=bnd, tempDirectAt=directAt, rtDataType="numOvertakeTable"))
      # colnames(myOvtTableByStn)[which(colnames(myOvtTableByStn)=="trainNo2")] <- "TRNID"
      # colnames(myOvtTableByStn)[8:ncol(myOvtTableByStn)] <- sapply(substr(colnames(myOvtTableByStn)[8:ncol(myOvtTableByStn)],2,99), 
      #                                                              function(x) { operDistDF$statnId2[which(substr(operDistDF$statnId,start = 2,stop = 99) == x)]  }) 
      # myOvtTableByStn <- myOvtTableByStn[,c(1,8:ncol(myOvtTableByStn))]
      # 
      # myOvtTableNm <- paste0("overtake_",bnd)  # 변수 이름 생성성 
      # assign(myOvtTableNm, myOvtTableByStn)    # 변수 할당
      # rm(myOvtTableNm, myOvtTableByStn)        # 변수 삭제
    }
    
    
    ####################### 2020-06-29 월요일부터 시작 ########################
    ##### 열차 출발, 도착 로그 테이블 결측치보정!! ####
    #### !!! 현재 코드는 전체 테이블에 대해 NA의 위치를 참고하여 보정하는 형식임. 따라서, 출발 로그와 도착로그의 행수가 맞지않으면, 코드가 제대로 작동하지 않으므로
    #### !!! 출발 로그와 도착 로그 테이블의 행이 다른 날짜가 있는 경우, 결측값 수정을 열차별로 하도록함.
    
    n_trn_dep_bnd0 <- nrow(dep_log_bnd0)
    n_trn_arr_bnd0 <- nrow(arr_log_bnd0)
    n_trn_dep_bnd1 <- nrow(dep_log_bnd1)
    n_trn_arr_bnd1 <- nrow(arr_log_bnd1)
    # n_trn_overtake_bnd0 <- nrow(overtake_bnd0)
    # n_trn_overtake_bnd1 <- nrow(overtake_bnd1)
    # 
    
    if((n_trn_dep_bnd0 == n_trn_arr_bnd0)&(n_trn_dep_bnd1 == n_trn_arr_bnd1)){
      ### 열차 출발 로그 테이블 결측치 보정
      #### 각각의 출발, 도착 로그 테이블에 결측값있는경우 상호보완하여 결측값을 보정
      
      
      dep_log_bnd0_rvsd <- dep_log_bnd0[order(dep_log_bnd0$TRNID),]
      arr_log_bnd0_rvsd <- arr_log_bnd0[order(arr_log_bnd0$TRNID),]
      dep_log_bnd0_rvsd_temp <- dep_log_bnd0_rvsd
      arr_log_bnd0_rvsd_temp <- arr_log_bnd0_rvsd
      
      dep_log_bnd1_rvsd <- dep_log_bnd1[order(dep_log_bnd1$TRNID),]
      arr_log_bnd1_rvsd <- arr_log_bnd1[order(arr_log_bnd1$TRNID),]
      dep_log_bnd1_rvsd_temp <- dep_log_bnd1_rvsd
      arr_log_bnd1_rvsd_temp <- arr_log_bnd1_rvsd
      
      # overtake_bnd0_rvsd <- overtake_bnd0[order(overtake_bnd0$TRNID),]
      # overtake_bnd1_rvsd <- overtake_bnd1[order(overtake_bnd1$TRNID),]
      
      
      #### 결측값(NA)의 위치를 찾은 후, 나머지 로그테이블에서 관측값 후 저장 
      #### Deleay Table 생성 후, 보정시간 적용                              2020-06-29 임의로 90초 사용
      #### (출발 시각 추정값) = (도착 시각 관측값) + (보정 시간)
      #### (도착 시각 추정값) = (출발 시각 관측값) - (보정 시간)
      
      dep_log_bnd0_NA <- is.na(dep_log_bnd0_rvsd)
      arr_log_bnd0_NA <- is.na(arr_log_bnd0_rvsd)
      
      dep_log_bnd1_NA <- is.na(dep_log_bnd1_rvsd)
      arr_log_bnd1_NA <- is.na(arr_log_bnd1_rvsd)
      
      
      dep_log_bnd0_rvsd_temp[dep_log_bnd0_NA] <- arr_log_bnd0_rvsd[dep_log_bnd0_NA] 
      arr_log_bnd0_rvsd_temp[arr_log_bnd0_NA] <- dep_log_bnd0_rvsd[arr_log_bnd0_NA]
      
      dep_log_bnd1_rvsd_temp[dep_log_bnd1_NA] <- arr_log_bnd1_rvsd[dep_log_bnd1_NA] 
      arr_log_bnd1_rvsd_temp[arr_log_bnd1_NA] <- dep_log_bnd1_rvsd[arr_log_bnd1_NA]
      
      
      dep_log_bnd0_rvsd <- dep_log_bnd0_rvsd_temp
      arr_log_bnd0_rvsd <- arr_log_bnd0_rvsd_temp
      
      dep_log_bnd1_rvsd <- dep_log_bnd1_rvsd_temp
      arr_log_bnd1_rvsd <- arr_log_bnd1_rvsd_temp
      
      
      ## 방향별로 열차타입 구분하여 피추월 횟수별 열차타입별 보정시간 계산
      {
        
        ## 방향별 구분
        
        ##BND0
        {
          ### 열차타입 구분
          G_TRNID_bnd0 <- arr_log_bnd0_trnInfo$TRNID[(arr_log_bnd0_trnInfo$directAt %in% 0)]
          # E_TRNID_bnd0 <- arr_log_bnd0_trnInfo$TRNID[(arr_log_bnd0_trnInfo$directAt %in% 1)] 
          
          # G_overtake_bnd0_rvsd_base <- overtake_bnd0_rvsd[(overtake_bnd0_rvsd$TRNID %in% G_TRNID_bnd0),]
          # E_overtake_bnd0_rvsd_base <- overtake_bnd0_rvsd[(overtake_bnd0_rvsd$TRNID %in% E_TRNID_bnd0),]
          # 
          # G_overtake_bnd0_rvsd <- G_overtake_bnd0_rvsd_base
          # E_overtake_bnd0_rvsd <- E_overtake_bnd0_rvsd_base
          # 
          
          G_overtake_bnd0_rvsd <- data.frame(TRNID = G_TRNID_bnd0)
          G_overtake_bnd0_rvsd$TRNID <- as.character(G_overtake_bnd0_rvsd$TRNID)
          
          for(jh in 1:n_sta){
            tempStnColumn <- rep(NA,length(G_TRNID_bnd0))
            G_overtake_bnd0_rvsd <- cbind(G_overtake_bnd0_rvsd,tempStnColumn)
          }
          
          names(G_overtake_bnd0_rvsd) <- c("TRNID",sta_id)

          n_cur_col <- 2
          for(n_cur_col in 2:ncol(G_overtake_bnd0_rvsd)){
            temp_column <- G_overtake_bnd0_rvsd[,n_cur_col]
            temp_column[temp_column %in% NA] <- delay_standard$delay_G[delay_standard$nOvt == 0]

            G_overtake_bnd0_rvsd[,n_cur_col] <- temp_column
            rm(temp_column)
          }
          
          ### 계산 후 병합
          overtake_bnd0_calc <- rbind(G_overtake_bnd0_rvsd)
          overtake_bnd0_calc <- overtake_bnd0_calc[order(overtake_bnd0_calc$TRNID),]
          
        }##BND0
        
        ### 전체 역에 대해 보정시간표 계산
        {
          
          calibration_table_bnd0 <- overtake_bnd0_calc
          
          #### 급행 열차 : 급행 정차 X 역에 대해 별도의 보정시간부여
          
          ##### 급행미정차역
          ExpNonStopStn <- stn_id
          
          
        }### 전체 역에 대해 보정시간표 계산
        
        
        
        ##BND1
        {
          ### 열차타입 구분
          G_TRNID_bnd1 <- arr_log_bnd1_trnInfo$TRNID[(arr_log_bnd1_trnInfo$directAt %in% 0)] 
          # E_TRNID_bnd1 <- as.integer(arr_log_bnd1_trnInfo$TRNID[(arr_log_bnd1_trnInfo$directAt %in% 1)]) 
          
          # G_overtake_bnd1_rvsd_base <- overtake_bnd1_rvsd[(overtake_bnd1_rvsd$TRNID %in% G_TRNID_bnd1),]
          # # E_overtake_bnd1_rvsd_base <- overtake_bnd1_rvsd[(overtake_bnd1_rvsd$TRNID %in% E_TRNID_bnd1),]
          # 
          # G_overtake_bnd1_rvsd <- G_overtake_bnd1_rvsd_base
          # E_overtake_bnd1_rvsd <- E_overtake_bnd1_rvsd_base
          
          G_overtake_bnd1_rvsd <- data.frame(TRNID = G_TRNID_bnd1)
          G_overtake_bnd1_rvsd$TRNID <- as.character(G_overtake_bnd1_rvsd$TRNID)
          
          for(jh in 1:n_sta){
            tempStnColumn <- rep(NA,length(G_TRNID_bnd1))
            G_overtake_bnd1_rvsd <- cbind(G_overtake_bnd1_rvsd,tempStnColumn)
          }
          
          names(G_overtake_bnd1_rvsd) <- c("TRNID",sta_id)
          
          n_cur_col <- 2
          for(n_cur_col in 2:ncol(G_overtake_bnd1_rvsd)){
            temp_column <- G_overtake_bnd1_rvsd[,n_cur_col]
            temp_column[temp_column %in% NA] <- delay_standard$delay_G[delay_standard$nOvt == 0]
            
            G_overtake_bnd1_rvsd[,n_cur_col] <- temp_column
            rm(temp_column)
          }
         
          ### 계산 후 병합
          overtake_bnd1_calc <- rbind(G_overtake_bnd1_rvsd)
          overtake_bnd1_calc <- overtake_bnd1_calc[order(overtake_bnd1_calc$TRNID),]
          
        }##BND1
        
        ### 전체 역에 대해 보정시간표 계산
        {
          
          calibration_table_bnd1 <- overtake_bnd1_calc
          
          #### 급행 열차 : 급행 정차 X 역에 대해 별도의 보정시간부여
          
          ##### 급행미정차역
          ExpNonStopStn <- stn_id
          
          
        }### 전체 역에 대해 보정시간표 계산
      }
      
      
      #### 기존 결측값에 보정시간 적용
      
      ##### BND0
      if(sum(dep_log_bnd0_rvsd$TRNID != calibration_table_bnd0$TRNID)==0){
        dep_log_bnd0_NA_calibrate <- dep_log_bnd0_NA[,-1] * calibration_table_bnd0[,-1]
        dep_log_bnd0_NA_calibrate <- cbind(G_TRNID_bnd0,dep_log_bnd0_NA_calibrate)
        names(dep_log_bnd0_NA_calibrate)[1] <- "TRNID"
      }else{
        print(paste0("HAS ERROR IN CALCULATING CALIBRATE TABLE"))
      }
      
      if(sum(arr_log_bnd0_rvsd$TRNID != calibration_table_bnd0$TRNID)==0){
        arr_log_bnd0_NA_calibrate <- arr_log_bnd0_NA[,-1] * calibration_table_bnd0[,-1]
        arr_log_bnd0_NA_calibrate <- cbind(G_TRNID_bnd0,arr_log_bnd0_NA_calibrate)
        names(arr_log_bnd0_NA_calibrate)[1] <- "TRNID"
      }else{
        print(paste0("HAS ERROR IN CALCULATING CALIBRATE TABLE"))
      }
      
      ##### BND1
      if(sum(dep_log_bnd1_rvsd$TRNID != calibration_table_bnd1$TRNID)==0){
        dep_log_bnd1_NA_calibrate <- dep_log_bnd1_NA[,-1] * calibration_table_bnd1[,-1]
        dep_log_bnd1_NA_calibrate <- cbind(G_TRNID_bnd1,dep_log_bnd1_NA_calibrate)
        names(dep_log_bnd1_NA_calibrate) <- "TRNID"
        
        # #  Update on : 2021-06-11
        # # 김포공항역(4102)에서 출발하는 급행열차의 출발시각을 도착시각으로부터 추정할 때는 120초를 적용함(윤석이형 의견 반영)  
        # # 출발시각 운영실적 테이블 중, 결측값이 있는 급행열차에만 새로운 보정시간 적용
        # whichEtrnWhich <- which((calibration_table_bnd1$TRNID) %in% E_TRNID_bnd1)
        # dep_log_bnd1_NA_calibrate$`4102`[whichEtrnWhich][!(dep_log_bnd1_NA_calibrate$`4102`[whichEtrnWhich] %in% 0)] <- E_TRNID_OccupiedTime_in_GimpoStn
        # #dep_log_bnd1_NA_calibrate[102:110,]
        # #test11 <-dep_log_bnd1_NA_calibrate$`4102`[whichEtrnWhich]
      }else{
        print(paste0("HAS ERROR IN CALCULATING CALIBRATE TABLE"))
      }
      
      if(sum(arr_log_bnd1_rvsd$TRNID != calibration_table_bnd1$TRNID)==0){
        arr_log_bnd1_NA_calibrate <- arr_log_bnd1_NA[,-1] * calibration_table_bnd1[,-1]
        arr_log_bnd1_NA_calibrate <- cbind(G_TRNID_bnd1,arr_log_bnd1_NA_calibrate)
        names(arr_log_bnd1_NA_calibrate) <- "TRNID"
      }else{
        print(paste0("HAS ERROR IN CALCULATING CALIBRATE TABLE"))
      }
      
      #dep_log_bnd0[2,1:10]
      #dep_log_bnd0_rvsd[2,1:10]
      
      #head(dep_log_bnd0_rvsd[,-1] + dep_log_bnd0_NA_calibrate[,-1])
      #head(dep_log_bnd0_rvsd[,-1])
      #head(dep_log_bnd0_NA_calibrate[,-1])
      
      
      ## 도착 추정시, 출발 시각 - 보정시간
      ## 출발 추정시, 도착 시각 + 보정시간
      
      dep_log_bnd0_rvsd[,-1] <- (dep_log_bnd0_rvsd[,-1] + dep_log_bnd0_NA_calibrate[,-1])
      
      arr_log_bnd0_rvsd[,-1] <- (arr_log_bnd0_rvsd[,-1] - arr_log_bnd0_NA_calibrate[,-1])
      
      dep_log_bnd1_rvsd[,-1] <- (dep_log_bnd1_rvsd[,-1] + dep_log_bnd1_NA_calibrate[,-1]) 
      arr_log_bnd1_rvsd[,-1] <- (arr_log_bnd1_rvsd[,-1] - arr_log_bnd1_NA_calibrate[,-1])
      
      #dep_log_bnd0_rvsd[9,]
      dep_log_bnd0_rvsd_backup <- dep_log_bnd0_rvsd
      arr_log_bnd0_rvsd_backup <- arr_log_bnd0_rvsd
      dep_log_bnd1_rvsd_backup <- dep_log_bnd1_rvsd
      arr_log_bnd1_rvsd_backup <- arr_log_bnd1_rvsd
      
      log_list <- c("dep_log_bnd0","arr_log_bnd0","dep_log_bnd1","arr_log_bnd1")
      #cur_log <- 1
      
      #### 출발, 도착 추정값이 해당 열차 시간 순서가 논리적으로 맞는지 확인 ( Sta A-1 time < Sta A time < Sta A+1 time)
      #cur_log <- 2
      
      #### 운행방향별 열차별로 오류 유형에 대해 정리
      #### -> 카드데이터와 매칭 완료 후, 카드데이터마다 매칭된 열차에 대한 오류 유형을 부여할 예정
      cur_log<-1
      for(cur_log in 1:length(log_list)){
        
        trn_err_type <- NULL
        
        temp_Nm_log <- log_list[cur_log]
        temp_bnd <- substr(temp_Nm_log,start = (nchar(temp_Nm_log)-3) ,stop = nchar(temp_Nm_log))
        
        TRNID <- get(paste0(temp_Nm_log,"_rvsd"))[,1] 
        temp_log_org <- get(paste0(temp_Nm_log))[,-1]
        temp_log_table <- get(paste0(temp_Nm_log,"_rvsd"))[,-1] 
        temp_log_NA_origin <- get(paste0(temp_Nm_log,"_NA"))[,-1] 
        
        if(lineNm_sub == "LineA"){
          if(temp_bnd=="bnd0"){
            bnd_min_which <- 1
            bnd_max_which <- n_sta
          }else{
            bnd_min_which <- n_sta
            bnd_max_which <- 1
          }
        }
        
        ## 출발, 도착로그의 운행방향 파악 후, 열차 시간 순서 확인
        ### 광교 -> 강남
        ### 종합운동장 -> 개화
        ### LineA : 
        if(bnd_min_which > bnd_max_which){
          
          cur_trn_err <- 2
          
          for(cur_trn_err in 1:nrow(temp_log_table)){
            ### 열차 번호 가져오기
            temp_TRNID <- TRNID[cur_trn_err]
            
            ### 열차별 시각표상 NA가 있는지 확인 
            temp_NA_error <- ifelse((sum(is.na(temp_log_table[cur_trn_err,]))==0), 0, 1)
            
            ### 열차별 수정된 시각표가 시간흐름에 위배되지않는지 확인
            temp_trn_squence <- frank(x = as.numeric(temp_log_table[cur_trn_err,]),na.last = NA)
            
            ### 열차 로그가 모두 NA인 경우, 생략
            if(sum(temp_trn_squence)!=0){            
              max_sq <- max(temp_trn_squence)
              min_sq <- min(temp_trn_squence)
              
              temp_exct_sequence <- max_sq:min_sq
              
              temp_sequence_error <- ifelse((sum(temp_trn_squence != temp_exct_sequence)==0),0,1)
              
              temp_err_type <- (temp_NA_error)  +  (temp_sequence_error*2)
              
              
              temp_trn_err <- data.frame(BND = temp_bnd,TRNID = temp_TRNID,Including_NA = temp_NA_error, Sequence_error = temp_sequence_error, Error_type = temp_err_type)
              
              
              
              trn_err_type <- rbind(trn_err_type,temp_trn_err)
              
              rm(temp_TRNID,temp_NA_error,temp_trn_squence,max_sq,min_sq,temp_exct_sequence,temp_sequence_error,temp_err_type,temp_trn_err)
            }#if(sum(temp_trn_squence)!=0){     
            
          }#for(cur_trn_err in 1:nrow(temp_log_table)){
          
          assign(paste0(temp_Nm_log,"_err_type"),trn_err_type,)
          
          rm(trn_err_type)
          
          # 2020-07-02 이전 작업 || 열차 추정값으로 인해 시간 흐름 위배 경우가 생기는 경우, 추정값 제거
          # 시간 흐름의 위배 여부만 확인 후, 값 수정은 따로 진행하지 않기로함
          ## 이전 역의 시간 차이 계산
          #prev_temp_log_table <- temp_log_table[,-1]
          #prev_temp_log_table <- cbind(prev_temp_log_table,as.POSIXct(NA))
          #time_diff_table <- temp_log_table - prev_temp_log_table
          #
          #all_NA <- is.na(prev_temp_log_table) * is.na(temp_log_table) 
          #one_NA <- (is.na(prev_temp_log_table) * (!is.na(temp_log_table))) + ((!is.na(prev_temp_log_table)) * (is.na(temp_log_table))) 
          
          #both <- all_NA + one_NA
          
          #both[both >=2] 
          
          #time_diff_table[one_NA == 1]
          
          
          ### 강남 -> 광교
          ### 개화 -> 종합운동장
        }else if(bnd_min_which < bnd_max_which){#if(bnd_min_which > bnd_max_which){
          cur_trn_err <- 1
          
          for(cur_trn_err in 1:nrow(temp_log_table)){
            ### 열차 번호 가져오기
            temp_TRNID <- TRNID[cur_trn_err]
            
            ### 열차별 시각표상 NA가 있는지 확인 
            temp_NA_error <- ifelse((sum(is.na(temp_log_table[cur_trn_err,]))==0), 0, 1)
            
            ### 열차별 수정된 시각표가 시간흐름에 위배되지않는지 확인
            temp_trn_squence <- frank(x = as.numeric(temp_log_table[cur_trn_err,]),na.last = NA)
            ### 열차 로그가 모두 NA인 경우, 생략
            if(sum(temp_trn_squence)!=0){
              
              
              max_sq <- max(temp_trn_squence)
              min_sq <- min(temp_trn_squence)
              
              temp_exct_sequence <- min_sq:max_sq
              
              temp_sequence_error <- ifelse((sum(temp_trn_squence != temp_exct_sequence)==0),0,1)
              
              temp_err_type <-  temp_NA_error + (temp_sequence_error * 2)
              
              temp_trn_err <- data.frame(BND = temp_bnd,TRNID = temp_TRNID,Including_NA = temp_NA_error, Sequence_error = temp_sequence_error, Error_type = temp_err_type)
              
              trn_err_type <- rbind(trn_err_type,temp_trn_err)
              
              rm(temp_TRNID,temp_NA_error,temp_trn_squence,max_sq,min_sq,temp_exct_sequence,temp_sequence_error,temp_err_type,temp_trn_err)
            }
            
          }#for(cur_trn_err in 1:nrow(temp_log_table)){
          
          assign(paste0(temp_Nm_log,"_err_type"),trn_err_type,)
          
          rm(trn_err_type)
          
        }else if(bnd_min_which == bnd_max_which){
          
          print(paste0("HAS ERROR IN CHECKING THE TRAIN LOGICAL TIME SEQUENCE1 ||",temp_Nm_log))
          
        }
        
        
        #assign((paste0(temp_Nm_log,"_rvsd")),cbind(TRNID,temp_log_table))
        
        ## 논리적 오류보다 NA 개수가 작으면 오류 출력
        ### 위 작업을 통해 논리적 오류를 제거하기 때문에 이전보다 NA가 더 많아짐
        #if(sum(is.na(get(paste0(temp_Nm_log,"_rvsd_backup")))) > sum(is.na(get(paste0(temp_Nm_log,"_rvsd"))))){
        #  
        #  print(paste0("HAS ERROR IN CHECKING THE TRAIN LOGICAL TIME SEQUENCE2 ||",temp_Nm_log))
        #  
        #  break()
        #}
        
        
      }#for(cur_log in 1:length(log_list)){
      
      
      ## 에러 발견된 데이터 rds로 저장
      ### !!!추정값 반영된 시각표를 기준으로 저장하였음
      
      err_save<-1
      for(err_save in 1:length(log_list)){
        temp_Nm_log <- log_list[err_save]
        which_temp_log_err_trn <- which((apply(get(paste0(temp_Nm_log,"_err_type"))[,-c(1,2)], MARGIN = 1, FUN = sum))!=0)
        saveRDS(get(paste0(temp_Nm_log,"_err_type")),file = paste0(rtPosResDataPathLineSub,"/",oDateAdjLog_char,"-",lineNm,"-",substr(temp_Nm_log,9,12),"-rtPos_",substr(temp_Nm_log,1,3),"TrnErrType-",fwDate,".RDS")) ## 출발, 도착로그 오류 유형 (승,하차 재차인원 만들때 사용)
        saveRDS(get(temp_Nm_log)[which_temp_log_err_trn,],file = paste0(rtPosResDataPathLineSub,"/",oDateAdjLog_char,"-",lineNm,"-",substr(temp_Nm_log,9,12),"-rtPos_",substr(temp_Nm_log,1,3),"ErrTrnLog-",fwDate,".RDS")) ## 1개이상의 오류가 발견되면 저장
        rm(temp_Nm_log,which_temp_log_err_trn)
      }
      
      ## 출발 및 도착 로그 에러종류 저장
      
      n_dep_bnd0_NA <- sum(dep_log_bnd0_err_type$Including_NA==1)
      n_dep_bnd0_sequence_err <- sum(dep_log_bnd0_err_type$Sequence_error==1)
      n_dep_bnd0_both <- sum(dep_log_bnd0_err_type$Error_type==3)
      n_dep_bnd0_all_recorded <- sum(dep_log_bnd0_err_type$Error_type==0)
      n_dep_bnd0_total_trn <- nrow(dep_log_bnd0_err_type)
      
      n_arr_bnd0_NA <- sum(arr_log_bnd0_err_type$Including_NA==1)
      n_arr_bnd0_sequence_err <- sum(arr_log_bnd0_err_type$Sequence_error==1)
      n_arr_bnd0_both <- sum(arr_log_bnd0_err_type$Error_type==3)
      n_arr_bnd0_all_recorded <- sum(arr_log_bnd0_err_type$Error_type==0)
      n_arr_bnd0_total_trn <- nrow(arr_log_bnd0_err_type)
      
      n_dep_bnd1_NA <- sum(dep_log_bnd1_err_type$Including_NA==1)
      n_dep_bnd1_sequence_err <- sum(dep_log_bnd1_err_type$Sequence_error==1)
      n_dep_bnd1_both <- sum(dep_log_bnd1_err_type$Error_type==3)
      n_dep_bnd1_all_recorded <- sum(dep_log_bnd1_err_type$Error_type==0)
      n_dep_bnd1_total_trn <- nrow(dep_log_bnd1_err_type)
      
      n_arr_bnd1_NA <- sum(arr_log_bnd1_err_type$Including_NA==1)
      n_arr_bnd1_sequence_err <- sum(arr_log_bnd1_err_type$Sequence_error==1)
      n_arr_bnd1_both <- sum(arr_log_bnd1_err_type$Error_type==3)
      n_arr_bnd1_all_recorded <- sum(arr_log_bnd1_err_type$Error_type==0)
      n_arr_bnd1_total_trn <- nrow(arr_log_bnd1_err_type)
      
      log_err_type_summary <- data.frame(type = c("Err1-including_NA","Err2-sequence_error","Err3-both","all_recorded","total"), depLog_bnd0_obs = c(n_dep_bnd0_NA,n_dep_bnd0_sequence_err,n_dep_bnd0_both,n_dep_bnd0_all_recorded,n_dep_bnd0_total_trn),depLog_bnd0_ratio = c(n_dep_bnd0_NA,n_dep_bnd0_sequence_err,n_dep_bnd0_both,n_dep_bnd0_all_recorded,n_dep_bnd0_total_trn)/n_dep_bnd0_total_trn, arrLog_bnd0_obs = c(n_arr_bnd0_NA,n_arr_bnd0_sequence_err,n_arr_bnd0_both,n_arr_bnd0_all_recorded,n_arr_bnd0_total_trn), arrLog_bnd0_ratio = c(n_arr_bnd0_NA,n_arr_bnd0_sequence_err,n_arr_bnd0_both,n_arr_bnd0_all_recorded,n_arr_bnd0_total_trn)/n_arr_bnd0_total_trn, depLog_bnd1_obs = c(n_dep_bnd1_NA,n_dep_bnd1_sequence_err,n_dep_bnd1_both,n_dep_bnd1_all_recorded,n_dep_bnd1_total_trn), depLog_bnd1_ratio = c(n_dep_bnd1_NA,n_dep_bnd1_sequence_err,n_dep_bnd1_both,n_dep_bnd1_all_recorded,n_dep_bnd1_total_trn)/n_dep_bnd1_total_trn, arrLog_bnd1_obs = c(n_arr_bnd1_NA,n_arr_bnd1_sequence_err,n_arr_bnd1_both,n_arr_bnd1_all_recorded,n_arr_bnd1_total_trn), arrLog_bnd1_ratio = c(n_arr_bnd1_NA,n_arr_bnd1_sequence_err,n_arr_bnd1_both,n_arr_bnd1_all_recorded,n_arr_bnd1_total_trn)/n_arr_bnd1_total_trn)
      
      print(paste0("SUCESS(",myline_sub,") || oDate ",oDateAdjLog," || ",Sys.time()))
      write.csv(log_err_type_summary,file = paste0(rtPosResDataPathLineSub,"/",oDateAdjLog_char,"-",lineNm_sub,"-","bndA","-rtPos_trnLog_errType_summary-",fwDate,".csv"),row.names = F)
      
      
      rm(n_arr_bnd0_all_recorded,n_arr_bnd0_both,n_arr_bnd0_NA,n_arr_bnd0_sequence_err,n_arr_bnd0_total_trn,n_dep_bnd0_all_recorded,n_dep_bnd0_both,n_dep_bnd0_NA,n_dep_bnd0_sequence_err,n_dep_bnd0_total_trn,n_arr_bnd1_all_recorded,n_arr_bnd1_both,n_arr_bnd1_NA,n_arr_bnd1_sequence_err,n_arr_bnd1_total_trn,n_dep_bnd1_all_recorded,n_dep_bnd1_both,n_dep_bnd1_NA,n_dep_bnd1_sequence_err,n_dep_bnd1_total_trn)
    }else{
      
      print(paste0("HAS ERROR IN REVISING NA VALUE FOR DEPARTURE || ARRIVAL LOG TABLE || OVERTAKE TABLE : oDate ",oDateAdjLog,"(",myline_sub,")"))
      break()
    }
    
    
    #paste0(substr(x = arr_table_Nm_bnd0,start = 1,stop = 32),"adjTtable",substr(x = arr_table_Nm_bnd0,start = 39,stop = (nchar(arr_table_Nm_bnd0)-10)),fwDate,".csv")
    arr_log_bnd0_rvsd_res <- merge(arr_log_bnd0_trnInfo, arr_log_bnd0_rvsd, by.x="TRNID", by.y="TRNID")
    dep_log_bnd0_rvsd_res <- merge(dep_log_bnd0_trnInfo, dep_log_bnd0_rvsd, by.x="TRNID", by.y="TRNID")
    arr_log_bnd1_rvsd_res <- merge(arr_log_bnd1_trnInfo, arr_log_bnd1_rvsd, by.x="TRNID", by.y="TRNID")
    dep_log_bnd1_rvsd_res <- merge(dep_log_bnd1_trnInfo, dep_log_bnd1_rvsd, by.x="TRNID", by.y="TRNID")
    
    if(TimeDiffSwitch == TRUE){
      ## 실제 ATS 지상자와 출입문 개폐시간의 시간 차이 통계 데이터를 통해 출입문 개폐시간 추정에 사용 (9호선 열차별 역별 7일간 데이터)
      ## trueDwTime 관련
      ## 역별, 운행 방향별, 열차 등급별로 시간 차이가 다름
      ## 2021-05-26 성종 Update
      #tempStn <- "4101"
      
      stnList <- as.character(stnID_table_main$tcdStatnId)
      #tempStn <-"4103"
      for(tempStn in stnList){
        
        arr_log_bnd0_rvsd_res[(arr_log_bnd0_rvsd_res$TRNID %in% G_TRNID_bnd0),tempStn] <- arr_log_bnd0_rvsd_res[(arr_log_bnd0_rvsd_res$TRNID %in% G_TRNID_bnd0),tempStn] + timeDiffGeneral[((timeDiffGeneral$BND %in% "bnd0")&(timeDiffGeneral$TYPE %in% "arr")),tempStn]
        # arr_log_bnd0_rvsd_res[(arr_log_bnd0_rvsd_res$TRNID %in% E_TRNID_bnd0),tempStn] <- arr_log_bnd0_rvsd_res[(arr_log_bnd0_rvsd_res$TRNID %in% E_TRNID_bnd0),tempStn]  + timeDiffExpress[((timeDiffExpress$BND %in% "bnd0")&(timeDiffExpress$TYPE %in% "arr")),tempStn]
        
        arr_log_bnd1_rvsd_res[(arr_log_bnd1_rvsd_res$TRNID %in% G_TRNID_bnd1),tempStn] <- arr_log_bnd1_rvsd_res[(arr_log_bnd1_rvsd_res$TRNID %in% G_TRNID_bnd1),tempStn] + timeDiffGeneral[((timeDiffGeneral$BND %in% "bnd1")&(timeDiffGeneral$TYPE %in% "arr")),tempStn]
        # arr_log_bnd1_rvsd_res[(arr_log_bnd1_rvsd_res$TRNID %in% E_TRNID_bnd1),tempStn] <- arr_log_bnd1_rvsd_res[(arr_log_bnd1_rvsd_res$TRNID %in% E_TRNID_bnd1),tempStn]  + timeDiffExpress[((timeDiffExpress$BND %in% "bnd1")&(timeDiffExpress$TYPE %in% "arr")),tempStn]
        
        dep_log_bnd0_rvsd_res[(dep_log_bnd0_rvsd_res$TRNID %in% G_TRNID_bnd0),tempStn]  <- dep_log_bnd0_rvsd_res[(dep_log_bnd0_rvsd_res$TRNID %in% G_TRNID_bnd0),tempStn] - timeDiffGeneral[((timeDiffGeneral$BND %in% "bnd0")&(timeDiffGeneral$TYPE %in% "dep")),tempStn]
        # dep_log_bnd0_rvsd_res[(dep_log_bnd0_rvsd_res$TRNID %in% E_TRNID_bnd0),tempStn]   <- dep_log_bnd0_rvsd_res[(dep_log_bnd0_rvsd_res$TRNID %in% E_TRNID_bnd0),tempStn] - timeDiffExpress[((timeDiffExpress$BND %in% "bnd0")&(timeDiffExpress$TYPE %in% "dep")),tempStn]
        
        dep_log_bnd1_rvsd_res[(dep_log_bnd1_rvsd_res$TRNID %in% G_TRNID_bnd1),tempStn] <- dep_log_bnd1_rvsd_res[(dep_log_bnd1_rvsd_res$TRNID %in% G_TRNID_bnd1),tempStn] - timeDiffGeneral[((timeDiffGeneral$BND %in% "bnd1")&(timeDiffGeneral$TYPE %in% "dep")),tempStn]
        # dep_log_bnd1_rvsd_res[(dep_log_bnd1_rvsd_res$TRNID %in% E_TRNID_bnd1),tempStn] <- dep_log_bnd1_rvsd_res[(dep_log_bnd1_rvsd_res$TRNID %in% E_TRNID_bnd1),tempStn]  - timeDiffExpress[((timeDiffExpress$BND %in% "bnd1")&(timeDiffExpress$TYPE %in% "dep")),tempStn]
        
      }
    }
    ## 보정된 열차 로그 테이블은 급행열차 미정차역의 급행열차 로그가 기록되어있음
    write.csv(arr_log_bnd0_rvsd_res,file = paste0(rtPosResDataPathLineSub,"/",substr(x = arr_table_Nm_bnd0,start = 1,stop = 32+etcWord),"adjTtable",substr(x = arr_table_Nm_bnd0,start = 39+etcWord,stop = (nchar(arr_table_Nm_bnd0)-10)),fwDate,".csv"),row.names = F)
    write.csv(dep_log_bnd0_rvsd_res,file = paste0(rtPosResDataPathLineSub,"/",substr(x = dep_table_Nm_bnd0,start = 1,stop = 32+etcWord),"adjTtable",substr(x = dep_table_Nm_bnd0,start = 39+etcWord,stop = (nchar(dep_table_Nm_bnd0)-10)),fwDate,".csv"),row.names = F)
    
    write.csv(arr_log_bnd1_rvsd_res,file = paste0(rtPosResDataPathLineSub,"/",substr(x = arr_table_Nm_bnd1,start = 1,stop = 32+etcWord),"adjTtable",substr(x = arr_table_Nm_bnd1,start = 39+etcWord,stop = (nchar(arr_table_Nm_bnd1)-10)),fwDate,".csv"),row.names = F)
    write.csv(dep_log_bnd1_rvsd_res,file = paste0(rtPosResDataPathLineSub,"/",substr(x = dep_table_Nm_bnd1,start = 1,stop = 32+etcWord),"adjTtable",substr(x = dep_table_Nm_bnd1,start = 39+etcWord,stop = (nchar(dep_table_Nm_bnd1)-10)),fwDate,".csv"),row.names = F)
    
  }
  
  
  
}
