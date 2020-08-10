#TEXT ANALYSIS WITH R / NATURAL LANGUAGE PROCESSING
########################################################
#PART1
#intsalling packages
install.packages(c("quanteda","irlba"))
library(ggplot2)
library(e1071)
library(caret)
library(quanteda)
library(irlba)
#quanteda- main for text analytics
#irlba- singular value decomposition (svd) or word to vec

#reading the data
spam.raw<-read.csv("spam.csv",stringsAsFactors = F)
#strings are not made factors as by default r makes string data as catagorical var

#clean up the data frame
#remove the unnecessary cols
spam.raw<-spam.raw[,1:2]
#change the label names
names(spam.raw)<-c("Label","Text")

#check for missing values
length(which(!complete.cases(spam.raw)))


#convert our class label into a factor
spam.raw$Label<-factor(spam.raw$Label)

#exploring the data
#looking at distribution of the class labels(i.e ham v/s spam)
#table tells how many hams and spams
#prop.table converts the numbers into % or fractional of marginal data
prop.table(table(spam.raw$Label))

# lets lookk at the length of each sentence int the dataset by adding a new feature
#for the length of message
#nchar-counts the no of characters
spam.raw$textlength<-nchar(spam.raw$Text)
summary(spam.raw$textlength)

#visualize distribution with ggplot2 adding segmentation for ham/spam

ggplot(spam.raw,aes(x=textlength,fill=Label))+
  theme_bw()+
  geom_histogram(binwidth = 5)+
  labs(y="Text count",x="length of text",
       title = "Distribution of text lengthd with class labels")

#data has non trivial class imbalance , we'll use the caret package to
#create a random train/test split that ensures the correct ham/spam class
#label proportions(we'll use caret for random stratified split)

#use caret to create 70-30 stratified(maintain propotion acr0ss the split) split, and set random seed for reproducibility
set.seed(32984)
indexes<-createDataPartition(spam.raw$Label,times = 1,
                            p=0.7,list = F )
train<-spam.raw[indexes,]
test<-spam.raw[-indexes,]

#verify proportions-check for stratification
prop.table(table(train$Label))
prop.table(table(test$Label))

#html-escaped ampersand charracter
train$Text[21]

#a url
train$Text[357]
#how to deal with url depends upon the problem we are dealing with
#ex https presnt in the message can be classified as spam

help(package="quanteda")

#tokenization sms text messages
train.tokens<-tokens(train$Text,what = "word",
                     remove_numbers = T,remove_punct = T,
                     remove_symbols = T,split_hyphens = T)
# we can also tokenize by characters
# take a look at a specific sms message and see how it transforms
train.tokens[[357]]

#lower case the tokens.
train.tokens<-tokens_tolower(train.tokens)

#use quanteda's built in stopwords list for english
#note- you should always inspect stopword list for applicability to
#your problem/domain

train.tokens<-tokens_select(train.tokens,stopwords(),
                            selection = "remove")
train.tokens[[357]]


#Perform stemming on the tokens

train.tokens<-tokens_wordstem(train.tokens,language = "english")
train.tokens[[20]]

######################################################################################
#create the BAG OF WORDS model
#dfm creates a matrix of tokens
train.tokens.dfm<-dfm(train.tokens,tolower = F)

#transform to a matrix and inspect
train.tokens.matrix<-as.matrix(train.tokens.dfm)


dim(train.tokens.matrix)
#problems - large matrix faces curse of dimensionality are are sparse matrix
#that is a lot of 0's data gtes big in text analytics

#investigate the effects of stemming.
colnames(train.tokens.dfm)[1:50]

#per best practices, we will leverage cross validation(cv)as
#the basis of our modelling process. using cv we can create
#estimates of how well our model will do in production on new,
#unseen data. Cv is powerful, but requires more processing and therefore more time.



# setup a the feature data frame with labels


train.tokens_dfm<-convert(train.tokens.dfm,to ="data.frame")
train.tokens.df<-cbind(Label=train$Label,train.tokens_dfm)
dim(train.tokens.df)
train.tokens.df<-train.tokens.df[-2]
#often, tokenization requires some additional preprocessing
names(train.tokens.df)[c(146,148,238)]
#some ml algo does not understand such names so we need to 
#convert such col names to universally known names
#clean up col names
names(train.tokens.df)<-make.names(names(train.tokens.df))


#use caret to create stratified fold for 10 folds cv repeated
#3 times (i.e create 30 random staratified samples)
#useing stratified cv cuz of classs imbalance
set.seed(48749)
cv.folds<-createMultiFolds(train$Label,k=10,times=3)

cv.control<-trainControl(method="repeatedcv",number=10,
                         repeats = 3,index = cv.folds)

#for stratified we use index parameter


#note: our df is non trivial in size, as such, cv runs will take,
#quiet a long time to run. to cut down on total execution time
#we use the doSNOW package to allow for multi core training in parallel.


install.packages("doSNOW")
library(doSNOW)

#time the code execution
start.time<-Sys.time()

#create a cluster to work on 10 logical cores.
cl<-makeCluster(2,type="SOCK")
registerDoSNOW(cl)

# as our data is non trivial in size at this point, use a single decision
#tree algo as first model .


rpart.cv.1<-train(Label~.,data=train.tokens.df,method="rpart",
                  trControl=cv.control,tuneLength=7)

#tuneLength specifies try 7 differnt values for r and select the best 
#that is hyper parameter tuning

#processing is done, stop cluster
stopCluster(cl)

#total time of execution on workspace
total.time<-Sys.time()-start.time
total.time

#check results
rpart.cv.1
##########################################################################3
#TF-IDF

#our function for calculating relative term frequency(tf)
term.frequency<-function(row){
  row/sum(row)
}

#our function for calculatiing inverse document frequency(IDF)
inverse.doc.freq<-function(col){
 corpus.size<-length(col)
 doc.count<-length(which(col>0))
log10(corpus.size/doc.count)
 
 }
#note to veridy apply the tf-idf function in quantida and set normalization
#to truw so as to see that results match or not


# our function for calculation tf-idf
tf.idf<-function(tf,idf){
  tf*idf
}

#First step, normalize all documents via TF
train.tokens.df2<-apply(train.tokens.matrix,1,term.frequency)
dim(train.tokens.df2)
 # this above funcn transposed the matrix column
View(train.tokens.df2[1:20,1:100])

#second step, calculate the idf vector that we will use- both
#for training and test data
train.tokens.idf<-apply(train.tokens.matrix,2,inverse.doc.freq)
str(train.tokens.idf)


  
#lastly calculate the TF_idf for our training corpus
train.tokens.tfidf<-apply(train.tokens.df2,2,tf.idf,idf=train.tokens.idf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25,1:25])
# the more frequently used word will we at a top with a lower score indicating 
#its imp is low


#transpose the matrix for appling ml models to convert it in document freqencyrows
train.tokens.tfidf<-t(train.tokens.tfidf)
dim(train.tokens.tfidf)

#check for incomplete cases after invoking tf-idf cu not generating a no will envoke a error
incomplete.cases<-which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases]
# r strips symbols into empty strings

#fix incomplete cases
#replacing the incomplete cols with 0
train.tokens.tfidf[incomplete.cases,]<-rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))

#make a clean data frame using same process as before
train.tokens.tfidf.df<-cbind(Label=train$Label,data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df)<-make.names(names(train.tokens.tfidf.df))


set.seed(48749)
cv.folds<-createMultiFolds(train$Label,k=10,times=3)

cv.control<-trainControl(method="repeatedcv",number=10,
                         repeats = 3,index = cv.folds)

#for stratified we use index parameter


#note: our df is non trivial in size, as such, cv runs will take,
#quiet a long time to run. to cut down on total execution time
#we use the doSNOW package to allow for multi core training in parallel.



library(doSNOW)

 #time the code execution
start.time<-Sys.time()

#create a cluster to work on 10 logical cores.
cl<-makeCluster(2,type="SOCK")
registerDoSNOW(cl)

# as our data is non trivial in size at this point, use a single decision
#tree algo as first model .


rpart.cv.2<-train(Label~.,data=train.tokens.tfidf.df,method="rpart",
                  trControl=cv.control,tuneLength=7)

#tuneLength specifies try 7 differnt values for r and select the best 
#that is hyper parameter tuning

#processing is done, stop cluster
stopCluster(cl)

#total time of execution on workspace
total.time<-Sys.time()-start.time
total.time

#check results
rpart.cv.2
#my memory allocated to r was lo so I increased the limit to 10000
memory.limit()
memory.limit(size=15000)
summary(fit)
######################################################################################3
#N-grams allows us to augment our document -term  freq matrices with
#word ordering thus increasing efficiecy. Adding bi-grams training data
# and the tf-idf transform the expanded feature matrix to see
#if accuracy improves


#adding bi-grams to our feautr matrix.
train.tokens<-tokens_ngrams(train.tokens,n=1:2)
train.tokens[[357]]


#dfm creates a matrix of tokens
train.tokens.dfm<-dfm(train.tokens,tolower = F)

#transform to a matrix and inspect
train.tokens.matrix<-as.matrix(train.tokens.dfm)
train.tokens.dfm



#First step, normalize all documents via TF
train.tokens.df2<-apply(train.tokens.matrix,1,term.frequency)
dim(train.tokens.df2)
# this above funcn transposed the matrix column
View(train.tokens.df2[1:20,1:100])

#second step, calculate the idf vector that we will use- both
#for training and test data
train.tokens.idf<-apply(train.tokens.matrix,2,inverse.doc.freq)
str(train.tokens.idf)



#lastly calculate the TF_idf for our training corpus
train.tokens.tfidf<-apply(train.tokens.df2,2,tf.idf,idf=train.tokens.idf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25,1:25])
# the more frequently used word will we at a top with a lower score indicating 
#its imp is low


#transpose the matrix for appling ml models to convert it in document freqencyrows
train.tokens.tfidf<-t(train.tokens.tfidf)
dim(train.tokens.tfidf)

#check for incomplete cases after invoking tf-idf cu not generating a no will envoke a error
incomplete.cases<-which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases]
# r strips symbols into empty strings

#fix incomplete cases
#replacing the incomplete cols with 0
train.tokens.tfidf[incomplete.cases,]<-rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))

#make a clean data frame using same process as before
train.tokens.tfidf.df<-cbind(Label=train$Label,data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df)<-make.names(names(train.tokens.tfidf.df))

#clean up unused objects in memory
gc()

# the following code for training the model using n grams and r part with 10 folds cv with 3 repettetions will take a lot of time to execute 
#set.seed(48749)
#cv.folds<-createMultiFolds(train$Label,k=10,times=3)

#cv.control<-trainControl(method="repeatedcv",number=10,
 #                        repeats = 3,index = cv.folds)

#for stratified we use index parameter


#note: our df is non trivial in size, as such, cv runs will take,
#quiet a long time to run. to cut down on total execution time
#we use the doSNOW package to allow for multi core training in parallel.



#library(doSNOW)

#time the code execution
#start.time<-Sys.time()

#create a cluster to work on 10 logical cores.
#cl<-makeCluster(2,type="SOCK")
#registerDoSNOW(cl)

# as our data is non trivial in size at this point, use a single decision
#tree algo as first model .


#rpart.cv.3<-train(Label~.,data=train.tokens.tfidf.df,method="rpart",
           #       trControl=cv.control,tuneLength=7)

#tuneLength specifies try 7 differnt values for r and select the best 
#that is hyper parameter tuning

#processing is done, stop cluster
#stopCluster(cl)

#total time of execution on workspace
#total.time<-Sys.time()-start.time
#total.time

#check results
#rpart.cv.3


#NOTE:-the result of the above prcoseesing show slight decline in r part
#i.e accuracy decreases.As the addition of bigrams appears to -vely 
#impact a single decision tree, it'll help random forest.
###################################################################################################################

#SINGULAR VALUE DECOMPOSITION
#we'll leverage the irlba packgae for our singular value Decomposition
#the irlba package allows us to specify the no of the most imp singular 
#vectors we wish to calculate and retain for features.

#time to code execution
start.time<-Sys.time()

#Perform Svd. Specificaly reduce dimensionality down to 300 cols
#for our latent sematic analysis(LSA)
train.irlba<-irlba(t(train.tokens.tfidf),nv=300, maxit = 600)

#total time of execution on the workstation was
total.time<-Sys.time()-start.time
total.time

#take a look at the new feature data up close
View(train.irlba$v)

#as with tf-idf we will need to project new data(eg. the test data) into the svd semantic
#space. the following code illustrates how to do this using a row of training data
#that has already been transformed by tf-idf, per the maths in the previous video(last section)

sigma.inverse<-1/train.irlba$d
#d-singular values
u.transpose<-t(train.irlba$u)
#u-u matrix we got from irlba
document<-train.tokens.tfidf[1,]
doc.hat<-sigma.inverse* u.transpose %*% document
#%*%-sign for matrix multiplication

#look at the 1st 10 components of projected document and the corresponding
#row in our document semantic space(i.e the V matrix)
doc.hat[1:10]
train.irlba$v[1,1:10]

# create new feature data frame using our document semantic space of 300 features
#(i.e the V matrix from SVD)

train.svd<-data.frame(Label=train$Label,train.irlba$v)

#create a cluster to work on 10 logical cores.
cl<-makeCluster(2,type="SOCK")
registerDoSNOW(cl)

# as our data is non trivial in size at this point, use a single decision
#tree algo as first model .


rpart.cv.4<-train(Label~.,data=train.svd,method="rpart",
       trControl=cv.control,tuneLength=7)

#tuneLength specifies try 7 differnt values for r and select the best 
#that is hyper parameter tuning

#processing is done, stop cluster
stopCluster(cl)

#total time of execution on workspace
total.time<-Sys.time()-start.time
total.time

#check results
rpart.cv.4

#Note-we are performing 10-folds cv repeated 3 times. This means we need to build
#30 models. we are also asking caret to try 7 differnt values of the mtry parameter
#.Next up by default rf leverages 500 trees. lastly, caret will build 1 final model
#at the end of the process with the best m try value over all training data.
#no of trees= 1*3*7*500
#it will take a lot of time a

cl<-makeCluster(2,type="SOCK")
registerDoSNOW(cl)

# using rd fiwn 10 folds cv repeated 3 times with 7 tries and no of trees be 500
rf.cv.1<-train(Label~.,data=train.svd,method="rf",
               trControl=cv.control,tuneLength=7)


#tuneLength specifies try 7 differnt values for r and select the best 
#that is hyper parameter tuning

#processing is done, stop cluster
stopCluster(cl)

#total time of execution on workspace
total.time<-Sys.time()-start.time
total.time

#check results
rf.cv.1

#drill down the results
confusionMatrix(train.svd$Label,rf.cv.1$finalModel$predicted)

#we need to improve the spam value i.e to increase the acuuracy and also increase 
#sensitivity 
########################
#now add in the features we engineered before for sms
#text length to see if it improves things.
train.svd$textLength<-train$textlength
########################
#again run the rf to check what happens with sensitivity and specificity

cl<-makeCluster(2,type="SOCK")
registerDoSNOW(cl)

# using rd fiwn 10 folds cv repeated 3 times with 7 tries and no of trees be 500
rf.cv.2<-train(Label~.,data=train.svd,method="rf",
               trControl=cv.control,tuneLength=7,
               importance=T)


#tuneLength specifies try 7 differnt values for r and select the best 
#that is hyper parameter tuning

#processing is done, stop cluster
stopCluster(cl)

#total time of execution on workspace
total.time<-Sys.time()-start.time
total.time

#check results
rf.cv.2

#drill down the results
confusionMatrix(train.svd$Label,rf.cv.1$finalModel$predicted)


#how important wasa the new feature?
library(randomForest)
varImpPlot(rf.cv.1$finalModel)
varImpPlot(rf.cv.2$finalModel)



#turns out that our textLength. feature is very predective and pushed our
#overall accuracy over the training data to 97.1%. we can also use the power of
#cosine similarity to engineer a fearure for calculating, on average, how alike each sms 
#text message is to all of the spam messages. The hypothesis here is that our use of bigrams
#tfidf,and lsa have produced a representation where ham sms messages should have low
#cosine similarities with spam sms messages and vice versa.


#using the lsa package cosine function for our calculations.
install.packages("lsa")
library(lsa)

train.similarities<-cosine(t(as.matrix(train.svd[,-c(1,ncol(train.svd))])))
# lsa doesnt work on df so converted to matrix

#next up- take each sms text message and find what the mean cosine similarity is
#for each sms text mean with each of the spam sms messages. Per our hypotheisis, 
#ham sms text messages should have relatively low cosine similarities with spam
#messages and vice versa.

spam.indexes<-which(train$Label=="spam")

train.svd$spamSimilarity<-rep(0.0,nrow(train.svd))

for(i in 1:nrow(train.svd)){
  train.svd$spamSimilarity[i]<-mean(train.similarities[i,spam.indexes])
  }

# let's visualize our results using ggplot2
ggplot(train.svd,aes(x=spamSimilarity,fill=Label))+
  theme_bw()+
  geom_histogram(binwidth = 0.05)+
  labs(y="message count", x= "mean spam message cosine similarity",
       title = "distribution of ham vs. spam using spam cosine similarity")

#performing rf on this data set

cl<-makeCluster(2,type="SOCK")
registerDoSNOW(cl)

# using rd fiwn 10 folds cv repeated 3 times with 7 tries and no of trees be 500
rf.cv.3<-train(Label~.,data=train.svd,method="rf",
               trControl=cv.control,tuneLength=7,
               importance=T)


#tuneLength specifies try 7 differnt values for r and select the best 
#that is hyper parameter tuning

#processing is done, stop cluster
stopCluster(cl)

#total time of execution on workspace
total.time<-Sys.time()-start.time
total.time

#check results
rf.cv.3

#drill down the results
confusionMatrix(train.svd$Label,rf.cv.3$finalModel$predicted)


#how important wasa the new feature?
library(randomForest)
varImpPlot(rf.cv.1$finalModel)
varImpPlot(rf.cv.2$finalModel)
varImpPlot(rf.cv.3$finalModel)

#the high importance of the cosine similarirty can also be indicative of overfitting problem
#1 -dominated all the other features
#2 low specificity i.e increased the sensitivity



#testing our model om the text data
#first stage of this verification is running the test data through our preprocessing of pipeline
#  1- tokenization
#2- lower casing
#3-stopwords removal
#4-stemming
#5- add bigrams
#6-convert n-grams to quanteda document -term freq matrix
#7- ensure test dfm has same features as train dfm


#tokenization sms text messages
test.tokens<-tokens(test$Text,what = "word",
                     remove_numbers = T,remove_punct = T,
                     remove_symbols = T,split_hyphens = T)
# we can also tokenize by characters


#lower case the tokens.
test.tokens<-tokens_tolower(test.tokens)

#use quanteda's built in stopwords list for english
#note- you should always inspect stopword list for applicability to
#your problem/domain

test.tokens<-tokens_select(test.tokens,stopwords(),
                            selection = "remove")


#Perform stemming on the tokens

test.tokens<-tokens_wordstem(test.tokens,language = "english")


#add bigrams
test.tokens<-tokens_ngrams(test.tokens,n=1:2)


#convert n grams to quanteda document term freq matrix
test.tokens.dfm<-dfm(test.tokens,tolower=F)

#wxplore the train and test quanteda dfm objects
train.tokens.dfm
test.tokens.dfm

#ensure the test dfm has the same n-grams as the training dfm as the size of test data is less than training data
 
#note- make sure the preprocessing of the text data is same as train i.e should have same colums,
#should have same meaning and orientation

test.tokens.dfm<-dfm_select(test.tokens.dfm,train.tokens.dfm)
test.tokens.matrix<-as.matrix(test.tokens.dfm)
test.tokens.dfm



#with the raw test fetures in place next is the projecting the
#term counts for the unigrams into the same tf-idf vector space
#as our training data. the process is as follows
#1- normalize each document i.e each row
#2- perform  idf multiplication using training idf values


#normalize all documnets via TF
test.tokens.df<-apply(test.tokens.matrix,1,term.frequency)
str(test.tokens.df)

#lastly calculate tf-idf for our training corpus
test.tokens.tfidf<-apply(test.tokens.df,2,tf.idf,idf=train.tokens.idf)
dim(test.tokens.tfidf)
View(test.tokens.tfidf[1:25,1:25])


#transpose the matrix
test.tokens.tfidf<-t(test.tokens.tfidf)


#fix incomplete cases
summary(test.tokens.tfidf[1,])
test.tokens.tfidf[is.na(test.tokens.tfidf)]<-0.0
summary(test.tokens.tfidf[1,])


#with the test data projected into the tf-idf vector space of the training
#data we can now to the final projection into the training lsa semantic space
#i.e the svd matrix factoriationn

test.svd.raw<-t(sigma.inverse*u.transpose %*% t(test.tokens.tfidf))


#lastly we can now build the test data frame to feed into our trained ml model
#for predictions . but firstly add label and text length

test.svd<-data.frame(Label=test$Label,test.svd.raw,textLength=test$textlength)

#next step calculate spamSimilarity for all the test document . first up,
#create a spam similarity matrix

test.similarities<-rbind(test.svd.raw,train.irlba$v[spam.indexes,])
test.similarities<-cosine(t(test.similarities))
#here we have added the data from training data which were spam bcz they 
#would help us to determine avg similarity b/w each individual test text message and spam
#transposing convert row to cols


test.svd$spamSimilarity<-rep(0.0,nrow(test.svd))

spam.cols<-(nrow(test.svd)+1):ncol(test.similarities)

for(i in 1:nrow(test.svd)){
  
  test.svd$spamSimilarity[i]<-mean(train.similarities[i,spam.cols])
}


# now we can make predictions on the test data using our trained rf
preds<-predict(rf.cv.3,test.svd)

confusionMatrix(preds,test.svd$Label)

#Note- we can identify overfitting when the models accuracy drops drastically
#on test data in reference to the training data
# in our case the accuracy dropped from around 97 to 86 which clearly specifies that
# the similarity column is overfitting the data.
#from the confusion matrix we also got that the model predicts all the
#new data as ham i.e having sensitivity of 100 but 0 specificity
#our inference is the spam similarirty is good in differentiating the spam 
#from ham on the traing data but fails to do so in the test new data




#MODEL OPTIMIZATION 

train.svd$spamSimilarity<-NULL
test.svd$spamSimilarity<-NULL


#performing rf on this data set

cl<-makeCluster(2,type="SOCK")
registerDoSNOW(cl)

# using rd fiwn 10 folds cv repeated 3 times with 7 tries and no of trees be 500
rf.cv.4<-train(Label~.,data=train.svd,method="rf",
               trControl=cv.control,tuneLength=7,
               importance=T)


#tuneLength specifies try 7 differnt values for r and select the best 
#that is hyper parameter tuning

#processing is done, stop cluster
stopCluster(cl)

#total time of execution on workspace
total.time<-Sys.time()-start.time
total.time

#check results
rf.cv.4

#drill down the results
confusionMatrix(train.svd$Label,rf.cv.3$finalModel$predicted)

#our model now predicts with an accuracy of around 96% 

#other things you can try
#1 no of n grams
#2 using other classifiers
#3 feature engineering



knitr::stitch('spam classifier oongoing.r')














