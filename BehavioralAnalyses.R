setwd('E:/Bachman/MADeEEG1/Analysis/Matlab/Behavior/csvfiles/finalSample')
rm(list = ls())

## Softmax function

# Define the optimization function
dataNLL = function(data, parameters){
  # parameters is a vector where the first element is the learning rate delta, and the
  # second element is temperature parameter tau
  # data is the combination of choices and outcomes observed for the agent Al
  likelihood = NA
  actionValues = parameters[1]*data$FaceValue + (1-parameters[1])*data$ColorValue
  exponentiatedValue = exp(actionValues*parameters[2])
  exponentiatedValue[is.infinite(exponentiatedValue)] = .Machine$double.xmax
  likelihoodOfAccepting = exponentiatedValue/(exponentiatedValue + 1)
  likelihoodOfChoice = likelihoodOfAccepting
  likelihoodOfChoice[data$Resp == 0] = 1-likelihoodOfAccepting[data$Resp == 0]
  dataNLL = -1*sum(log(likelihoodOfChoice))
}

subjNames = c(150:164,166:171,173:174,176:178,180:183,185:186,188:195) #

nSubjects = length(subjNames)
delta = c()
tau = c()
for (subject in subjNames){
  subjectData = read.csv(paste(as.character(subject),".csv",sep = ""),na.strings=c("NaN"))
  missing <- is.na(subjectData$Resp)
  subjectData<- subset(subjectData,subset= !missing)  
  results = optim(par = c(.5,.5), dataNLL, data = subjectData, method = 'L-BFGS', lower = c(0,0), upper = c(1,100))
  delta<-append(delta,results$par[1])
  tau<-append(tau,results$par[2])
}
t.test(delta,mu=.5,alternative='two.sided')


## Creating soft-max predicted choices based on overall values, face values, and color values.
values = c(-100,-70,-50,-40,-30,-20,0,20,30,40,50,70,100)
probs <- array(numeric(),c(40,13))
for (curr_subject in 1:40){
  thisDelta <- delta[curr_subject]
  thisTau <- tau[curr_subject]
  subject <- subjNames[curr_subject]
  subjectData = read.csv(paste(as.character(subject),".csv",sep = ""),na.strings=c("NaN"))
  missing <- is.na(subjectData$Resp)
  subjectData<- subset(subjectData,subset= !missing)  
  
  for (currVal in 1:13){
    thisVal = values[currVal]
    thisData <- subset(subjectData,subjectData$Value==thisVal)
    
    actionValues = thisDelta*thisData$FaceValue + (1-thisDelta)*thisData$ColorValue
    exponentiatedValue = exp(actionValues*thisTau)
    exponentiatedValue[is.infinite(exponentiatedValue)] = .Machine$double.xmax
    likelihoodOfAccepting = exponentiatedValue/(exponentiatedValue + 1)
    likelihoodOfChoice = likelihoodOfAccepting
    likelihoodOfChoice[thisData$Resp == 0] = 1-likelihoodOfAccepting[thisData$Resp == 0]
    probs[curr_subject,currVal] = mean(likelihoodOfAccepting)
  }
}
write.csv(probs,"softmax_predicted2.csv", row.names = FALSE)


values = c(-50,-20,0,20,50)
face_probs <- array(numeric(),c(40,5))
for (curr_subject in 1:40){
  thisDelta <- delta[curr_subject]
  thisTau <- tau[curr_subject]
  subject <- subjNames[curr_subject]
  subjectData = read.csv(paste(as.character(subject),".csv",sep = ""),na.strings=c("NaN"))
  missing <- is.na(subjectData$Resp)
  subjectData<- subset(subjectData,subset= !missing)  
  for (currVal in 1:5){
    thisVal = values[currVal]
    thisData <- subset(subjectData,subjectData$FaceValue==thisVal)
    
    actionValues = thisDelta*thisData$FaceValue + (1-thisDelta)*thisData$ColorValue
    exponentiatedValue = exp(actionValues*thisTau)
    exponentiatedValue[is.infinite(exponentiatedValue)] = .Machine$double.xmax
    likelihoodOfAccepting = exponentiatedValue/(exponentiatedValue + 1)
    likelihoodOfChoice = likelihoodOfAccepting
    likelihoodOfChoice[thisData$Resp == 0] = 1-likelihoodOfAccepting[thisData$Resp == 0]
    face_probs[curr_subject,currVal] = mean(likelihoodOfAccepting)
  }
}
write.csv(face_probs,"softmax_predicted2_Faces.csv", row.names = FALSE)

color_probs <- array(numeric(),c(40,5))
for (curr_subject in 1:40){
  thisDelta <- delta[curr_subject]
  thisTau <- tau[curr_subject]
  subject <- subjNames[curr_subject]
  subjectData = read.csv(paste(as.character(subject),".csv",sep = ""),na.strings=c("NaN"))
  missing <- is.na(subjectData$Resp)
  subjectData<- subset(subjectData,subset= !missing)  
  for (currVal in 1:5){
    thisVal = values[currVal]
    thisData <- subset(subjectData,subjectData$ColorValue==thisVal)
    
    actionValues = thisDelta*thisData$FaceValue + (1-thisDelta)*thisData$ColorValue
    exponentiatedValue = exp(actionValues*thisTau)
    exponentiatedValue[is.infinite(exponentiatedValue)] = .Machine$double.xmax
    likelihoodOfAccepting = exponentiatedValue/(exponentiatedValue + 1)
    likelihoodOfChoice = likelihoodOfAccepting
    likelihoodOfChoice[thisData$Resp == 0] = 1-likelihoodOfAccepting[thisData$Resp == 0]
    color_probs[curr_subject,currVal] = mean(likelihoodOfAccepting)
  }
}
write.csv(color_probs,"softmax_predicted2_Colors.csv", row.names = FALSE)

## Plotting data
attribute_values = c(-50,-20,0,20,50)
overall_values <- c(-100 ,-70,-50,-40,-30,-20,0,20,30,40,50,70,100)
face_responses  <-matrix(NA,nrow=40,ncol=5)
color_responses  <-matrix(NA,nrow=40,ncol=5)
overall_responses <-matrix(NA,nrow=40,ncol=13)

# First, collate participant's actual responses.
for (curr_subject in 1:40){
  subject <- subjNames[curr_subject]
  subjectData = read.csv(paste(as.character(subject),".csv",sep = ""),na.strings=c("NaN"))
  missing <- is.na(subjectData$Resp)
  subjectData<- subset(subjectData,subset= !missing) 
  for (curr_value in 1:13) {
    overall_responses[curr_subject,curr_value] <- mean(subjectData$Resp[subjectData$Value == overall_values[curr_value]], na.rm = TRUE)
  }
  for (curr_value in 1:5) {
    face_responses[curr_subject,curr_value] <- mean(subjectData$Resp[subjectData$FaceValue == attribute_values[curr_value]], na.rm = TRUE)
    color_responses[curr_subject,curr_value] <- mean(subjectData$Resp[subjectData$ColorValue == attribute_values[curr_value]], na.rm = TRUE)
  }
}

# Determining accuracy (on trials where the sum ~= 0)
overall_acc <- overall_responses                      # Copy the matrix
overall_acc[, 1:6] <- 1 - overall_responses[, 1:6]    # Invert negative values
overall_acc <- overall_acc[, -7]                      # Remove the indifference point
mean(rowMeans(overall_acc, na.rm = TRUE), na.rm = TRUE) # Mean accuracy = .868            
sd(rowMeans(overall_acc, na.rm = TRUE), na.rm = TRUE)   # std accuracy = .079

# Now plot the actual responses compared to the softmax predicted ones.
# Load required libraries

library(ggplot2)
library(gridExtra)  

# Compute summary statistics
n <- length(subjNames)
overall_mean <- colMeans(overall_responses, na.rm = TRUE)
overall_se <- apply(overall_responses, 2, sd, na.rm = TRUE) / sqrt(n)

softmax_mean <- colMeans(probs, na.rm = TRUE)
softmax_se <- apply(probs, 2, sd, na.rm = TRUE) / sqrt(n)

# Create data frame for overall plot
overall_df <- data.frame(
  Value = overall_values,
  Human = overall_mean,
  Human_SE = overall_se,
  Softmax = softmax_mean,
  Softmax_SE = softmax_se
)

# Plot 1: Overall Value
p1 <- ggplot(overall_df, aes(x = Value)) +
  geom_errorbar(aes(ymin = Human - Human_SE, ymax = Human + Human_SE), color = "green", width = 5, size = 1) +
  geom_point(aes(y = Human), color = "green", shape = 16, size = 3) +
  geom_line(aes(y = Human), color = "green", size = 1) +
  geom_errorbar(aes(ymin = Softmax - Softmax_SE, ymax = Softmax + Softmax_SE), color = "cyan", width = 5, size = 1) +
  geom_point(aes(y = Softmax), color = "cyan", shape = 16, size = 3) +
  geom_line(aes(y = Softmax), color = "cyan", size = 1) +
  xlim(-110, 110) +
  scale_x_continuous(breaks = overall_values) +
  ylim(0, 1) +
  labs(title = "Overall Value", x = "Total points of displayed option", y = 'Proportion of "Accept" responses') +
  theme_minimal(base_size = 14)

# --- Plot 2: Attribute-based responses ---
# Compute means and SEs
face_mean <- colMeans(face_responses, na.rm = TRUE)
face_se <- apply(face_responses, 2, sd, na.rm = TRUE) / sqrt(n)

color_mean <- colMeans(color_responses, na.rm = TRUE)
color_se <- apply(color_responses, 2, sd, na.rm = TRUE) / sqrt(n)

softmax_face_mean <- colMeans(face_probs, na.rm = TRUE)
softmax_face_se <- apply(face_probs, 2, sd, na.rm = TRUE) / sqrt(n)

softmax_color_mean <- colMeans(color_probs, na.rm = TRUE)
softmax_color_se <- apply(color_probs, 2, sd, na.rm = TRUE) / sqrt(n)

# Create data frame for attribute plot
attribute_df <- data.frame(
  Value = rep(attribute_values, 4),
  Mean = c(face_mean, color_mean, softmax_face_mean, softmax_color_mean),
  SE = c(face_se, color_se, softmax_face_se, softmax_color_se),
  Condition = factor(rep(c("Face", "Color", "Softmax Face", "Softmax Color"), each = length(attribute_values)))
)

# Plot 2: Face/Color Value
p2 <- ggplot(attribute_df, aes(x = Value, y = Mean, color = Condition, group = Condition)) +
  geom_errorbar(aes(ymin = Mean - SE, ymax = Mean + SE), width = 3, size = 1) +
  geom_point(shape = 16, size = 3) +
  geom_line(size = 1) +
  scale_color_manual(values = c("blue", "red", "cyan", "magenta")) +
  xlim(-55, 55) +
  scale_x_continuous(breaks = attribute_values) +
  ylim(0, 1) +
  labs(title = "Face/Color Value", x = "Total points of displayed option", y = 'Proportion of "Accept" responses') +
  theme_minimal(base_size = 14)





# Arrange the two plots side by side
grid.arrange(p1, p2, ncol = 2)

# Optional: Save to EPS
ggsave("ChoiceAccept_n40_v2_wSoftMax.eps", plot = grid.arrange(p1, p2, ncol = 2), device = "eps", width = 12, height = 6)
