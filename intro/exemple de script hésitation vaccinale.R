#####
# Illustration Hésitation à la vaccination du covid-19 aux Etats-Unis 
####

library(tidyverse)
library(dplyr)
library(broom)
library (janitor)
library(ggplot2)
library(rsample)
library(caret)
library(ROCR)
library(vip)
library(pdp)
library(ggstats)
library(broom.helpers)
library(RcmdrMisc) #stepwise  

load("C:/Users/houee/AOBOX/Enseignement/sciencesEco/modeles_choix_discrets/data/vh_data.Rdata")

vh_data14 %>%
  tabyl(vaccine_hesitant)

# qq détails sur les variables : 
# ns index : Natural Science Literacy which captures how scientifically literate
#  an individual is, as we expect this to impact how they process 
# and interpret vaccine information
# ps index : politically sophisticated
# variables that capture the chances of getting infected by modeling whether they have 
# already been infected (Infected Personal: 0=No;1=Yes),whether someone in their immediate
# network has been infected (Infected Network: 0=No;1=Yes), the proportion of people in the county 
# in which they reside that have been infected(County Cases), the proportion of 
# people in their county that have been infected in the past two weeks(County Cases2wk),
# and the population density of their county(County Density)
# Race (1 =white;2=Black;3=Hispanic;4=other
# Male (0 =No;1=Yes); College Degree (1 =Yes;0=No); Household Income (1–12).      

# anticipate that financial and mental health-based incentives
# could play a large role in motivating vaccine acceptance/hesitancy.

# vaccine_hesitant =0 si deja vaccine, =1 si hesitation

# convertir en facteur les variables indicatrices 
vh_data14 <- vh_data14 %>% mutate_at(vars(condition_pregnant, condition_asthma, condition_lung, condition_diabetes, condition_immune, condition_obesity,
                                          condition_heart, condition_organ, psindex, nsindex, college, infected_personal, infected_network, evangelical),
                                     list(factor))


## stat descriptive
####################
ggplot(vh_data14, aes(x=vaccine_hesitant,fill=vaccine_hesitant)) +
  geom_bar() +
  ggtitle("Hésitation vaccinale")

ggplot(vh_data14, aes(y=age,x=vaccine_hesitant,col=vaccine_hesitant)) +
  geom_boxplot() +
  ggtitle("Hésitation vaccinale selon l'âge")

ggplot(vh_data14, aes(x=male, fill=male)) + 
  geom_bar(aes(y = after_stat(count / sum(count)))) + 
  ggtitle("Hésitation vaccinale selon le sexe") + 
  facet_wrap(facets = ~ vaccine_hesitant) + 
  ylab("proportion")

## Regression logistique 

set.seed(1234)
# modele complet
vaccine_mod_complet <- glm (vaccine_hesitant ~ ., family = "binomial", na.action = na.omit,data = vh_data14)
vaccine_mod_complet
summary(vaccine_mod_complet)
questionr::odds.ratio(vaccine_mod_complet)
ggstats::ggcoef_model(vaccine_mod_complet, exponentiate = TRUE)

# deviance (residual deviance)
deviance(vaccine_mod_complet)
# Type-II Table d'analyse de la déviance 
M0 <- glm(vaccine_hesitant~1, family = "binomial", na.action = na.omit,data=vh_data14)
anova(M0,vaccine_mod_complet,test="Chisq")

# R2 de MacFadden (privilégier le rapport de vraisemblance)
PseudoR2 <- 1-vaccine_mod_complet$deviance/vaccine_mod_complet$null.deviance
PseudoR2 

# Wald's z-tests significativité des coef
summary(vaccine_mod_complet)$coefficients
# interval de confiance à 0.95
confint.default(vaccine_mod_complet)
# Type-II Table d'analyse de la déviance 
car::Anova(vaccine_mod_complet)

### Selection de modèles
##########################

# Stepwise (k>15 variables)
mod_step = stepwise(vaccine_mod_complet,direction="forward/backward",criterion="AIC")
summary(mod_step)$coefficients
ggcoef_model(mod_step, exponentiate = TRUE)
mod_step_bic = stepwise(vaccine_mod_complet,direction="forward/backward",criterion="BIC")
summary(mod_step_bic)$coefficients

### prediction et performance 
probas = predict(mod_step,type="response")
plot(probas~vh_data14$vaccine_hesitant,bty="l",xlab="Vaccine Hesitant",
     ylab="Probabilité estimée",cex.lab=1.25,
     cex.axis=1.25,cex.main="1.25",pch=16,
     main="Probabilité d'hésitation ")

# règle de classification de Bayes
pred.class = ifelse(probas>=0.5,"1-Hesitant","0-non hesitant")
confusion = table(vh_data14$vaccine_hesitant,pred.class,dnn=list("Observé","Predit"))
confusion

taux_bon_classement = (2296 + 543)/3153 *100 ; taux_bon_classement
sensibilite = 543/(543+212) ; sensibilite
specificite = 2296/(2296+102); specificite
# notre obj est d'améliorer la sensibilité


# ou avec le package caret 
pred.class = ifelse(probas>=0.5,"1","0") # recoder pred.class
caret::confusionMatrix(factor(pred.class),vh_data14$vaccine_hesitant,mode="everything",positive = "1")

# Courbe ROC
pred = prediction(predictions=probas,labels=vh_data14$vaccine_hesitant)
perf = performance(pred,measure="tpr",x.measure="fpr")
plot(perf,lwd=2,col="blue")
# AUC : Area under the ROC curve
performance(pred,measure="auc")@"y.values"[[1]]


#Variable Importance Plot
vip(mod_step,num_features = 10)


# validation croisée avec caret
###############################
#Data Partition
library(tidymodels)
set.seed(2345)
dataset_split <- initial_split (vh_data14, prop=0.75, strata = vaccine_hesitant)
training_set<- training(dataset_split)
test_set <- testing (dataset_split)

# sur le modele complet
cv_Vaccine <- train(
  vaccine_hesitant ~ .,  
  data = training_set,
  method= "glm",
  family= stats::binomial(link = "logit"),
  na.action = na.exclude,
  trControl= trainControl(method = "cv", number = 10)
)
# en selectionnant qq variables 
cv_Vaccine <- train(
  vaccine_hesitant ~ vaccine_trust + age + perceived_network_risk + trust_gov_local + college + condition_immune +
    doctor_comfort + biden + trust_media + race + trust_science_politicians + trust_gov_state +
    trust_science_media + infected_network + income,  
  data = training_set,
  method= "glm",
  family= stats::binomial(link = "logit"),
  na.action = na.exclude,
  trControl= trainControl(method = "cv", number = 10)
)
summary(cv_Vaccine)

#Predict class
pred_class <- predict(cv_Vaccine, test_set)
head(pred_class)
table(pred_class)
#Confusion Matrix
confusionMatrix(
  data = relevel(pred_class, ref = "1"),
  reference = relevel(test_set$vaccine_hesitant, ref = "1")
)



## selection et validation croisée (en incluant la selection à chaque segment)
#######################################

library(groupdata2)
folds = fold(vh_data14,k=10,cat_col="vaccine_hesitant")$".folds" # Creation des segments
folds
table(vh_data14$vaccine_hesitant,folds)
cvpredictions = rep("1",nrow(vh_data14)) # Initialisation du vecteur de classes prédites
for (j in 1:10) {
  train = vh_data14[folds!=j,]
  test = vh_data14[folds==j,]
  mod = glm(vaccine_hesitant~.,family = "binomial",data=train,trace=FALSE,maxit=200)
  select = stepwise(mod,direction="forward/backward",criterion="AIC",trace=0)
  #print(summary(select)) 
  cvpredictions[folds==j] = predict(select,newdata=test,type="response")
  print(paste("Segment ",j,sep=""))
}
# Step 3: performance critère cross validé
cvpred = prediction(predictions=as.numeric(cvpredictions),labels=vh_data14$vaccine_hesitant)
cvAUC = performance(cvpred,measure="auc")@"y.values"[[1]]
cvAUC
#Cette AUC cross-validée est la plus fiable, elle n'est pas sur-ajustée.  
