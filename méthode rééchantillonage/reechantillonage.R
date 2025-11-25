library(tidyverse)
library(dplyr)
library(ggplot2)
library(rsample)
library(ROCR)
library(caret)

# Data
### 
load("vh_data.Rdata")

# convertir en facteur les variables indicatrices 
vh_data14 <- vh_data14 %>% 
  mutate_at(vars(condition_pregnant, condition_asthma, condition_lung, 
                 condition_diabetes, condition_immune, condition_obesity,
                 condition_heart, condition_organ, psindex, nsindex, 
                 college, infected_personal, infected_network, evangelical),
            list(factor))

vh_data14 %>%
  janitor::tabyl(vaccine_hesitant)



set.seed(1234)

# creation jdd apprentissage et jdd test
perm <- sample(nrow(vh_data14),round(0.7*nrow(vh_data14),0))
app <- vh_data14[perm,]
test <- vh_data14[-perm,]


logit_complet <- stats::glm(vaccine_hesitant~.,data=app,family=binomial)
summary(logit_complet)
# selection de variables 
logit_step <- stats::step(logit_complet,direction="backward", trace=0)

# sous-modèle sélectionné : 
formula(logit_step) 
# vaccine_hesitant ~ trust_gov_state + trust_gov_local + perceived_network_risk + 
#  doctor_comfort + condition_asthma + condition_immune + race + 
#  age + nsindex + college + biden + vaccine_trust + trust_science_polmotives + 
#  trust_science_politicians

# critères de performance calculés sur le jeu de données test
probas = stats::predict(logit_step,newdata=test,type="response")
# Cross-validated AUC
pred = ROCR::prediction(predictions=probas,labels=test$vaccine_hesitant)
AUC = ROCR::performance(pred,measure="auc")@"y.values"[[1]]
AUC
predtest.class = ifelse(probas>=0.5,"1","0")
caret::confusionMatrix(factor(predtest.class),test$vaccine_hesitant,mode="everything",positive = "1")

# sensibilite (taux vrais positifs) 0.75 à améliorer 
# specificite (tx vrais negatifs)  0.96 les négatifs sont très bien détectés
# taux de bon classement 0.91 résultat trompeur car ici la prévalence est faible 21.9% (part d'hésitant)
# Neg Pred Value élevé : qd l'evt negatif est prédit, on peut faire confiance au modele
# Kappa 0.73 le modele apporte par rapport au hasard

# Pos Pred value 0.82 correct mais pas exceptionnel non plus

# bon modele pour minimiser les faux positifs
# fonctionne bien pour prédire ce qui est négatif mais moins bien pour repérer ce qui est positif


##################################
#### Methodes echantillonnage ####
##################################

set.seed(1234)

##=============================
## random oversampling
##=============================
?caret::upSample # dupications aléatoires
newtrain2 <- caret::upSample(x = app[,-ncol(app)], y = app$vaccine_hesitant)
summary(newtrain2$Class); names(newtrain2)[ncol(newtrain2)]<- "vaccine_hesitant"

summary(app)[,c(1:3,39)]
summary(newtrain2)[,c(1:3,39)]
dim(newtrain2)
dim(unique(newtrain2)) ;dim(app)
# ex. pour visualiser les lignes dupliquées
dup_idx <- duplicated(newtrain2[,-ncol(newtrain2)]) 
newtrain2[dup_idx, ]
# sinon ajouter un identifiant avant de ré-échantillonner, et voir ensuite ceux qui sont dupliqués

##=============================
## SMOTE
##=============================
# 
library(recipes)
library(themis)
rec <- recipes::recipe(vaccine_hesitant ~ ., data = app) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(all_outcomes(), neighbors = 4, over_ratio = 1)

# Préparation + transformation
rec_prep <- recipes::prep(rec)
# Données entraînement après sur-échantillonnage
train_smote <- recipes::bake(rec_prep, new_data = NULL)
summary(train_smote$vaccine_hesitant)
# Données test transformées (sans resampling)
test_transformed <- recipes::bake(rec_prep, new_data = test)
summary(test_transformed$vaccine_hesitant)

## aperçu des individus ajoutés 
# au préalable il faut modifier app de la mm façon que train smote cad transformer les qualitatives
rec_no_smote <- recipes::recipe(vaccine_hesitant ~ ., data = app) %>%
  step_dummy(all_nominal_predictors())
rec_no_smote_prep <- recipes::prep(rec_no_smote)
app_transformed <- recipes::bake(rec_no_smote_prep, new_data = NULL)
ajouts <- anti_join(train_smote, app_transformed, by = names(app_transformed))
nrow(ajouts)   # nombre d'individus créés par SMOTE
head(ajouts)   # aperçu des individus synthétiques
## 

##=============================
## Adasyn
##=============================
recada <- recipes::recipe(vaccine_hesitant ~ ., data = app) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_adasyn(all_outcomes(), neighbors = 4, over_ratio = 1)

# Préparation + transformation
recada_prep <- recipes::prep(recada)
# Données entraînement après sur-échantillonnage
train_adasyn <- recipes::bake(recada_prep, new_data = NULL)
summary(train_adasyn$vaccine_hesitant)
# Données test transformées (sans resampling)
test_transformed <- recipes::bake(recada_prep, new_data = test)
summary(test_transformed$vaccine_hesitant)

# 
#vaccine_hesitant ~ individual_responsibility + trust_gov_state + 
#  trust_gov_local + trust_media + perceived_network_risk + 
#  doctor_comfort + age + vaccine_trust + trust_science_politicians + 
#  trust_science_media + income + condition_asthma_X1 + condition_lung_X1 + 
#  condition_immune_X1 + condition_obesity_X1 + race_X2 + college_X1 + 
#  infected_network_X1 + biden_Yes

##=============================
## random undersampling
##=============================
newtrain3 <- caret::downSample(x = app[,-ncol(app)], y = app$vaccine_hesitant)
summary(newtrain3$Class); names(newtrain3)[ncol(newtrain3)]<- "vaccine_hesitant"

## exemple : si on veut contrôler le ratio il faut le faire manuellement ou utiliser d'autres packages
# Séparer les classes
class0 <- app[app$vaccine_hesitant == "0", ]
class1 <- app[app$vaccine_hesitant == "1", ]
# Garder 50% de la classe 0
set.seed(123)
class0_under <- class0[sample(1:nrow(class0), size = 0.5*nrow(class0)), ]
# Garder 100% de la classe 1
class1_under <- class1
newtrain3b <- rbind(class0_under, class1_under)
summary(newtrain3b$vaccine_hesitant)
## 


##=============================
## tomek
##=============================
rectomek <- recipes::recipe(vaccine_hesitant ~ ., data = app) %>%
  step_dummy(all_nominal_predictors()) %>%
  themis::step_tomek(all_outcomes())
# Préparation + transformation
rectomek_prep <- recipes::prep(rectomek)
# Données entraînement après sur-échantillonnage
train_tomek <- recipes::bake(rectomek_prep, new_data = NULL)
summary(train_tomek$vaccine_hesitant); summary(app$vaccine_hesitant)
# Données test transformées (sans resampling)
test_transformed <- recipes::bake(rectomek_prep, new_data = test)
summary(test_transformed$vaccine_hesitant)


##=============================
## class weights
##=============================
# poids inversement proportionnel à la fréquence de sa classe
library(glmnet)
Xtrain <- model.matrix(logit_step, data=app)[,-1]
ytrain <- app$vaccine_hesitant
Xtest <- model.matrix(logit_step, data=test)[,-1]

# calcul des poids inverses de fréquence
w0 <- nrow(app) / (2 * sum(ytrain == 0))
w1 <- nrow(app) / (2 * sum(ytrain == 1))
weights <- ifelse(ytrain == 0, w0, w1)

# régression logistique pénalisée avec poids
cvfit <- cv.glmnet(Xtrain, ytrain, family="binomial", weights=weights)
cvprobas <- predict(cvfit, newx=Xtest, type="response", s="lambda.min")

cvpred = ROCR::prediction(predictions=cvprobas,labels=test$vaccine_hesitant)
predtest.class = ifelse(cvprobas>=0.5,"1","0")
caret::confusionMatrix(test$vaccine_hesitant,factor(predtest.class),mode="everything",positive = "1")


#################################################### 
# Choix d'un algorithme par validation croisée
# Choisir un algorithme parmi les précédents en utilisant comme critère l’erreur de classification ainsi que la courbe ROC et l’AUC. 
# On pourra faire une validation croisée 10 folds

#On définit d’abord les 10 folds pour la validation croisée.
set.seed(1234)
fold <- sample(1:10,nrow(vh_data14),replace=TRUE)
table(fold)

#score <- data.frame(matrix(0,nrow=nrow(bank),ncol=1))
score_logit <- rep(0,nrow(vh_data14))
SCORE <- list(init=score_logit,over=score_logit,under=score_logit, 
              smote=score_logit, tomek=score_logit, adasyn=score_logit)
# validation croisée, pour chaque valeur de k entre 1 et 10 :
# calcule les échantillons d’apprentissage et de test 

for (k in 1:10){
  print(k)
  ind.test <- fold==k
  df.app <- vh_data14[!ind.test,]
  df.test <- vh_data14[ind.test,]
  # on ré-équilibre les données d’apprentissage uniquement 
  
  ech.app <- list(norm=df.app,
                  over=caret::upSample(x = df.app[,-ncol(df.app)], y = df.app$vaccine_hesitant),
                  under=caret::downSample(x = df.app[,-ncol(df.app)], y = df.app$vaccine_hesitant))
  names(ech.app[[2]])[ncol(ech.app[[2]])] <- "vaccine_hesitant"
  names(ech.app[[3]])[ncol(ech.app[[3]])] <- "vaccine_hesitant"
  
  for (j in 1:3){ 
    mod.logit <- glm(vaccine_hesitant ~ trust_gov_state + trust_gov_local + perceived_network_risk + 
                       doctor_comfort + condition_asthma + condition_immune + race + 
                       age + nsindex + college + biden + vaccine_trust + trust_science_polmotives + 
                       trust_science_politicians,data=ech.app[[j]],family="binomial")
    
    # on calcule enfin les probabilités estimées 
    SCORE[[j]][ind.test] <- predict(mod.logit,newdata=df.test,type="response")
  }
  
  rec <- recipes::recipe(vaccine_hesitant ~ ., data = df.app) %>%
    step_dummy(all_nominal_predictors())  %>%
    step_smote(all_outcomes(), neighbors = 4, over_ratio = 1)
  rec_prep <- recipes::prep(rec)
  train_smote <- recipes::bake(rec_prep, new_data = NULL)
  test_transformed <- recipes::bake(rec_prep, new_data = df.test)
  
  rectomek <- recipes::recipe(vaccine_hesitant ~ ., data = df.app) %>%
    step_dummy(all_nominal_predictors()) %>%
    themis::step_tomek(all_outcomes())
  rectomek_prep <- recipes::prep(rectomek)
  train_tomek <- recipes::bake(rectomek_prep, new_data = NULL)
  test_transformed <- recipes::bake(rectomek_prep, new_data = df.test)
  
  recada <- recipes::recipe(vaccine_hesitant ~ ., data = df.app) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_adasyn(all_outcomes(), neighbors = 4, over_ratio = 1)
  recada_prep <- recipes::prep(recada)
  train_adasyn <- recipes::bake(recada_prep, new_data = NULL)
  test_transformed <- recipes::bake(recada_prep, new_data = df.test)
  
  ech.app <- c(ech.app, list(smote=train_smote, tomek=train_tomek, adasyn=train_adasyn))
  
  for (j in 4:6){
    mod.logit2 <- glm(vaccine_hesitant ~ individual_responsibility + trust_gov_state + 
                        trust_gov_local + trust_media + perceived_network_risk + 
                        doctor_comfort + age + vaccine_trust + trust_science_politicians + 
                        trust_science_media + income + condition_asthma_X1 + condition_lung_X1 + 
                        condition_immune_X1 + condition_obesity_X1 + race_X2 + college_X1 + 
                        infected_network_X1 + biden_Yes, data=ech.app[[j]],family="binomial")
    
    SCORE[[j]][ind.test] <- predict(mod.logit2,newdata=test_transformed,type="response")
  }
}

acc<-NULL
bal_acc<-NULL
F1 <- NULL
kap<-NULL
sensi <- NULL
for (j in 1:6){
  predtest.class = factor(ifelse(SCORE[[j]]>=0.5,"1","0"))
  acc[j] <- caret::confusionMatrix(vh_data14$vaccine_hesitant,predtest.class,mode="everything",positive = "1")$overall[1]
  bal_acc[j] <- caret::confusionMatrix(vh_data14$vaccine_hesitant,predtest.class,mode="everything",positive = "1")$byClass[11]
  F1[j] <- caret::confusionMatrix(vh_data14$vaccine_hesitant,predtest.class,mode="everything",positive = "1")$byClass[7]
  kap[j] <- caret::confusionMatrix(vh_data14$vaccine_hesitant,predtest.class,mode="everything",positive = "1")$overall[2]
  sensi[j]<- caret::confusionMatrix(vh_data14$vaccine_hesitant,predtest.class,mode="everything",positive = "1")$byClass[1]
}

tt <- data.frame(accuracy=round(acc,4), balanced_accuracy=round(bal_acc,4), F1_score=round(F1,4), Kappa = round(kap,4), Sensibilite = round(sensi,4))
rownames(tt) <- c("donnees_brutes","over","under","smote","tomek","adasyn")
tt

# commentairesà modifier : 
# l’apport des méthodes de ré-échantillonnage est discutable sur cet exemple. 
# Le ré-équilibrage peut néanmoins améliorer certains critères 
# dans cet exemple, la méthode tomek améliore la sensibilité et ne déteriore pas les autres
# critères par rapport au scénario de base. 
# cette méthode serait celle à privilégier dans cet exemple.