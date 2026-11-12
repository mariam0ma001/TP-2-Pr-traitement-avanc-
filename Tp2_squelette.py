#PARTIE 1:
#1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler , StandardScaler

m = pd.read_csv("transactions_TP2.csv")

print(m.head())

print("Dimensions :", m.shape)

print(m.dtypes)

#2
manq=pd.DataFrame({
    "le nombre":m.isnull().sum(),
    "le pourcentage":m.isnull().mean()*100
})
manq=manq.sort_values(by="le pourcentage",ascending=False)
print(manq.round(2))
#3
print("le nombre des doublens en id_operation :",m["id_operation"].duplicated().sum())
m = m.drop_duplicates(subset="id_operation")
print("Dimensions après suppression :", len(m))
#4
plt.figure(figsize=(8,3))

sns.histplot(data=m, x="montant_dh", bins=30, kde=True)

plt.title("Distribution des montants des transactions")
plt.xlabel("Montant (DH)")
plt.ylabel("Fréquence")

plt.show()

plt.figure(figsize=(8,3))

sns.boxplot(x=m["montant_dh"])

plt.title("Boxplot des montants des transactions")
plt.xlabel("Montant (DH)")

plt.show()
#5
print(m["fraude"].value_counts())
print(m["fraude"].value_counts(normalize=True) * 100)

m["fraude"].value_counts().plot(kind="bar")

plt.title("Répartition des fraudes")
plt.xlabel("Fraude")
plt.ylabel("Nombre")
plt.show()

#PARTIE 2:
#7
col_brra = ["id_operation","id_client","resultat_enquete"]
x = m.drop(columns=col_brra + ["fraude"])
y = m["fraude"]
print("dimension de x :",x.shape)
print("dimension de y :",y.shape)
print("variales de x :")
print(x.head())
print("variables de y :")
print(y.head())

#8_train_test_slpit
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split( x, y, test_size=0.2, random_state=42, stratify=y)

print("train :",x_train.shape)
print("test :",x_test.shape)
print("fraude dans train :")
print(y_train.value_counts(normalize=True)*100)

print("fraude dans test :")
print(y_test.value_counts(normalize=True)*100)
#9
num_cols=x_train.select_dtypes(include=np.number).columns.tolist()
cat_cols=x_train.select_dtypes(include="str").columns.tolist()
print("variables numeriques :",num_cols)
print("variables categorielles :",cat_cols)

#10
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
num_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
num_pipeline.fit(x_train[num_cols])
print("medianes :")
print(pd.Series(num_pipeline.named_steps["imputer"].statistics_,index=num_cols))
print("moyennes :")
print(pd.Series(num_pipeline.named_steps["scaler"].mean_,index=num_cols))
print("echelles :")
print(pd.Series(num_pipeline.named_steps["scaler"].scale_,index=num_cols))

#11
from sklearn.preprocessing import OneHotEncoder
cat_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(handle_unknown="ignore"))])
#12
from sklearn.compose import ColumnTransformer
traitement=ColumnTransformer(transformers=[("num", num_pipeline,num_cols),("cat",cat_pipeline,cat_cols)])
x_train_prepared=traitement.fit_transform(x_train)
x_test_prepared=traitement.transform(x_test)

print("avant traitemnt :")
print("train :",x_train.shape)
print("test :",x_test.shape)
print("apres traitement :")
print("train :",x_train_prepared.shape)
print("test :",x_test_prepared.shape)
#13
from sklearn.preprocessing import RobustScaler
robust_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),("scaler",RobustScaler())])
robust_pipeline.fit(x_train[num_cols])

montant=x_train["montant_dh"].dropna()
indice_normal=(montant-montant.median()).abs().idxmin()
indice_max=montant.idxmax()

comparison = x_train.loc[[indice_normal, indice_max], num_cols]

resultat_standard=num_pipeline.transform(comparison)
resultat_robust=robust_pipeline.transform(comparison)
i=num_cols.index("montant_dh")
resultat=pd.DataFrame({"montant original":comparison["montant_dh"].values,"standardscaler":resultat_standard[:,i],"robustscaler":resultat_robust[:,i]},
                      index=["operation normale","operation elvee"])
print(resultat)



#PARTIE 3:
#15
Q1 = x_train["montant_dh"].quantile(0.25)
Q3 = x_train["montant_dh"].quantile(0.75)

IQR = Q3 - Q1

print("Q1 :", Q1)
print("Q3 :", Q3)
print("IQR :", IQR)
#16
lower_lim = Q1 - 1.5 * IQR
upper_lim = Q3 + 1.5 * IQR

print("limite inferieure :", lower_lim)
print("limite superieure :", upper_lim)
outliers=x_train[(x_train["montant_dh"]<lower_lim) | (x_train["montant_dh"]>upper_lim)]
print("nombre des outliers :",len(outliers))
print(outliers[["montant_dh"]].head())
#17
montant_limite=x_train["montant_dh"].clip(lower=lower_lim,upper=upper_lim)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.boxplot(montant)#montant=x_train["montant_dh"].dropna()
plt.title("montants originaux")

plt.subplot(1,2,2)
plt.boxplot(montant_limite.dropna())
plt.title("montants limites")
plt.show()

#18 minmax standardscaler robustscaler
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
minmax=MinMaxScaler()#[0,1]
standard=StandardScaler()#moy et ecart type
robust=RobustScaler()#moy et IQR

m_train_minmax=minmax.fit_transform(x_train[["montant_dh"]])
m_test_minmax=minmax.transform(x_test[["montant_dh"]])

m_train_standard=standard.fit_transform(x_train[["montant_dh"]])
m_test_standard=standard.transform(x_test[["montant_dh"]])

m_train_robust=robust.fit_transform(x_train[["montant_dh"]])
m_test_robust=robust.transform(x_test[["montant_dh"]])

#19 Comparer les montants avant et apres transformation
montants_norm=montant.iloc[:3]
montants_elev=montant.loc[[montant.idxmax()]]
selection=pd.concat([montants_norm,montants_elev])

r = pd.DataFrame({
    "Montant original": selection.values,
    "MinMaxScaler": minmax.transform(selection.to_frame()).ravel(),
    "StandardScaler": standard.transform(selection.to_frame()).ravel(),
    "RobustScaler": robust.transform(selection.to_frame()).ravel()
})

print(r)
a = pd.DataFrame({
    "MinMaxScaler": MinMaxScaler().fit_transform(montants_norm.to_frame()).ravel(),
    "StandardScaler": StandardScaler().fit_transform(montants_norm.to_frame()).ravel(),
    "RobustScaler": RobustScaler().fit_transform(montants_norm.to_frame()).ravel()
})

diff = (r.iloc[:3, 1:] - a).abs()

print(diff)
print(diff.mean())

#20les histogrammes train
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.hist(m_train_minmax, bins=30)
plt.title("MinMaxScaler")
plt.xlabel("Montant transformé")
plt.ylabel("Fréquence")

plt.subplot(1, 3, 2)
plt.hist(m_train_standard, bins=30)
plt.title("StandardScaler")
plt.xlabel("Montant transformé")
plt.ylabel("Fréquence")

plt.subplot(1, 3, 3)
plt.hist(m_train_robust, bins=30)
plt.title("RobustScaler")
plt.xlabel("Montant transformé")
plt.ylabel("Fréquence")

plt.tight_layout()
plt.show()
#21 example numerique 
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
ex=[[100],[200],[300]]
scaler.fit(ex)
new=[[500]]
resultat=scaler.transform(new) 
print("montant original :",new[0][0])
print("montant transforme :",resultat[0][0]) # (500-100)/(300-100)=2

#PARTIE 4:
#22 desiquilibre y_train
print(y_train.value_counts())

print(y_train.value_counts(normalize=True) * 100)

eff=y_train.value_counts()
rapport=eff.loc[0]/eff.loc[1]#2194/206
print("le rapport entre les deaux classes :",rapport)

#23 randomoversampling
from imblearn.over_sampling import RandomOverSampler
ranov = RandomOverSampler(random_state=42)
x_train_over,y_train_over=ranov.fit_resample(x_train,y_train)

print("avant randomoversamling :")
print(y_train.value_counts())
print("apres randomoversampling :")
print(y_train_over.value_counts())

#24 randomundersampling
from imblearn.under_sampling import RandomUnderSampler
ranun=RandomUnderSampler(random_state=42)
x_train_under,y_train_under=ranun.fit_resample(x_train,y_train)

print("avant randomundersampling :")
print(y_train.value_counts())
print("apres randomundersamling :")
print(y_train_under.value_counts())

#24 nombre des lignes supprimees
eff_avant=y_train.value_counts()
eff_apres=y_train_under.value_counts()

print ("lignes restees :")
print(eff_apres)
print("lignes supprimees :")
print(eff_avant-eff_apres) #perte d'information 

#25 exemple
from imblearn.over_sampling import SMOTE

# Données
A = pd.DataFrame({
    "montant": [100,200, 300, 400, 500,600],
    "heure": [10, 20, 15, 12, 18, 22]
})

B = pd.Series([1, 1, 0, 0, 0, 0], name="fraude")

# SMOTE
smote = SMOTE(k_neighbors=1,random_state=42)

A_smote, B_smote = smote.fit_resample(A, B)

# Comparaison
print("Avant SMOTE :")
print(B.value_counts())

print("Après SMOTE :")
print(B_smote.value_counts())
print("Les nouvelles observations synthétiques :")
print(A_smote.iloc[len(A):])#new=A+lambda(B-A)

#q27
comparaison = pd.DataFrame({
    "Avant": y_train.value_counts(),
    "Apres RandomOverSampler": y_train_over.value_counts(),
    "Apres RandomUnderSampler": y_train_under.value_counts()
})

print(comparaison)

#PARTIE 5:
#q28
num_cols = ["montant_dh", "anciennete_mois", "nb_ops_24h", "heure"]

cat_cols = ["canal", "appareil", "region", "international"]

num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", RobustScaler())])

cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OneHotEncoder(handle_unknown="ignore"))])

prep = ColumnTransformer([("num", num_pipeline, num_cols),("cat", cat_pipeline, cat_cols)]) 

#q29
X_train_prepared = prep.fit_transform(x_train)
X_test_prepared = prep.transform(x_test)
X_train_prepared = pd.DataFrame( X_train_prepared,columns=prep.get_feature_names_out(), index=x_train.index)

X_test_prepared = pd.DataFrame( X_test_prepared,columns=prep.get_feature_names_out(), index=x_test.index)

print("Dimensions train :", X_train_prepared.shape)
print("Dimensions test :", X_test_prepared.shape)
#q29 valeurs manquantes:
print("Valeurs manquantes train :")
print(X_train_prepared.isnull().sum().sum())

print("Valeurs manquantes test :")
print(X_test_prepared.isnull().sum().sum())


#q30 
new_l=x_train.iloc[[0]].copy()
new_l["canal"]="whatsapp"
print(new_l)
new_l_prepared = prep.transform(new_l)
print(new_l_prepared)

#q31
import joblib
joblib.dump(prep,"prep.joblib")
prep2=joblib.load("prep.joblib")
echant=x_test.iloc[:5]
r1=prep.transform(echant)
r2=prep2.transform(echant)
print("les deux transformateurs donne les memes resultats : ",np.allclose(r1,r2))

#Bonus A

moy= x_train["montant_dh"].mean()
mediane = x_train["montant_dh"].median()

print("Moyenne :", moy)
print("Mediane :", mediane)


m_test = pd.concat([
    x_train["montant_dh"],
    pd.Series([x_train["montant_dh"].max() * 100])
])


print("Moyenne maintenant :", m_test.mean())
print("Mediane maintenant :", m_test.median())
