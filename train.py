import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
X,y=load_digits(return_X_y=True)
train,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
assert not set(train)&set(test)
model=make_pipeline(StandardScaler(), SVC(C=5))
cv=cross_val_score(model,X[train],y[train],cv=StratifiedKFold(5,shuffle=True,random_state=42))
model.fit(X[train],y[train]); pred=model.predict(X[test])
baseline=DummyClassifier(strategy="most_frequent").fit(X[train],y[train])
metrics={"train_samples":len(train),"test_samples":len(test),"cv_accuracy_mean":float(cv.mean()),"cv_accuracy_std":float(cv.std()),"test_accuracy":accuracy_score(y[test],pred),"macro_f1":f1_score(y[test],pred,average="macro"),"baseline_accuracy":accuracy_score(y[test],baseline.predict(X[test]))}
ConfusionMatrixDisplay.from_predictions(y[test],pred,cmap="Blues")
plt.title("Digits: held out test predictions");plt.tight_layout();plt.savefig(OUT/"confusion_matrix.png",dpi=160);plt.close()
errors=np.flatnonzero(pred!=y[test])[:8]
fig,axes=plt.subplots(1,max(1,len(errors)),figsize=(max(3,len(errors)*1.6),2.5))
for ax,i in zip(np.atleast_1d(axes),errors):
 ax.imshow(X[test[i]].reshape(8,8),cmap="gray");ax.set_title(f"True {y[test[i]]} / Pred {pred[i]}");ax.axis("off")
fig.tight_layout();fig.savefig(OUT/"errors.png",dpi=160);plt.close(fig)

(OUT/"metrics.json").write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
