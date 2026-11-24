# Data

`pima_diabetes.csv` — Pima Indians Diabetes Database (768 rows, 8 numeric features + `Outcome`).

- Original study: Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988).
  Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.
  *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261–265.
- Copy downloaded from OpenML dataset 37 ("diabetes"): https://www.openml.org/d/37
  Columns were renamed (`preg, plas, pres, skin, insu, mass, pedi, age, class` →
  `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`)
  and `class` was encoded as `Outcome` (1 = tested_positive, 0 = tested_negative). No other changes.
- Population: females aged 21+ of Pima Indian heritage. `Glucose` is 2-hour plasma glucose from an
  oral glucose tolerance test (mg/dL). Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI
  are physiologically impossible and are treated as missing by `ihds/data.py`.

`clinical_knowledge.csv` — small hand-written knowledge base (paraphrased public guidance with the
source named on every row) used by the retrieval step in `ihds/workflow.py`. It is not clinical advice.
