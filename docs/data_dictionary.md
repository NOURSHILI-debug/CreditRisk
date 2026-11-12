# Data dictionary

## application_train.csv

- Path: `data/raw/application_train.csv`
- Rows: `307,511`
- Columns: `122`
- Target: `TARGET`
- Target counts: `0 = 282,686` (91.93%), `1 = 24,825` (8.07%)
- Imbalance ratio: `11.39:1` (negative:positive)

| Column | Type | Description | Missing | Notes |
|---|---|---|---:|---|
| `SK_ID_CURR` | integer | ID of the current loan application. | 0.00% | ID; excluded from training |
| `TARGET` | integer | 1 = repayment difficulty, 0 = no defined repayment difficulty. | 0.00% | Target; excluded from features; leakage |
| `NAME_CONTRACT_TYPE` | string | Loan contract type (cash or revolving). | 0.00% | Categorical |
| `CODE_GENDER` | string | Applicant gender. | 0.00% | Categorical |
| `FLAG_OWN_CAR` | string | Flag for car ownership. | 0.00% | Binary flag |
| `FLAG_OWN_REALTY` | string | Flag for property ownership. | 0.00% | Binary flag |
| `CNT_CHILDREN` | integer | Number of children. | 0.00% |  |
| `AMT_INCOME_TOTAL` | number | Total applicant income. | 0.00% |  |
| `AMT_CREDIT` | number | Requested credit amount. | 0.00% |  |
| `AMT_ANNUITY` | number | Loan annuity amount. | 0.00% |  |
| `AMT_GOODS_PRICE` | number | Price of goods financed by the loan. | 0.09% |  |
| `NAME_TYPE_SUITE` | string | Person accompanying the applicant. | 0.42% | Categorical |
| `NAME_INCOME_TYPE` | string | Applicant income type. | 0.00% | Categorical |
| `NAME_EDUCATION_TYPE` | string | Applicant education level. | 0.00% | Categorical |
| `NAME_FAMILY_STATUS` | string | Applicant family status. | 0.00% | Categorical |
| `NAME_HOUSING_TYPE` | string | Applicant housing type. | 0.00% | Categorical |
| `REGION_POPULATION_RELATIVE` | number | Normalized population of the applicant's region. | 0.00% |  |
| `DAYS_BIRTH` | integer | Applicant age in days at application (negative). | 0.00% | Date offset; negative = before application |
| `DAYS_EMPLOYED` | integer | Days employed at application (negative). | 0.00% | Date offset; negative = before application; 365243 = unemployed sentinel |
| `DAYS_REGISTRATION` | number | Days since registration change (negative). | 0.00% | Date offset; negative = before application |
| `DAYS_ID_PUBLISH` | integer | Days since ID document change (negative). | 0.00% | Date offset; negative = before application |
| `OWN_CAR_AGE` | number | Age of the applicant's car in years. | 65.99% | High missing |
| `FLAG_MOBIL` | integer | Flag for mobile phone provided. | 0.00% | Binary flag |
| `FLAG_EMP_PHONE` | integer | Flag for employer phone provided. | 0.00% | Binary flag |
| `FLAG_WORK_PHONE` | integer | Flag for work phone provided. | 0.00% | Binary flag |
| `FLAG_CONT_MOBILE` | integer | Flag for contact mobile phone provided. | 0.00% | Binary flag |
| `FLAG_PHONE` | integer | Flag for home phone provided. | 0.00% | Binary flag |
| `FLAG_EMAIL` | integer | Flag for email provided. | 0.00% | Binary flag |
| `OCCUPATION_TYPE` | string | Applicant occupation. | 31.35% | Categorical |
| `CNT_FAM_MEMBERS` | number | Number of family members. | 0.00% |  |
| `REGION_RATING_CLIENT` | integer | Region rating where the applicant lives (1-3). | 0.00% |  |
| `REGION_RATING_CLIENT_W_CITY` | integer | Region rating including the city (1-3). | 0.00% |  |
| `WEEKDAY_APPR_PROCESS_START` | string | Weekday of application processing. | 0.00% | Categorical |
| `HOUR_APPR_PROCESS_START` | integer | Hour of application processing. | 0.00% |  |
| `REG_REGION_NOT_LIVE_REGION` | integer | Flag: registration region differs from living region. | 0.00% | Binary flag |
| `REG_REGION_NOT_WORK_REGION` | integer | Flag: registration region differs from work region. | 0.00% | Binary flag |
| `LIVE_REGION_NOT_WORK_REGION` | integer | Flag: living region differs from work region. | 0.00% | Binary flag |
| `REG_CITY_NOT_LIVE_CITY` | integer | Flag: registration city differs from living city. | 0.00% | Binary flag |
| `REG_CITY_NOT_WORK_CITY` | integer | Flag: registration city differs from work city. | 0.00% | Binary flag |
| `LIVE_CITY_NOT_WORK_CITY` | integer | Flag: living city differs from work city. | 0.00% | Binary flag |
| `ORGANIZATION_TYPE` | string | Type of organization where the applicant works. | 0.00% | Categorical |
| `EXT_SOURCE_1` | number | Normalized score from external source 1. | 56.38% | High missing |
| `EXT_SOURCE_2` | number | Normalized score from external source 2. | 0.21% |  |
| `EXT_SOURCE_3` | number | Normalized score from external source 3. | 19.83% |  |
| `APARTMENTS_AVG` | number | Normalized apartment area, average. | 50.75% | High missing |
| `BASEMENTAREA_AVG` | number | Normalized basement area, average. | 58.52% | High missing |
| `YEARS_BEGINEXPLUATATION_AVG` | number | Normalized building exploitation age, average. | 48.78% |  |
| `YEARS_BUILD_AVG` | number | Normalized building age, average. | 66.50% | High missing |
| `COMMONAREA_AVG` | number | Normalized common area, average. | 69.87% | High missing |
| `ELEVATORS_AVG` | number | Normalized elevator feature, average. | 53.30% | High missing |
| `ENTRANCES_AVG` | number | Normalized entrance feature, average. | 50.35% | High missing |
| `FLOORSMAX_AVG` | number | Normalized maximum floor, average. | 49.76% |  |
| `FLOORSMIN_AVG` | number | Normalized minimum floor, average. | 67.85% | High missing |
| `LANDAREA_AVG` | number | Normalized land area, average. | 59.38% | High missing |
| `LIVINGAPARTMENTS_AVG` | number | Normalized living apartment area, average. | 68.35% | High missing |
| `LIVINGAREA_AVG` | number | Normalized living area, average. | 50.19% | High missing |
| `NONLIVINGAPARTMENTS_AVG` | number | Normalized non-living apartment area, average. | 69.43% | High missing |
| `NONLIVINGAREA_AVG` | number | Normalized non-living area, average. | 55.18% | High missing |
| `APARTMENTS_MODE` | number | Normalized apartment area, mode. | 50.75% | High missing |
| `BASEMENTAREA_MODE` | number | Normalized basement area, mode. | 58.52% | High missing |
| `YEARS_BEGINEXPLUATATION_MODE` | number | Normalized building exploitation age, mode. | 48.78% |  |
| `YEARS_BUILD_MODE` | number | Normalized building age, mode. | 66.50% | High missing |
| `COMMONAREA_MODE` | number | Normalized common area, mode. | 69.87% | High missing |
| `ELEVATORS_MODE` | number | Normalized elevator feature, mode. | 53.30% | High missing |
| `ENTRANCES_MODE` | number | Normalized entrance feature, mode. | 50.35% | High missing |
| `FLOORSMAX_MODE` | number | Normalized maximum floor, mode. | 49.76% |  |
| `FLOORSMIN_MODE` | number | Normalized minimum floor, mode. | 67.85% | High missing |
| `LANDAREA_MODE` | number | Normalized land area, mode. | 59.38% | High missing |
| `LIVINGAPARTMENTS_MODE` | number | Normalized living apartment area, mode. | 68.35% | High missing |
| `LIVINGAREA_MODE` | number | Normalized living area, mode. | 50.19% | High missing |
| `NONLIVINGAPARTMENTS_MODE` | number | Normalized non-living apartment area, mode. | 69.43% | High missing |
| `NONLIVINGAREA_MODE` | number | Normalized non-living area, mode. | 55.18% | High missing |
| `APARTMENTS_MEDI` | number | Normalized apartment area, median. | 50.75% | High missing |
| `BASEMENTAREA_MEDI` | number | Normalized basement area, median. | 58.52% | High missing |
| `YEARS_BEGINEXPLUATATION_MEDI` | number | Normalized building exploitation age, median. | 48.78% |  |
| `YEARS_BUILD_MEDI` | number | Normalized building age, median. | 66.50% | High missing |
| `COMMONAREA_MEDI` | number | Normalized common area, median. | 69.87% | High missing |
| `ELEVATORS_MEDI` | number | Normalized elevator feature, median. | 53.30% | High missing |
| `ENTRANCES_MEDI` | number | Normalized entrance feature, median. | 50.35% | High missing |
| `FLOORSMAX_MEDI` | number | Normalized maximum floor, median. | 49.76% |  |
| `FLOORSMIN_MEDI` | number | Normalized minimum floor, median. | 67.85% | High missing |
| `LANDAREA_MEDI` | number | Normalized land area, median. | 59.38% | High missing |
| `LIVINGAPARTMENTS_MEDI` | number | Normalized living apartment area, median. | 68.35% | High missing |
| `LIVINGAREA_MEDI` | number | Normalized living area, median. | 50.19% | High missing |
| `NONLIVINGAPARTMENTS_MEDI` | number | Normalized non-living apartment area, median. | 69.43% | High missing |
| `NONLIVINGAREA_MEDI` | number | Normalized non-living area, median. | 55.18% | High missing |
| `FONDKAPREMONT_MODE` | string | Building renovation fund mode (unclear). | 68.39% | Categorical; High missing |
| `HOUSETYPE_MODE` | string | Housing type mode. | 50.18% | Categorical; High missing |
| `TOTALAREA_MODE` | number | Total area mode. | 48.27% |  |
| `WALLSMATERIAL_MODE` | string | Wall material mode. | 50.84% | Categorical; High missing |
| `EMERGENCYSTATE_MODE` | string | Emergency state mode. | 47.40% | Categorical |
| `OBS_30_CNT_SOCIAL_CIRCLE` | number | Observed social circle members with 30-day overdue. | 0.33% |  |
| `DEF_30_CNT_SOCIAL_CIRCLE` | number | Defaulted social circle members with 30-day overdue. | 0.33% |  |
| `OBS_60_CNT_SOCIAL_CIRCLE` | number | Observed social circle members with 60-day overdue. | 0.33% |  |
| `DEF_60_CNT_SOCIAL_CIRCLE` | number | Defaulted social circle members with 60-day overdue. | 0.33% |  |
| `DAYS_LAST_PHONE_CHANGE` | number | Days since last phone change (negative). | 0.00% | Date offset; negative = before application |
| `FLAG_DOCUMENT_2` | integer | Flag for document 2 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_3` | integer | Flag for document 3 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_4` | integer | Flag for document 4 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_5` | integer | Flag for document 5 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_6` | integer | Flag for document 6 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_7` | integer | Flag for document 7 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_8` | integer | Flag for document 8 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_9` | integer | Flag for document 9 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_10` | integer | Flag for document 10 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_11` | integer | Flag for document 11 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_12` | integer | Flag for document 12 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_13` | integer | Flag for document 13 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_14` | integer | Flag for document 14 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_15` | integer | Flag for document 15 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_16` | integer | Flag for document 16 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_17` | integer | Flag for document 17 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_18` | integer | Flag for document 18 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_19` | integer | Flag for document 19 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_20` | integer | Flag for document 20 provided. | 0.00% | Binary flag |
| `FLAG_DOCUMENT_21` | integer | Flag for document 21 provided. | 0.00% | Binary flag |
| `AMT_REQ_CREDIT_BUREAU_HOUR` | number | Credit bureau enquiries in the last hour. | 13.50% |  |
| `AMT_REQ_CREDIT_BUREAU_DAY` | number | Credit bureau enquiries in the last day. | 13.50% |  |
| `AMT_REQ_CREDIT_BUREAU_WEEK` | number | Credit bureau enquiries in the last week. | 13.50% |  |
| `AMT_REQ_CREDIT_BUREAU_MON` | number | Credit bureau enquiries in the last month. | 13.50% |  |
| `AMT_REQ_CREDIT_BUREAU_QRT` | number | Credit bureau enquiries in the last quarter. | 13.50% |  |
| `AMT_REQ_CREDIT_BUREAU_YEAR` | number | Credit bureau enquiries in the last year. | 13.50% |  |

### Notes for application_train.csv

- `SK_ID_CURR` is the applicant ID and is excluded from training.
- `TARGET` is the outcome label and is excluded from features.
- All other columns are application-time features. `TARGET` is the only leakage
  column in this file.
- `DAYS_*` columns are day offsets. Negative values are before the application.
- `DAYS_EMPLOYED = 365243` is the unemployed sentinel.

## installments_payments.csv

- Path: `data/raw/installments_payments.csv`
- Rows: `13,605,401`
- Columns: `8`

| Column | Type | Description | Missing | Notes |
|---|---|---|---:|---|
| `SK_ID_PREV` | integer | ID of the previous credit. | 0.00% | ID; excluded from training |
| `SK_ID_CURR` | integer | ID of the current application; links to application_train. | 0.00% | ID; excluded from training |
| `NUM_INSTALMENT_VERSION` | number | Installment calendar version. | 0.00% | Categorical/ordinal |
| `NUM_INSTALMENT_NUMBER` | integer | Installment number in the payment schedule. | 0.00% |  |
| `DAYS_INSTALMENT` | number | Scheduled installment date, days relative to application (negative = before). | 0.00% | Date offset; possible leakage if after application |
| `DAYS_ENTRY_PAYMENT` | number | Actual payment date, days relative to application (negative = before). | 0.02% | Date offset; possible leakage if after application |
| `AMT_INSTALMENT` | number | Scheduled installment amount. | 0.00% | Leakage risk if record is after application |
| `AMT_PAYMENT` | number | Amount actually paid. | 0.02% | Leakage risk if record is after application |

### Notes for installments_payments.csv

- `SK_ID_PREV` is the previous-credit ID and is excluded from training.
- `SK_ID_CURR` is the link to `application_train.csv` and is excluded from
  training.
- `DAYS_INSTALMENT` and `DAYS_ENTRY_PAYMENT` are date offsets. In this file the
  maximum value is `-1`, so all records are before the application date.
  Records with positive values would be after the application and must be
  excluded for the LSTM.
- `AMT_INSTALMENT` and `AMT_PAYMENT` must also be limited to records before the
  application date.

## How the files link

Both files use `SK_ID_CURR`.

- `291,643` application IDs have installment records.
- `15,868` application IDs have no installment records.
- `47,944` installment `SK_ID_CURR` values are not in
  `application_train.csv`.

## Columns excluded from training

| Dataset | Column | Why |
|---|---|---|
| application_train | `SK_ID_CURR` | Identifier, not a feature. |
| application_train | `TARGET` | Outcome label; leakage. |
| installments_payments | `SK_ID_PREV` | Identifier, not a feature. |
| installments_payments | `SK_ID_CURR` | Identifier and link key. |

For the LSTM, also exclude any installment record dated after the application.
