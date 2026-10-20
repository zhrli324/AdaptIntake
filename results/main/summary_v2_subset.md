# Results

Conversations: 80 (80 ordinary cases)

## Overall (mean per conversation)

| system                 |   Information completeness |   Essential-fact completeness |   Summary fidelity (supported) |   Summary accuracy (vs case facts) |   Questions asked |   Unnecessary questions |   Leading-question rate |   Patient words |
|:-----------------------|---------------------------:|------------------------------:|-------------------------------:|-----------------------------------:|------------------:|------------------------:|------------------------:|----------------:|
| B1 fixed questionnaire |                      0.6   |                         0.559 |                          0.961 |                              0.966 |             12.75 |                     2.8 |                   0.012 |          235.3  |
| B2 non-adaptive LLM    |                      0.671 |                         0.716 |                          0.96  |                              0.961 |             13.75 |                     0   |                   0.017 |          253.45 |
| AdaptIntake            |                      0.704 |                         0.7   |                          0.965 |                              0.961 |             12.5  |                     0   |                   0     |          269.15 |
| AdaptIntake v2         |                      0.717 |                         0.79  |                          0.944 |                              0.944 |             13.4  |                     0   |                   0     |          275.85 |

## Information completeness by persona

| system                 |   clear |   fragmented |   uncertain |   contradictory |
|:-----------------------|--------:|-------------:|------------:|----------------:|
| B1 fixed questionnaire |   0.841 |        0.3   |       0.474 |           0.787 |
| B2 non-adaptive LLM    |   0.865 |        0.454 |       0.628 |           0.738 |
| AdaptIntake            |   0.918 |        0.52  |       0.554 |           0.825 |
| AdaptIntake v2         |   0.855 |        0.546 |       0.642 |           0.823 |

## Questions asked by persona

| system                 |   clear |   fragmented |   uncertain |   contradictory |
|:-----------------------|--------:|-------------:|------------:|----------------:|
| B1 fixed questionnaire |    13   |         13   |        13   |            12   |
| B2 non-adaptive LLM    |    11.4 |         16.4 |        14.4 |            12.8 |
| AdaptIntake            |     8.8 |         16.2 |        14   |            11   |
| AdaptIntake v2         |    11.2 |         16.4 |        14.2 |            11.8 |

## Unnecessary questions by persona

| system                 |   clear |   fragmented |   uncertain |   contradictory |
|:-----------------------|--------:|-------------:|------------:|----------------:|
| B1 fixed questionnaire |       4 |          2.2 |           2 |               3 |
| B2 non-adaptive LLM    |       0 |          0   |           0 |               0 |
| AdaptIntake            |       0 |          0   |           0 |               0 |
| AdaptIntake v2         |       0 |          0   |           0 |               0 |

## Leading-question rate by persona

| system                 |   clear |   fragmented |   uncertain |   contradictory |
|:-----------------------|--------:|-------------:|------------:|----------------:|
| B1 fixed questionnaire |       0 |         0.05 |       0     |           0     |
| B2 non-adaptive LLM    |       0 |         0    |       0.032 |           0.035 |
| AdaptIntake            |       0 |         0    |       0     |           0     |
| AdaptIntake v2         |       0 |         0    |       0     |           0     |
