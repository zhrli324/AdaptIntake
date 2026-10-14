# Results

Conversations: 126 (120 ordinary cases)

## Overall (mean per conversation)

| system                 |   Information completeness |   Essential-fact completeness |   Summary fidelity (supported) |   Summary accuracy (vs case facts) |   Questions asked |   Unnecessary questions |   Leading-question rate |   Patient words |
|:-----------------------|---------------------------:|------------------------------:|-------------------------------:|-----------------------------------:|------------------:|------------------------:|------------------------:|----------------:|
| B1 fixed questionnaire |                      0.634 |                         0.59  |                          0.937 |                              0.95  |            12.875 |                   2.975 |                   0.025 |          247.75 |
| B2 non-adaptive LLM    |                      0.645 |                         0.686 |                          0.968 |                              0.967 |            14.075 |                   0     |                   0.017 |          261.8  |
| AdaptIntake            |                      0.735 |                         0.746 |                          0.973 |                              0.969 |            12.7   |                   0     |                   0.006 |          280.25 |

## Information completeness by persona

| system                 |   clear |   fragmented |   uncertain |   contradictory |
|:-----------------------|--------:|-------------:|------------:|----------------:|
| B1 fixed questionnaire |   0.856 |        0.356 |       0.523 |           0.801 |
| B2 non-adaptive LLM    |   0.83  |        0.474 |       0.576 |           0.698 |
| AdaptIntake            |   0.945 |        0.55  |       0.601 |           0.841 |

## Questions asked by persona

| system                 |   clear |   fragmented |   uncertain |   contradictory |
|:-----------------------|--------:|-------------:|------------:|----------------:|
| B1 fixed questionnaire |    13   |         13   |        13   |            12.5 |
| B2 non-adaptive LLM    |    11.6 |         16.6 |        15.2 |            12.9 |
| AdaptIntake            |     9.1 |         16.5 |        14   |            11.2 |

## Unnecessary questions by persona

| system                 |   clear |   fragmented |   uncertain |   contradictory |
|:-----------------------|--------:|-------------:|------------:|----------------:|
| B1 fixed questionnaire |       4 |          2.3 |         2.3 |             3.3 |
| B2 non-adaptive LLM    |       0 |          0   |         0   |             0   |
| AdaptIntake            |       0 |          0   |         0   |             0   |

## Leading-question rate by persona

| system                 |   clear |   fragmented |   uncertain |   contradictory |
|:-----------------------|--------:|-------------:|------------:|----------------:|
| B1 fixed questionnaire |   0.017 |        0.05  |       0.017 |           0.017 |
| B2 non-adaptive LLM    |   0.01  |        0.006 |       0.024 |           0.027 |
| AdaptIntake            |   0.012 |        0.005 |       0.007 |           0     |

## Red-flag cases

| system                 |   red_flag_detected |   red_flag_turn |
|:-----------------------|--------------------:|----------------:|
| B1 fixed questionnaire |                   1 |               1 |
| B2 non-adaptive LLM    |                   1 |               1 |
| AdaptIntake            |                   1 |               1 |
