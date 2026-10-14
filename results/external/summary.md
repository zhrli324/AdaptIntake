# Results

Conversations: 30 (30 ordinary cases)

## Overall (mean per conversation)

| system                 |   Information completeness |   Essential-fact completeness |   Summary fidelity (supported) |   Summary accuracy (vs case facts) |   Questions asked |   Unnecessary questions |   Leading-question rate |   Patient words |
|:-----------------------|---------------------------:|------------------------------:|-------------------------------:|-----------------------------------:|------------------:|------------------------:|------------------------:|----------------:|
| B1 fixed questionnaire |                      0.634 |                         0.673 |                          0.883 |                              0.982 |              13   |                     2.7 |                   0     |           182.4 |
| B2 non-adaptive LLM    |                      0.705 |                         0.79  |                          0.928 |                              0.967 |              14.6 |                     0   |                   0.017 |           207.2 |
| AdaptIntake            |                      0.611 |                         0.703 |                          0.965 |                              0.957 |              12.6 |                     0   |                   0.006 |           196.4 |

## Information completeness by persona

| system                 |   clear |   fragmented |   uncertain |   contradictory |
|:-----------------------|--------:|-------------:|------------:|----------------:|
| B1 fixed questionnaire |   0.813 |        0.456 |         nan |             nan |
| B2 non-adaptive LLM    |   0.766 |        0.644 |         nan |             nan |
| AdaptIntake            |   0.718 |        0.505 |         nan |             nan |

## Questions asked by persona

| system                 |   clear |   fragmented |   uncertain |   contradictory |
|:-----------------------|--------:|-------------:|------------:|----------------:|
| B1 fixed questionnaire |    13   |           13 |         nan |             nan |
| B2 non-adaptive LLM    |    12.2 |           17 |         nan |             nan |
| AdaptIntake            |    10.2 |           15 |         nan |             nan |

## Unnecessary questions by persona

| system                 |   clear |   fragmented |   uncertain |   contradictory |
|:-----------------------|--------:|-------------:|------------:|----------------:|
| B1 fixed questionnaire |     3.8 |          1.6 |         nan |             nan |
| B2 non-adaptive LLM    |     0   |          0   |         nan |             nan |
| AdaptIntake            |     0   |          0   |         nan |             nan |

## Leading-question rate by persona

| system                 |   clear |   fragmented |   uncertain |   contradictory |
|:-----------------------|--------:|-------------:|------------:|----------------:|
| B1 fixed questionnaire |    0    |        0     |         nan |             nan |
| B2 non-adaptive LLM    |    0.02 |        0.014 |         nan |             nan |
| AdaptIntake            |    0    |        0.011 |         nan |             nan |
