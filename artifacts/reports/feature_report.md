# Feature Distribution Report

## Numeric Features
### start_x
- count: 500
- mean ± std: 47.288 ± 24.192
- median (p5-p95): 46.132 (7.775–88.281)
- min / max: 0.768 / 104.825
- skewness: 0.146
- missing: 0.00%

### start_y
- count: 500
- mean ± std: 31.975 ± 19.964
- median (p5-p95): 31.464 (2.521–64.634)
- min / max: 0.000 / 68.000
- skewness: 0.132
- missing: 0.00%

### end_x
- count: 500
- mean ± std: 50.733 ± 24.615
- median (p5-p95): 50.111 (11.080–92.933)
- min / max: 2.238 / 104.144
- skewness: 0.074
- missing: 0.00%

### dist_to_goal
- count: 500
- mean ± std: 62.083 ± 21.523
- median (p5-p95): 62.110 (27.423–98.133)
- min / max: 6.873 / 107.644
- skewness: -0.003
- missing: 0.00%

### dist_to_goal_norm
- count: 500
- mean ± std: 0.591 ± 0.205
- median (p5-p95): 0.592 (0.261–0.935)
- min / max: 0.065 / 1.025
- skewness: -0.003
- missing: 0.00%

### delta_time_prev
- count: 500
- mean ± std: 2.587 ± 18.061
- median (p5-p95): 0.000 (0.000–0.166)
- min / max: 0.000 / 303.201
- skewness: 11.529
- missing: 0.00%

### speed
- count: 26
- mean ± std: 81.656 ± 404.822
- median (p5-p95): 1.274 (0.206–9.850)
- min / max: 0.069 / 2066.418
- skewness: 5.099
- missing: 94.80%

### speed_clipped
- count: 26
- mean ± std: 2.948 ± 4.323
- median (p5-p95): 1.274 (0.206–9.850)
- min / max: 0.069 / 20.000
- skewness: 2.877
- missing: 94.80%

### attack_progress_momentum
- count: 500
- mean ± std: 1.463 ± 10.113
- median (p5-p95): 0.000 (-12.283–19.126)
- min / max: -30.030 / 47.441
- skewness: 1.188
- missing: 0.00%

### dynamic_pressure_index
- count: 500
- mean ± std: 0.004 ± 0.031
- median (p5-p95): 0.000 (0.000–0.002)
- min / max: 0.000 / 0.528
- skewness: 12.136
- missing: 0.00%

### pitch_control_margin
- count: 500
- mean ± std: 0.019 ± 0.202
- median (p5-p95): 0.000 (-0.196–0.356)
- min / max: -0.853 / 1.324
- skewness: 1.951
- missing: 0.00%

### forward_control_ratio
- count: 500
- mean ± std: 0.263 ± 0.437
- median (p5-p95): 0.000 (0.000–1.000)
- min / max: 0.000 / 1.000
- skewness: 1.078
- missing: 0.00%

### pass_availability_score
- count: 500
- mean ± std: 0.644 ± 0.476
- median (p5-p95): 1.000 (0.000–1.000)
- min / max: 0.000 / 1.000
- skewness: -0.605
- missing: 0.00%

### off_ball_energy
- count: 500
- mean ± std: 0.015 ± 0.319
- median (p5-p95): 0.000 (0.000–0.001)
- min / max: 0.000 / 7.129
- skewness: 22.347
- missing: 0.00%

## Categorical Features
### type_filled
- missing: 0.00%
- coverage (top bucket share): 95.20%
- top categories:
  - Pass: 247
  - Carry: 118
  - Recovery: 34
  - Interception: 19
  - Duel: 19
  - Tackle: 9
  - Intervention: 9
  - Pass_Freekick: 8
  - Throw-In: 7
  - Cross: 6

### result_filled
- missing: 0.00%
- coverage (top bucket share): 100.00%
- top categories:
  - Successful: 255
  - NoResult: 192
  - Unsuccessful: 51
  - Blocked: 2

### event_context
- missing: 0.00%
- coverage (top bucket share): 93.40%
- top categories:
  - Pass__Successful: 210
  - Carry__NoResult: 118
  - Pass__Unsuccessful: 37
  - Recovery__NoResult: 34
  - Interception__NoResult: 19
  - Duel__Successful: 19
  - Intervention__NoResult: 9
  - Pass_Freekick__Successful: 8
  - Throw-In__Successful: 7
  - Tackle__Unsuccessful: 6

### prev_event_context
- missing: 0.00%
- coverage (top bucket share): 100.00%
- top categories:
  - START__START: 474
  - Pass__Successful: 10
  - Carry__NoResult: 5
  - Recovery__NoResult: 2
  - Duel__Successful: 2
  - Goal Kick__Successful: 2
  - Interception__NoResult: 2
  - Clearance__NoResult: 1
  - Block__NoResult: 1
  - Shot__Blocked: 1

### start_quadrant
- missing: 0.00%
- coverage (top bucket share): 100.00%
- top categories:
  - Q10: 137
  - Q11: 109
  - Q00: 90
  - Q01: 69
  - Q20: 52
  - Q21: 43

### end_quadrant
- missing: 0.00%
- coverage (top bucket share): 100.00%
- top categories:
  - Q10: 128
  - Q11: 112
  - Q00: 78
  - Q01: 64
  - Q20: 63
  - Q21: 55

### type_group
- missing: 0.00%
- coverage (top bucket share): 100.00%
- top categories:
  - Pass: 253
  - Carry: 118
  - Defensive: 75
  - Duel: 20
  - Other: 10
  - Restart: 9
  - SetPiece: 8
  - Keeper: 5
  - Shot: 2
