#pragma once

#define GATE_THRESHOLD  0.2600f
#define GATE_N_ACTIVE   3

const float GATE_MEANS[GATE_N_ACTIVE] = {
    59.79833657f,
    -0.16487492f,
    2.09280876f
};

const float GATE_STDS[GATE_N_ACTIVE] = {
    84.83441641f,
    1.22633182f,
    2.05930082f
};

const float GATE_COEF[GATE_N_ACTIVE] = {
    2.63566243f,
    1.47902191f,
    1.61449235f
};

const float GATE_INTERCEPT = -4.97501388f;
