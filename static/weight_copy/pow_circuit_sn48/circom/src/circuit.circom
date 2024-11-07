pragma circom 2.0.0;

include "./circomlib/comparators.circom";
include "./circomlib/gates.circom";
include "./circomlib/bitify.circom";
include "./circomlib/mux1.circom";
include "./intdiv.circom";

template CalculateScore(n) {
    signal input actual_price[n];
    signal input predicted_price[n];
    signal input date_difference[n];
    signal input price_weight[n];
    signal input date_weight[n];
    signal output final_score[n];
    signal raw_date_points[n];
    signal diff1[n];
    signal diff2[n];
    signal price_score[n];

    component actualPricePositive[n];
    component predictedPricePositive[n];
    component dateDiffPositive[n];
    component lessThan14[n];
    component date_div[n];
    component isLessThan[n];
    component abs_diff[n];
    component price_div[n];
    component lessThan100[n];
    component price_weighted[n];
    component date_weighted[n];


    var max_points = 14;
    var SCALE = 1000000000;


    for (var i=0; i<n; i++) {

        lessThan14[i] = LessThan(64);
        lessThan14[i].in[0] <== date_difference[i];
        lessThan14[i].in[1] <== max_points;

        raw_date_points[i] <== (max_points - date_difference[i]) * lessThan14[i].out;

        date_div[i] = IntDiv(64);
        date_div[i].in[0] <== raw_date_points[i] * SCALE;
        date_div[i].in[1] <== max_points;

        isLessThan[i] = LessThan(64);
        isLessThan[i].in[0] <== predicted_price[i];
        isLessThan[i].in[1] <== actual_price[i];


        diff1[i] <== actual_price[i] - predicted_price[i];
        diff2[i] <== predicted_price[i] - actual_price[i];

        abs_diff[i] = Mux1();
        abs_diff[i].c[0] <== diff2[i];
        abs_diff[i].c[1] <== diff1[i];
        abs_diff[i].s <== isLessThan[i].out;

        price_div[i] = IntDiv(64);
        price_div[i].in[0] <== abs_diff[i].out * SCALE;
        price_div[i].in[1] <== actual_price[i];

        lessThan100[i] = LessThan(64);
        lessThan100[i].in[0] <== price_div[i].out;
        lessThan100[i].in[1] <== SCALE;

        price_score[i] <== (SCALE - price_div[i].out) * lessThan100[i].out;

        price_weighted[i] = IntDiv(64);
        price_weighted[i].in[0] <== price_score[i] * price_weight[i];
        price_weighted[i].in[1] <== 100;

        date_weighted[i] = IntDiv(64);
        date_weighted[i].in[0] <== date_div[i].out * date_weight[i];
        date_weighted[i].in[1] <== 100;

        final_score[i] <== price_weighted[i].out + date_weighted[i].out;
    }
}

component main {public [actual_price, predicted_price, date_difference, price_weight, date_weight]} = CalculateScore(1024);
