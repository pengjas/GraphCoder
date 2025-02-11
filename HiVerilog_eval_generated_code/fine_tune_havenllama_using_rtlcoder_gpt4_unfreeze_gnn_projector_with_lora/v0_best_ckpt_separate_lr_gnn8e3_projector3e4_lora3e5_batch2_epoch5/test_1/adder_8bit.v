module adder_8bit(
    input [7:0] a, b, 
    input cin, 
    output [7:0] sum, 
    output cout);

    wire [7:0] sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;

    assign sum1 = a[0] + b[0] + cin;
    assign sum2 = a[1] + b[1] + sum1[0];
    assign sum3 = a[2] + b[2] + sum2[0];
    assign sum4 = a[3] + b[3] + sum3[0];
    assign sum5 = a[4] + b[4] + sum4[0];
    assign sum6 = a[5] + b[5] + sum5[0];
    assign sum7 = a[6] + b[6] + sum6[0];
    assign sum8 = a[7] + b[7] + sum7[0];
    assign sum = {sum8[0], sum7, sum6, sum5, sum4, sum3, sum2, sum1};
    assign cout = sum8[7];

endmodule