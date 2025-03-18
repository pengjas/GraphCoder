module parallel_adder1x4(
    input [7:0] a0,
    input [7:0] a1,
    input [7:0] a2,
    input [7:0] a3,
    input [7:0] b0,
    input [7:0] b1,
    input [7:0] b2,
    input [7:0] b3,
    output [8:0] sum0,
    output [8:0] sum1,
    output [8:0] sum2,
    output [8:0] sum3
);
    // 4 AUs
    AU au0(.a(a0), .b(b0), .sum(sum0));
    AU au1(.a(a1), .b(b1), .sum(sum1));
    AU au2(.a(a2), .b(b2), .sum(sum2));
    AU au3(.a(a3), .b(b3), .sum(sum3));
endmodule

module AU(
    input [7:0] a,
    input [7:0] b,
    output [8:0] sum
);
    assign sum = a + b;
endmodule