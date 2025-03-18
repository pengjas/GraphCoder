module adder_32bit(
    input [31:0] A,
    input [31:0] B,
    input Cin,
    output [31:0] Sum,
    output Cout
);

    wire C1, C2, C3;

    adder_8bit add0 (
        .a(A[7:0]),
        .b(B[7:0]),
        .cin(Cin),
        .sum(Sum[7:0]),
        .cout(C1)
    );

    adder_8bit add1 (
        .a(A[15:8]),
        .b(B[15:8]),
        .cin(C1),
        .sum(Sum[15:8]),
        .cout(C2)
    );

    adder_8bit add2 (
        .a(A[23:16]),
        .b(B[23:16]),
        .cin(C2),
        .sum(Sum[23:16]),
        .cout(C3)
    );

    adder_8bit add3 (
        .a(A[31:24]),
        .b(B[31:24]),
        .cin(C3),
        .sum(Sum[31:24]),
        .cout(Cout)
    );

endmodule

module adder_8bit(
    input [7:0] a,
    input [7:0] b,
    input cin,
    output [7:0] sum,
    output cout
);

    assign {cout, sum} = a + b + cin;

endmodule
