module mult_4bit(X, Y, P);
    input [3:0] X;      // 4-bit input multiplicand X
    input [3:0] Y;      // 4-bit input multiplier Y
    output [7:0] P;     // 8-bit output product

    wire [3:0] partial_product_0, partial_product_1, partial_product_2, partial_product_3;
    wire [7:0] sum_0, sum_1, sum_2;

    // Generate partial products
    assign partial_product_0 = Y[0] ? X : 4'b0000;
    assign partial_product_1 = Y[1] ? X : 4'b0000;
    assign partial_product_2 = Y[2] ? X : 4'b0000;
    assign partial_product_3 = Y[3] ? X : 4'b0000;

    // Shift partial products
    wire [7:0] shift_partial_1 = {partial_product_1, 1'b0};
    wire [7:0] shift_partial_2 = {partial_product_2, 2'b00};
    wire [7:0] shift_partial_3 = {partial_product_3, 3'b000};

    // Add partial products
    adder_4bit adder_0(
        .A({4'b0000, partial_product_0}),
        .B(shift_partial_1),
        .D(sum_0)
    );
    adder_4bit adder_1(
        .A(sum_0),
        .B(shift_partial_2),
        .D(sum_1)
    );
    adder_4bit adder_2(
        .A(sum_1),
        .B(shift_partial_3),
        .D(P)
    );

endmodule

module adder_4bit(A, B, D);
    input [7:0] A;      // 8-bit input operand A
    input [7:0] B;      // 8-bit input operand B
    output [7:0] D;     // 8-bit output sum

    assign D = A + B;   // Simple bitwise addition

endmodule
