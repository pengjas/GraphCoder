module adder_32bit(
    input [31:0] X,    // First 32-bit input operand
    input [31:0] Y,    // Second 32-bit input operand
    output [31:0] Sum, // 32-bit output sum
    output Carry_out   // Carry output
);

    wire [3:0] carry;   // Internal carry wires between submodules

    // Instantiate the four 8-bit adders
    adder_8bit adder0 (
        .a(X[7:0]),
        .b(Y[7:0]),
        .cin(1'b0),
        .sum(Sum[7:0]),
        .cout(carry[0])
    );

    adder_8bit adder1 (
        .a(X[15:8]),
        .b(Y[15:8]),
        .cin(carry[0]),
        .sum(Sum[15:8]),
        .cout(carry[1])
    );

    adder_8bit adder2 (
        .a(X[23:16]),
        .b(Y[23:16]),
        .cin(carry[1]),
        .sum(Sum[23:16]),
        .cout(carry[2])
    );

    adder_8bit adder3 (
        .a(X[31:24]),
        .b(Y[31:24]),
        .cin(carry[2]),
        .sum(Sum[31:24]),
        .cout(Carry_out)
    );

endmodule

module adder_8bit(
    input [7:0] a,    // 8-bit part of the first operand
    input [7:0] b,    // 8-bit part of the second operand
    input cin,        // Carry input from the previous stage
    output [7:0] sum, // Sum output of this stage
    output cout       // Carry output of this stage
);

    // Logic to perform bit-wise addition
    assign {cout, sum} = a + b + cin;

endmodule
