module dual_mode_arithmetic(
    input clk,
    input rst,
    input mode,
    input [15:0] a,
    input [15:0] b,
    output [15:0] result
);

// Intermediate wires to hold the addition and subtraction results
wire [15:0] add_result;
wire [15:0] sub_result;

// Instantiating the adder module
adder u_adder (
    .a(a),
    .b(b),
    .result(add_result)
);

// Instantiating the subtractor module
subtractor u_subtractor (
    .a(a),
    .b(b),
    .result(sub_result)
);

// Multiplexer to select output based on mode
assign result = mode ? sub_result : add_result;

endmodule

// Adder module
module adder(
    input [15:0] a,
    input [15:0] b,
    output [15:0] result
);
    assign result = a + b;
endmodule

// Subtractor module
module subtractor(
    input [15:0] a,
    input [15:0] b,
    output [15:0] result
);
    assign result = a - b;
endmodule