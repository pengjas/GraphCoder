module ds(
    input clk,
    input rst,
    input [31:0] x,
    input [31:0] y,
    input [31:0] z,
    output [31:0] quotient,
    output [31:0] difference
);

// Intermediate wires for outputs of submodules
wire [31:0] division_result;
wire [31:0] subtraction_result;

// Instantiating the divider module
divider u_divider (
    .x(x),
    .y(y),
    .quotient(division_result)
);

// Instantiating the subtractor module
subtractor u_subtractor (
    .a(division_result),
    .b(z),
    .result(subtraction_result)
);

// Assigning the results to the outputs
assign quotient = division_result;
assign difference = subtraction_result;

endmodule

// Divider module
module divider(
    input [31:0] x,
    input [31:0] y,
    output [31:0] quotient
);
    assign quotient = x / y;
endmodule

// Subtractor module
module subtractor(
    input [31:0] a,
    input [31:0] b,
    output [31:0] result
);
    assign result = a - b;
endmodule
