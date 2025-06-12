module as_module(
    input clk,
    input rst,
    input mode,
    input [31:0] op1,
    input [31:0] op2,
    output [31:0] result
);

wire [31:0] add_result;
wire [31:0] sub_result;

// Instantiating the adder module
adder u_adder (
    .a(op1),
    .b(op2),
    .sum(add_result)
);

// Instantiating the subtractor module
subtractor u_subtractor (
    .a(op1),
    .b(op2),
    .difference(sub_result)
);

// Selecting output based on mode
assign result = (mode == 0) ? add_result : sub_result;

endmodule

// Adder module
module adder(
    input [31:0] a,
    input [31:0] b,
    output [31:0] sum
);
    assign sum = a + b;
endmodule

// Subtractor module
module subtractor(
    input [31:0] a,
    input [31:0] b,
    output [31:0] difference
);
    assign difference = a - b;
endmodule
