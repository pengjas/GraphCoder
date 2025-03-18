module alu(
    input clk,
    input op_sel,
    input [31:0] a,
    input [31:0] b,
    output [31:0] result
);

// Intermediate wires for submodule results
wire [31:0] add_result;
wire [31:0] and_result;

// Instantiating the adder module
adder u_adder (
    .a(a),
    .b(b),
    .result(add_result)
);

// Instantiating the AND module
bitwise_and u_bitwise_and (
    .a(a),
    .b(b),
    .result(and_result)
);

// Selecting the result based on operation select input
assign result = (op_sel == 0) ? add_result : and_result;

endmodule

// Adder module
module adder(
    input [31:0] a,
    input [31:0] b,
    output [31:0] result
);
    assign result = a + b;
endmodule

// Bitwise AND module
module bitwise_and(
    input [31:0] a,
    input [31:0] b,
    output [31:0] result
);
    assign result = a & b;
endmodule
