module alu_module(
    input clk,
    input [31:0] a,
    input [31:0] b,
    input [1:0] op_sel,
    output reg [31:0] result
);

wire [31:0] add_result;
wire [31:0] sub_result;
wire [31:0] and_result;
wire [31:0] or_result;

// Instantiate the adder module
adder u_adder (
    .a(a),
    .b(b),
    .result(add_result)
);

// Instantiate the subtractor module
subtractor u_subtractor (
    .a(a),
    .b(b),
    .result(sub_result)
);

// Instantiate the and module
and_module u_and (
    .a(a),
    .b(b),
    .result(and_result)
);

// Instantiate the or module
or_module u_or (
    .a(a),
    .b(b),
    .result(or_result)
);

always @(posedge clk) begin
    case(op_sel)
        2'b00: result <= add_result;
        2'b01: result <= sub_result;
        2'b10: result <= and_result;
        2'b11: result <= or_result;
        default: result <= 32'b0;
    endcase
end

endmodule

// Adder module
module adder(
    input [31:0] a,
    input [31:0] b,
    output [31:0] result
);
    assign result = a + b;
endmodule

// Subtractor module
module subtractor(
    input [31:0] a,
    input [31:0] b,
    output [31:0] result
);
    assign result = a - b;
endmodule

// AND module
module and_module(
    input [31:0] a,
    input [31:0] b,
    output [31:0] result
);
    assign result = a & b;
endmodule

// OR module
module or_module(
    input [31:0] a,
    input [31:0] b,
    output [31:0] result
);
    assign result = a | b;
endmodule