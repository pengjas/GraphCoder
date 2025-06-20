`define ADD_OP 2'b00
`define SUB_OP 2'b01
`define AND_OP 2'b10
`define OR_OP 2'b11

module simple_alu(
    input clk,
    input rst,
    input [1:0] op_code,
    input [31:0] operand_a,
    input [31:0] operand_b,
    output reg [31:0] result
);

// Operation results
wire [31:0] res_add, res_sub, res_and, res_or;

// Instantiate operation modules
add_operation add(
    .a(operand_a),
    .b(operand_b),
    .result(res_add)
);

subtract_operation subtract(
    .a(operand_a),
    .b(operand_b),
    .result(res_sub)
);

and_operation bitwise_and(
    .a(operand_a),
    .b(operand_b),
    .result(res_and)
);

or_operation bitwise_or(
    .a(operand_a),
    .b(operand_b),
    .result(res_or)
);

always @(posedge clk) begin
    if (rst) begin
        result <= 0;
    end else begin
        case (op_code)
            `ADD_OP: result <= res_add;
            `SUB_OP: result <= res_sub;
            `AND_OP: result <= res_and;
            `OR_OP: result <= res_or;
            default: result <= 0;
        endcase
    end
end

endmodule

// Addition module
module add_operation(
    input [31:0] a,
    input [31:0] b,
    output [31:0] result
);
assign result = a + b;
endmodule

// Subtraction module
module subtract_operation(
    input [31:0] a,
    input [31:0] b,
    output [31:0] result
);
assign result = a - b;
endmodule

// AND operation module
module and_operation(
    input [31:0] a,
    input [31:0] b,
    output [31:0] result
);
assign result = a & b;
endmodule

// OR operation module
module or_operation(
    input [31:0] a,
    input [31:0] b,
    output [31:0] result
);
assign result = a | b;
endmodule