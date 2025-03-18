`define ADD_OP 2'b00
`define SUB_OP 2'b01
`define AND_OP 2'b10
`define OR_OP 2'b11

module simple_alu4bit(
    input clk,
    input [1:0] op_code,
    input [3:0] operand_a,
    input [3:0] operand_b,
    output reg [3:0] result
);
wire [3:0] add_result;
wire [3:0] sub_result;
wire [3:0] and_result;
wire [3:0] or_result;

adder_stage adder(.operand_a(operand_a), .operand_b(operand_b), .result(add_result));
subtractor_stage subtractor(.operand_a(operand_a), .operand_b(operand_b), .result(sub_result));
and_stage and_logical(.operand_a(operand_a), .operand_b(operand_b), .result(and_result));
or_stage or_logical(.operand_a(operand_a), .operand_b(operand_b), .result(or_result));

always @(posedge clk) begin
    case(op_code)
        `ADD_OP: result <= add_result;
        `SUB_OP: result <= sub_result;
        `AND_OP: result <= and_result;
        `OR_OP: result <= or_result;
        default: result <= 0;
    endcase
end
endmodule

module adder_stage(
    input [3:0] operand_a,
    input [3:0] operand_b,
    output [3:0] result
);
    assign result = operand_a + operand_b;
endmodule

module subtractor_stage(
    input [3:0] operand_a,
    input [3:0] operand_b,
    output [3:0] result
);
    assign result = operand_a - operand_b;
endmodule

module and_stage(
    input [3:0] operand_a,
    input [3:0] operand_b,
    output [3:0] result
);
    assign result = operand_a & operand_b;
endmodule

module or_stage(
    input [3:0] operand_a,
    input [3:0] operand_b,
    output [3:0] result
);
    assign result = operand_a | operand_b;
endmodule