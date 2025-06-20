`define ADD_OP 2'b00
`define SUB_OP 2'b01
`define MUL_OP 2'b10
`define DIV_OP 2'b11

module arithmetic_unit(
    input clk,
    input rst,
    input [1:0] opcode,
    input [7:0] data_a,
    input [7:0] data_b,
    output reg [15:0] result,
    output reg valid
);
wire [15:0] add_result;
wire [15:0] sub_result;
wire [15:0] mul_result;
wire [15:0] div_result;
wire div_valid;

adder add_module (.a(data_a), .b(data_b), .result(add_result));
subtractor sub_module (.a(data_a), .b(data_b), .result(sub_result));
multiplier mul_module (.a(data_a), .b(data_b), .result(mul_result));
divider div_module (.a(data_a), .b(data_b), .result(div_result), .valid(div_valid));

always @(posedge clk or posedge rst) begin
    if(rst) begin
        result <= 0;
        valid <= 1;
    end else begin
        case(opcode)
            `ADD_OP: result <= add_result;
            `SUB_OP: result <= sub_result;
            `MUL_OP: result <= mul_result;
            `DIV_OP: begin
                result <= div_result;
                valid <= div_valid;
            end
            default: begin
                result <= 0;
                valid <= 1;
            end
        endcase
    end
end
endmodule

// Arithmetic operation submodules
module adder(input [7:0] a, input [7:0] b, output [15:0] result);
assign result = a + b;
endmodule

module subtractor(input [7:0] a, input [7:0] b, output [15:0] result);
assign result = a - b;
endmodule

module multiplier(input [7:0] a, input [7:0] b, output [15:0] result);
assign result = a * b;
endmodule

module divider(input [7:0] a, input [7:0] b, output [15:0] result, output valid);
assign result = (b != 0) ? (a / b) : 0;
assign valid = (b != 0);
endmodule